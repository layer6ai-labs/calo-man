import os
import math

from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from pytorch_fid import fid_score
from sklearn.metrics import accuracy_score, roc_auc_score

from ..datasets import undo_preprocessing_showers, undo_preprocessing_epl, preprocess, undo_preprocess_einc, preprocess_einc
from .high_level_features import HighLevelFeatures
from two_step_zoo.datasets.supervised_dataset import SupervisedDataset
from two_step_zoo.datasets.loaders import get_loader
from two_step_zoo.factory import get_discriminator


class InceptionHelper():
    def __init__(self, module, gt_loader, gen_samples, gen_batch_size) -> None:
        self.module = module
        self.gt_loader = gt_loader
        self.gen_samples = gen_samples
        self.gen_batch_size = gen_batch_size
        self.inception = fid_score.InceptionV3().to(module.device)

    def gen_loader(self):
        for i in range(0, self.gen_samples, self.gen_batch_size):
            if self.gen_samples - i < self.gen_batch_size:
                batch_size = self.gen_samples - i
            else:
                batch_size = self.gen_batch_size

            yield self.module.sample(batch_size), None, None

    def get_inception_features(self, im_loader=None):
        if im_loader:
            loader_len = len(self.gt_loader)
            loader_type = "ground truth"
        else:
            loader_len = self.gen_samples // self.gen_batch_size
            loader_type = "generated"
            im_loader = self.gen_loader()

        feats = []
        for batch, _, _ in tqdm(im_loader, desc=f"Getting {loader_type} features", leave=False, total=loader_len):
            batch = batch.to(self.module.device)

            # Convert grayscale to RGB
            if batch.ndim == 3:
                batch.unsqueeze_(1)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            with torch.no_grad():
                batch_feats = self.inception(batch)[0]

            batch_feats = batch_feats.squeeze().cpu().numpy()
            feats.append(batch_feats)

        return np.concatenate(feats)

    def compute_inception_stats(self, im_loader=None):
        # Compute mean and covariance for generated and ground truth iterables
        feats = self.get_inception_features(im_loader)
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)

        return mu, sigma


class HistogramHelper():
    _VALID_HISTOGRAMS = ["etot_einc", "e_layers", "ec_etas", "ec_phis", "ec_width_etas", "ec_width_phis"]

    def __init__(self, module, gt_loader, gen_samples, dataset_name, max_deposited_ratio,
                 raw_shape, energy_min, energy_max, normalized_deposited_energy, 
                 deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
                 conditional_on_epl, caloflow_epl, normalize_to_epl):
        self.module = module
        self.gt_loader = gt_loader
        self.gen_samples = gen_samples
        self.dataset_name = dataset_name

        self.cfg = {
            "dataset": self.dataset_name,
            "max_deposited_ratio": max_deposited_ratio,
            "raw_shape": raw_shape,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "normalized_deposited_energy": normalized_deposited_energy,
            "deposited_energy_per_layer": deposited_energy_per_layer,
            "logitspace_voxels": logitspace_voxels,
            "logspace_incident_energies": logspace_incident_energies,
            "conditional_on_epl": conditional_on_epl,
            "caloflow_epl": caloflow_epl,
            "normalize_to_epl": normalize_to_epl,
        }

        particle = {'photons1': 'photon', 'pions1': 'pion',
                    'electrons2': 'electron', 'electrons3': 'electron'}[dataset_name]
        xml_filename = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "binning", f"binning_dataset_{dataset_name}.xml")
        self.gt_hlf = HighLevelFeatures(particle, xml_filename)
        self.sample_hlf = HighLevelFeatures(particle, xml_filename)


    def get_histogram_counts(self, loader=None, e_layers_bin_fn=None, epl_module=None, epl_cfg=None):
        if e_layers_bin_fn is None:
            def e_layers_bin_fn(key): return 20  # e_layers with groundtruth loader should have 20 bins
        else:
            assert loader is None, \
                "Only need to supply `e_layers_bin_fn` when checking generated counts."

        if loader is None:
            print("Sampling showers for histogram counts metric")
            shower, e_inc = sample_showers(self.gt_loader, self.module, self.cfg, self.gen_samples,
                                           epl_module=epl_module, epl_cfg=epl_cfg)
            hlf = self.sample_hlf
        else:
            # Use entire data split
            x = loader.dataset.x
            y = loader.dataset.y
            shower, e_inc = undo_preprocessing_showers(x.cpu().numpy(), y.cpu().numpy(), self.cfg)
            shower = shower.reshape(shower.shape[0], -1) # flatten
            hlf = self.gt_hlf

        hlf.CalculateFeatures(shower)
        hlf.Einc = e_inc.squeeze()

        feature_counts_dict = {}
        for hist_type in self._VALID_HISTOGRAMS:
            if hist_type == "e_layers":
                bin_fn = e_layers_bin_fn
            else:
                bin_fn = self._get_bin_fn(hist_type)

            stat_dict = self._get_statistics_dict(hlf, hist_type)

            counts_dict = {}
            if hist_type == "e_layers": bin_dict = {}  # Need to save the bins in this case for later use

            for key, value in stat_dict.items():
                counts, bins = np.histogram(value, bins=bin_fn(key), density=True)
                counts_dict[key] = counts*np.diff(bins)  # Makes the counts normalized

                if hist_type == "e_layers": bin_dict[key] = bins

            if hist_type == "e_layers":  # Create the bin fn post-hoc after inferring the bins from the data in this case
                def e_layers_bin_fn(key): return bin_dict[key]

            feature_counts_dict[hist_type] = counts_dict

        return feature_counts_dict, e_layers_bin_fn

    def get_chisq_distances(self, gt_feature_counts_dict, sample_feature_counts_dict):
        chisq_distances = []
        chisq_dict = {}
        for hist_type, counts_dict in gt_feature_counts_dict.items():
            single_hist_chisq_distances = []
            for key, gt_counts in counts_dict.items():
                sample_counts = sample_feature_counts_dict[hist_type][key]

                # Calculate the separation power (assuming normalized counts) as in
                # https://github.com/CaloChallenge/homepage/blob/ea6f0a758bdadb814d5bfab49e36ac4423e44163/code/evaluate.py
                sum_of_squares = (sample_counts - gt_counts)**2
                normalized_sum_of_squares = sum_of_squares / (sample_counts + gt_counts + 1e-16)
                single_hist_chisq_distances.append(0.5 * normalized_sum_of_squares.sum())

            chisq_distance = np.mean(single_hist_chisq_distances)
            chisq_distances.append(chisq_distance)
            chisq_dict[hist_type] = chisq_distance

        chisq_dict["ave_histogram_difference"] = np.mean(chisq_distances)
        return chisq_dict

    def _get_bin_fn(self, hist_type):
        """Return a function mapping from some key to a set of bins for the histogram"""

        assert hist_type != "e_layers", "Do not specify bin_fn for e_layers. It should be inferred."

        if hist_type == "etot_einc":
            def bin_fn(key): return np.linspace(0.5, 1.5, 101)

        elif hist_type in ["ec_etas", "ec_phis"]:
            def bin_fn(key):
                if self.dataset_name in ["electrons2", "electrons3"]:
                    lim = (-30, 30)
                elif key in [12, 13]:
                    lim = (-500, 500)
                else:
                    lim = (-100, 100)
                return np.linspace(*lim, 101)

        elif hist_type in ["ec_width_etas", "ec_width_phis"]:
            def bin_fn(key):
                if self.dataset_name in ["electrons2", "electrons3"]:
                    lim = (0, 30)
                elif key in [12, 13]:
                    lim = (0, 400)
                else:
                    lim = (0, 100)
                return np.linspace(*lim, 101)

        else:
            raise ValueError(f"Unknown hist_type {hist_type}")

        return bin_fn

    def _get_statistics_dict(self, hlf, hist_type):
        """Return dict from key to statistic for which to build the histogram"""

        if hist_type == "etot_einc":
            stat_dict = {"null": hlf.GetEtot()/hlf.Einc}

        else:
            stat_dict = {
                "e_layers": hlf.GetElayers(),
                "ec_etas": hlf.GetECEtas(),
                "ec_phis": hlf.GetECPhis(),
                "ec_width_etas": hlf.GetWidthEtas(),
                "ec_width_phis": hlf.GetWidthPhis(),
            }[hist_type]

        return stat_dict


class EPLHistogramHelper():
    _VALID_HISTOGRAMS = ["etot_einc", "e_layers"]

    def __init__(self, module, gt_loader, gen_samples, dataset_name, max_deposited_ratio, caloflow_epl):
        self.module = module
        self.gt_loader = gt_loader
        self.gen_samples = gen_samples
        self.dataset_name = dataset_name

        self.cfg = {
            "dataset": self.dataset_name,
            "max_deposited_ratio": max_deposited_ratio,
            "caloflow_epl": caloflow_epl,
        }

        particle = {'photons1': 'photon', 'pions1': 'pion',
                    'electrons2': 'electron', 'electrons3': 'electron'}[dataset_name]
        xml_filename = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "binning", f"binning_dataset_{dataset_name}.xml")
        self.gt_hlf = HighLevelFeatures(particle, xml_filename)
        self.sample_hlf = HighLevelFeatures(particle, xml_filename)


    def get_histogram_counts(self, loader=None, e_layers_bin_fn=None):
        if e_layers_bin_fn is None:
            def e_layers_bin_fn(key): return 20  # e_layers with groundtruth loader should have 20 bins
        else:
            assert loader is None, \
                "Only need to supply `e_layers_bin_fn` when checking generated counts."

        if loader is None:
            print("Sampling eperlayer for EPL histogram counts metric")
            eperlayers, e_inc = sample_eperlayers(self.gt_loader, self.module, self.cfg, self.gen_samples)
            hlf = self.sample_hlf
        else:
            # Use entire data split
            x = loader.dataset.x
            y = loader.dataset.y
            eperlayers, e_inc = undo_preprocessing_epl(x.cpu().numpy(), y.cpu().numpy(), self.cfg)
            eperlayers = eperlayers.reshape(eperlayers.shape[0], -1) # flatten
            hlf = self.gt_hlf

        hlf.CalculateEPLFeatures(eperlayers)
        
        hlf.Einc = e_inc.squeeze()

        feature_counts_dict = {}
        for hist_type in self._VALID_HISTOGRAMS:
            if hist_type == "e_layers":
                bin_fn = e_layers_bin_fn
            else:
                bin_fn = self._get_bin_fn(hist_type)

            stat_dict = self._get_statistics_dict(hlf, hist_type)

            counts_dict = {}
            if hist_type == "e_layers": bin_dict = {}  # Need to save the bins in this case for later use

            for key, value in stat_dict.items():
                counts, bins = np.histogram(value, bins=bin_fn(key), density=True)
                counts_dict[key] = counts*np.diff(bins)  # Makes the counts normalized

                if hist_type == "e_layers": bin_dict[key] = bins

            if hist_type == "e_layers":  # Create the bin fn post-hoc after inferring the bins from the data in this case
                def e_layers_bin_fn(key): return bin_dict[key]

            feature_counts_dict[hist_type] = counts_dict

        return feature_counts_dict, e_layers_bin_fn

    def get_chisq_distances(self, gt_feature_counts_dict, sample_feature_counts_dict):
        chisq_distances = []
        chisq_dict = {}
        for hist_type, counts_dict in gt_feature_counts_dict.items():
            single_hist_chisq_distances = []
            for key, gt_counts in counts_dict.items():
                sample_counts = sample_feature_counts_dict[hist_type][key]

                # Calculate the separation power (assuming normalized counts) as in
                # https://github.com/CaloChallenge/homepage/blob/ea6f0a758bdadb814d5bfab49e36ac4423e44163/code/evaluate.py
                sum_of_squares = (sample_counts - gt_counts)**2
                normalized_sum_of_squares = sum_of_squares / (sample_counts + gt_counts + 1e-16)
                single_hist_chisq_distances.append(0.5 * normalized_sum_of_squares.sum())

            chisq_distance = np.mean(single_hist_chisq_distances)
            chisq_distances.append(chisq_distance)
            chisq_dict[hist_type] = chisq_distance

        chisq_dict["ave_epl_histogram_difference"] = np.mean(chisq_distances)
        return chisq_dict

    def _get_bin_fn(self, hist_type):
        """Return a function mapping from some key to a set of bins for the histogram"""

        assert hist_type != "e_layers", "Do not specify bin_fn for e_layers. It should be inferred."

        if hist_type == "etot_einc":
            def bin_fn(key): return np.linspace(0.5, 1.5, 101)

        else:
            raise ValueError(f"Unknown hist_type {hist_type}")

        return bin_fn

    def _get_statistics_dict(self, hlf, hist_type):
        """Return dict from key to statistic for which to build the histogram"""

        if hist_type == "etot_einc":
            stat_dict = {"null": hlf.GetEtot()/hlf.Einc}
        elif hist_type == "e_layers":
            stat_dict = hlf.GetElayers()
        else:
            raise ValueError(f"Unknown hist_type {hist_type}")

        return stat_dict


class ClassifierHelper():
    def __init__(self, module, loader, dataset_name, max_deposited_ratio, raw_shape,
                energy_min, energy_max, normalized_deposited_energy, 
                deposited_energy_per_layer, logitspace_voxels, logspace_incident_energies,
                conditional_on_epl, normalize_to_epl, epl_module, epl_cfg,
                data_size=100000, batch_size=1000, hidden_dims=[512,512,512], lr=5e-4,
                max_epochs=60, max_bad_valid_epochs=10):
        self.module = module
        self.loader = loader
        self.best_valid_acc = 0
        self.bad_valid_epochs = 0
        if max_bad_valid_epochs is None:
            self.max_bad_valid_epochs = math.inf
        else:
            self.max_bad_valid_epochs = max_bad_valid_epochs


        self.cfg = {
            "dataset": dataset_name,
            "max_deposited_ratio": max_deposited_ratio,
            "raw_shape": raw_shape,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "normalized_deposited_energy": normalized_deposited_energy,
            "deposited_energy_per_layer": deposited_energy_per_layer,
            "logitspace_voxels": logitspace_voxels,
            "logspace_incident_energies": logspace_incident_energies,
            "conditional_on_epl": conditional_on_epl,
            "normalize_to_epl": normalize_to_epl,
        }

        self.valid_test_fraction = 0.2
        # Get real data in raw format
        x = self.loader.dataset.x
        y = self.loader.dataset.y
        real_shower, real_e_inc = undo_preprocessing_showers(x.cpu().numpy(), y.cpu().numpy(), self.cfg)
        real_shower = real_shower.reshape(real_shower.shape[0], -1) # flatten
        real_data = np.concatenate((real_shower, real_e_inc), axis=1)
        perm = torch.randperm(real_data.shape[0])
        real_data = real_data[perm][:data_size]
        data_dim = real_data.shape[1]
        real_labels = np.zeros(real_data.shape[0])
        # Get generated data in raw format, already shuffled
        if data_size > real_data.shape[0]:
            print(f"Fewer than requested data points available, only using {real_data.shape[0]} each for real and fake in classifier metric")
        print("Sampling showers for classifier metric")
        gen_shower, gen_e_inc = sample_showers(self.loader, self.module, self.cfg, real_data.shape[0],
                                               epl_module=epl_module, epl_cfg=epl_cfg)
        gen_data = np.concatenate((gen_shower, gen_e_inc), axis=1)
        gen_labels = np.ones(gen_data.shape[0])

        # Create dataloaders for training classifier by mixing real/gen
        valid_size = int(self.valid_test_fraction * real_data.shape[0])
        valid_data = np.concatenate((real_data[:valid_size], gen_data[:valid_size]), axis=0)
        valid_data = torch.tensor(valid_data, dtype=torch.get_default_dtype())
        valid_labels = np.concatenate((real_labels[:valid_size], gen_labels[:valid_size]), axis=0)
        valid_labels = torch.tensor(valid_labels, dtype=torch.get_default_dtype())

        test_data = np.concatenate((real_data[valid_size:2*valid_size], gen_data[valid_size:2*valid_size]), axis=0)
        test_data = torch.tensor(test_data, dtype=torch.get_default_dtype())
        test_labels = np.concatenate((real_labels[valid_size:2*valid_size], gen_labels[valid_size:2*valid_size]), axis=0)
        test_labels = torch.tensor(test_labels, dtype=torch.get_default_dtype())

        train_data = np.concatenate((real_data[2*valid_size:], gen_data[2*valid_size:]), axis=0)
        train_data = torch.tensor(train_data, dtype=torch.get_default_dtype())
        train_labels = np.concatenate((real_labels[2*valid_size:], gen_labels[2*valid_size:]), axis=0)
        train_labels = torch.tensor(train_labels, dtype=torch.get_default_dtype())

        train_dset = SupervisedDataset("classifier", "train", train_data, train_labels)
        valid_dset = SupervisedDataset("classifier", "valid", valid_data, valid_labels)
        test_dset = SupervisedDataset("classifier", "test", test_data, test_labels)

        self.train_loader = get_loader(train_dset, batch_size, drop_last=True, pin_memory=True)
        self.valid_loader = get_loader(valid_dset, batch_size, drop_last=False, pin_memory=True)
        self.test_loader = get_loader(test_dset, batch_size, drop_last=False, pin_memory=True)

        # Create model and optimizer
        self.model_cfg = {
            "use_labels": False,
            "latent_dim": data_dim,
            "model": "classifier",
            "discriminator_hidden_dims": hidden_dims,
            "lr": lr,
            "max_epochs": max_epochs,
        }
        self.classifier = get_discriminator(self.model_cfg).to(self.module.device)
        self.opt = torch.optim.Adam(self.classifier.parameters(), lr=self.model_cfg["lr"])

    def train(self):
        for ep in range(self.model_cfg["max_epochs"]):
            pbar = _tqdm_progress_bar(
                iterable=enumerate(self.train_loader),
                desc="Training Classifier",
                length=len(self.train_loader),
                leave=True
            )
            self.classifier.train()
            epoch_loss = 0
            for _, (batch, y, _) in pbar:
                batch = batch.to(self.module.device)
                y = y.to(self.module.device)

                self.opt.zero_grad()
                predictions = self.classifier(batch)
                loss = binary_cross_entropy_with_logits(predictions.flatten(), y)
                loss.backward()
                self.opt.step()

                epoch_loss += loss
            print(f"Training loss at epoch {ep}: {epoch_loss}")

            self.classifier.eval()
            with torch.no_grad():
                valid_acc = self.evaluate_metric(accuracy_score, self.valid_loader, probabilities=False)
            print(f"Validation accuracy at epoch {ep}: {valid_acc}")
            if valid_acc > self.best_valid_acc:
                best_ep = ep
                self.best_valid_acc = valid_acc
                best_params = self.classifier.state_dict()
                self.bad_valid_epochs = 0
            else:
                self.bad_valid_epochs += 1
                if self.bad_valid_epochs >= self.max_bad_valid_epochs:
                    print(f"Stopping classifier early after {self.max_bad_valid_epochs} bad valid epochs")
                    break
            if valid_acc > 0.999:
                print(f"Stopping classifier early with perfect validation accuracy")
                break

        print(f"Evaluating AUC on best model from epoch {best_ep}")
        self.classifier.load_state_dict(best_params)
        with torch.no_grad():
            roc_auc = self.evaluate_metric(roc_auc_score, self.test_loader)

        return roc_auc

    def evaluate_metric(self, metric_fn, loader, probabilities=True):
        preds = None
        self.classifier.eval()
        for x, y, _ in loader:
            logits_batch = self.classifier(x.to(self.module.device), None)
            preds_batch = torch.sigmoid(logits_batch).cpu().numpy()
            if preds is None:
                preds = preds_batch
                labels = y
            else:
                preds = np.concatenate((preds, preds_batch), axis=0)
                labels = np.concatenate((labels, y), axis=0)
            if not probabilities:
                preds = np.where(preds > 0.5, 1.0, 0.0)
        return metric_fn(labels, preds)

def _tqdm_progress_bar(iterable, desc, length, leave):
        return tqdm(
            iterable,
            desc=desc,
            total=length,
            bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
            leave=leave
        )

def sample_showers(loader, module, cfg, num_samples, epl_module=None, epl_cfg=None):
    """
    Returns a numpy array containing a number `num_samples` of flat, generated showers
    in the format of the raw datasets. Preprocessing is undone after sampling if required.
    """
    if epl_module and epl_cfg:
        print(f"Sampling showers with epl_module and batch size {loader.batch_size}.")
    else:
        print("Sampling showers without epl_module")
    samples_created = 0
    conditional_energies = None
    showers = None
    while samples_created < num_samples:
        dataloader_iterator = enumerate(loader)
        try:
            j, (_, y, _) = next(dataloader_iterator)

            if samples_created + y.shape[0] > num_samples:
                y = y[:num_samples - samples_created]
            samples_created += y.shape[0]

            y = y.to(module.device)
            if epl_module and epl_cfg:
                # Take e_inc from y and generate eperlayer
                assert y.shape[1] > 1 # context should already have eperlayer which we are replacing
                e_inc = y[:, :1] # Keep dim by slicing for single element
                e_inc = undo_preprocess_einc(e_inc, cfg)
                epl_e_inc = preprocess_einc(e_inc, epl_cfg)
                eperlayer_batch = epl_module.sample(epl_e_inc.shape[0], epl_e_inc)
                # undo preprocess of the epl model and preprocess according to the two_step model
                eperlayers, _ = undo_preprocessing_epl(eperlayer_batch, epl_e_inc, epl_cfg)
                # use the e_inc from the first undoing to avoid multiple transformations
                eperlayers, e_inc = preprocess(eperlayers, e_inc, cfg)

                if len(e_inc.shape) < len(eperlayers.shape):
                    e_inc = torch.unsqueeze(e_inc, 1)
                y = torch.cat((e_inc, eperlayers), dim=1)
            try:
                shower_batch = module.sample(y.shape[0], y).cpu().numpy()
            except AttributeError:
                print("No sample method available")
                return None, None
            y = y.cpu().numpy()
            if conditional_energies is None:
                conditional_energies = y
                showers = shower_batch
            else:
                conditional_energies = np.concatenate((conditional_energies, y), axis=0)
                showers = np.concatenate((showers, shower_batch), axis=0)
        except (StopIteration):
            continue

    #NOTE: Sampling can generate values out of expected range for undo_preprocessing
    showers = np.clip(showers, module.data_min.cpu().numpy(), module.data_max.cpu().numpy())
    showers, conditional_energies = undo_preprocessing_showers(showers, conditional_energies, cfg)
    showers = showers.reshape(showers.shape[0], -1) # flatten

    return showers, conditional_energies


def sample_eperlayers(loader, module, cfg, num_samples):
    """
    Returns a numpy array containing a number `num_samples` of flat, generated energies per layer
    in the format of the raw datasets. Preprocessing is undone after sampling if required.
    Loader should contain incident energies as y.
    Module should take incident energy as context for sampling eperlayer.
    """
    samples_created = 0
    conditional_energies = None
    eperlayers = None
    dataloader_iterator = enumerate(loader)
    while samples_created < num_samples:
        try:
            j, (_, y, _) = next(dataloader_iterator)

            if samples_created + y.shape[0] > num_samples:
                y = y[:num_samples - samples_created]
            samples_created += y.shape[0]

            y = y.to(module.device)
            try:
                eperlayer_batch = module.sample(y.shape[0], y).cpu().numpy()
            except AttributeError:
                print("No sample method available")
                return None, None
            y = y.cpu().numpy()
            if conditional_energies is None:
                conditional_energies = y
                eperlayers = eperlayer_batch
            else:
                conditional_energies = np.concatenate((conditional_energies, y), axis=0)
                eperlayers = np.concatenate((eperlayers, eperlayer_batch), axis=0)
        except (StopIteration):
            continue

    #NOTE: Sampling can generate values out of expected range for undo_preprocessing
    eperlayers = np.clip(eperlayers, module.data_min.cpu().numpy(), module.data_max.cpu().numpy())
    eperlayers, conditional_energies = undo_preprocessing_epl(eperlayers, conditional_energies, cfg)
    eperlayers = eperlayers.reshape(eperlayers.shape[0], -1) # flatten

    return eperlayers, conditional_energies
