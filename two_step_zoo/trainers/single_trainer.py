import math
import io
import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.utils

from two_step_zoo.evaluators import NullEvaluator, sample_showers, sample_eperlayers
from two_step_zoo.evaluators.high_level_features import HighLevelFeatures
from two_step_zoo.evaluators.evaluate import evaluate_samples, evaluate_energies_per_layer, get_particle_xml


class BaseTrainer:
    """Base class for SingleTrainer and AlternatingEpochTrainer"""
    _STEPS_PER_LOSS_WRITE = 10

    def __init__(
            self,
            cfg,

            module, *,
            ckpt_prefix,

            train_loader,
            valid_loader,
            test_loader,

            writer,

            max_epochs,

            early_stopping_metric=None,
            max_bad_valid_epochs=None,
            max_grad_norm=None,

            evaluator=None,

            only_test=False,
    ):
        self.cfg = cfg
        self.module = module
        self.ckpt_prefix = ckpt_prefix

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.writer = writer

        self.max_epochs = max_epochs

        self.early_stopping_metric = early_stopping_metric
        self.max_grad_norm = max_grad_norm
        self.bad_valid_epochs = 0
        self.best_valid_loss = math.inf

        if max_bad_valid_epochs is None:
            self.max_bad_valid_epochs = math.inf
        else:
            self.max_bad_valid_epochs = max_bad_valid_epochs

        self.iteration = 0
        self.epoch = 0

        if evaluator is None:
            self.evaluator = NullEvaluator(
                module, valid_loader=valid_loader, test_loader=test_loader)
        else:
            self.evaluator = evaluator

        # assert early_stopping_metric is None or early_stopping_metric in evaluator.valid_metrics, \
            # f"Early stopping metric must be one of the validation metrics provided."

        self.only_test = only_test

    def train(self, standalone=False, second_step=False, epl_step=False):
        if self.only_test:
            with torch.no_grad():
                self._test()
            return

        self.update_transform_parameters()

        while self.epoch < self.max_epochs and self.bad_valid_epochs < self.max_bad_valid_epochs:
            self.module.train()

            self.train_for_epoch()
            with torch.no_grad():
                valid_loss = self._validate(second_step=second_step, epl_step=epl_step)

                if self.early_stopping_metric:
                    if valid_loss < self.best_valid_loss:
                        self.bad_valid_epochs = 0
                        self.best_valid_loss = valid_loss
                        self.write_checkpoint("best_valid")

                        print(f"Best validation loss of {valid_loss} achieved on epoch {self.epoch}")

                    else:
                        self.bad_valid_epochs += 1

                        if self.bad_valid_epochs == self.max_bad_valid_epochs:
                            print(f"No validation improvement for {self.max_bad_valid_epochs}"
                                    + " epochs. Training halted.")
                            self.write_checkpoint("latest")

                            self.load_checkpoint("best_valid")
                            self._test()
                            
                            if standalone:
                                print("Sampling showers to hdf5 for SingleTrainer")
                                self._sample_showers_to_hdf5()
                            if epl_step:
                                self._sample_eperlayer_to_hdf5()

                            return

            self.write_checkpoint("latest")

        if self.bad_valid_epochs < self.max_bad_valid_epochs:
            with torch.no_grad():
                self._test()
                if standalone:
                    print("Sampling showers to hdf5 for SingleTrainer")
                    self._sample_showers_to_hdf5()
                if epl_step:
                    self._sample_eperlayer_to_hdf5()
            print(f"Maximum epochs reached. Training of {self.module.model_type} complete.")

    def train_for_epoch(self):
        pbar = self._tqdm_progress_bar(
            iterable=enumerate(self.train_loader),
            desc="Training",
            length=len(self.train_loader),
            leave=True
        )
        for j, (batch, y, idx) in pbar:
            if not self.module.use_labels:
                y = None
            loss_dict = self.train_single_batch(batch, y)

            if j == 0:
                full_loss_dict = loss_dict
            else:
                for k in loss_dict.keys():
                    full_loss_dict[k] += loss_dict[k]

        self.epoch += 1
        self.update_transform_parameters()

        for k, v in full_loss_dict.items():
            print(f"{self.module.model_type} {k}: {v/j:.4f} after {self.epoch} epochs")

    def _test(self):
        # Disable for physics experiments
        # if len(self.module.data_shape) > 1: # If image data
        #     self.sample_and_record()
        test_results = self.evaluator.test()
        self.record_dict(self.ckpt_prefix + "_test", test_results, self.epoch, save=True)

    def _validate(self, second_step=False, epl_step=False):
        valid_results = self.evaluator.validate()
        self.record_dict("validate", valid_results, self.epoch)
        if (not second_step) and (not epl_step):
            print("Sampling showers to tensorboard for SingleTrainer validation")
            self._sample_showers_to_tensorboard(self.evaluator.valid_loader, num_samples=5000, epl_module=None, epl_cfg=None)

        return valid_results.get(self.early_stopping_metric)

    def update_transform_parameters(self):
        raise NotImplementedError("Define in child classes")

    def _tqdm_progress_bar(self, iterable, desc, length, leave):
        return tqdm(
            iterable,
            desc=desc,
            total=length,
            bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
            leave=leave
        )

    def write_scalar(self, tag, value, step):
        self.writer.write_scalar(f"{self.module.model_type}/{tag}", value, step)

    def sample_and_record(self):
        NUM_SAMPLES = 64
        GRID_ROWS = 8

        with torch.no_grad():
            try:
                imgs = self.module.sample(NUM_SAMPLES)
            except AttributeError:
                print("No sample method available")
                return

            imgs.clamp_(self.module.data_min, self.module.data_max)
            grid = torchvision.utils.make_grid(imgs, nrow=GRID_ROWS, pad_value=1, normalize=True, scale_each=True)
            grid_permuted = grid.permute((1,2,0))

            plt.figure()
            plt.axis("off")
            plt.imshow(grid_permuted.detach().cpu().numpy())

            self.writer.write_image("samples", grid, global_step=self.epoch)

    def _sample_showers_to_tensorboard(self, loader, num_samples=5000, epl_module=None, epl_cfg=None):
        if not self.module.use_labels:
            print("Cannot create conditional samples when labels are not provided")
            return

        try:
            showers, energies = sample_showers(loader, self.module, self.cfg, num_samples, epl_module=epl_module, epl_cfg=epl_cfg)
        except(TypeError):
            return
        if showers is None:
            return
        particle, xml_filename = get_particle_xml(self.cfg["dataset"])
        try:
            hlf = HighLevelFeatures(particle, xml_filename)
            fig = hlf.DrawAverageShower(showers, title="Shower average", show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            image = torchvision.transforms.ToTensor()(im)

            self.writer.write_image("Shower average", image, global_step=self.epoch)
            plt.close()
        except(ValueError):
            print("Failed to draw showers")

    def _sample_showers_to_hdf5(self, eval=True, epl_module=None, epl_cfg=None):
        if not self.module.use_labels:
            print("Cannot create conditional samples when labels are not provided")
            return
        num_samples = len(self.evaluator.test_loader.dataset)
        t0 = time.time()
        showers, conditional_energies = sample_showers(self.evaluator.test_loader, self.module, self.cfg, num_samples, epl_module=epl_module, epl_cfg=epl_cfg)
        t1 = time.time()
        if showers is None:
            return
        print(f"{num_samples} conditional samples generated in {t1 - t0} seconds")
        data_dict = {
            'incident_energies': conditional_energies,
            'showers': showers,
        }
        self.writer.write_hdf5('generated_showers', data_dict)
        
        if eval:
            evaluate_samples(showers, conditional_energies, self.cfg, self.writer.logdir)

    def _sample_eperlayer_to_hdf5(self):
        if not self.module.use_labels:
            print("Cannot create conditional samples when labels are not provided")
            return
        num_samples = len(self.evaluator.test_loader.dataset)
        t0 = time.time()
        eperlayers, conditional_energies = sample_eperlayers(self.evaluator.test_loader, self.module, self.cfg, num_samples)
        t1 = time.time()
        if eperlayers is None:
            return
        print(f"{num_samples} conditional eperlayer samples generated in {t1 - t0} seconds")
        data_dict = {
            'incident_energies': conditional_energies,
            'energies_per_layer': eperlayers,
        }
        self.writer.write_hdf5('generated_energies_per_layer', data_dict)
        
        evaluate_energies_per_layer(eperlayers, conditional_energies, self.cfg, self.writer.logdir)

    def record_dict(self, tag_prefix, value_dict, step, save=False):
        for k, v in value_dict.items():
            print(f"{self.module.model_type} {k}: {v:.4f}")
            self.write_scalar(f"{tag_prefix}/{k}", v, step)

        if save:
            self.writer.write_json(
                f"{tag_prefix}_{self.module.model_type}_metrics",
                {k: v.item() for k, v in value_dict.items()}
            )

    def write_checkpoint(self, tag):
        raise NotImplementedError("Define in child classes")

    def load_checkpoint(self, tag):
        raise NotImplementedError("Define in child classes")

    def _get_checkpoint_name(self, tag):
        return f"{self.ckpt_prefix}_{self.module.model_type}_{tag}"


class SingleTrainer(BaseTrainer):
    """Class for training single module"""

    def train_single_batch(self, batch, y):
        batch = batch.to(self.module.device)
        if y is not None:
            y = y.to(self.module.device)
        loss_dict = self.module.train_batch(batch, y, max_grad_norm=self.max_grad_norm)

        if self.iteration % self._STEPS_PER_LOSS_WRITE == 0:
            for k, v in loss_dict.items():
                self.write_scalar("train/"+k, v, self.iteration+1)

        self.iteration += 1
        return loss_dict

    def update_transform_parameters(self):
        train_dset = self.train_loader.dataset.x

        self.module.data_min = train_dset.min()
        self.module.data_max = train_dset.max()
        self.module.data_shape = train_dset.shape[1:]


        if self.module.whitening_transform:
            self.module.set_whitening_params(
                torch.mean(train_dset, dim=0, keepdim=True),
                torch.std(train_dset, dim=0, keepdim=True)
            )

    def write_checkpoint(self, tag):
        if self.module.num_optimizers == 1:
            opt_state_dict = self.module.optimizer.state_dict()
            lr_state_dict = self.module.lr_scheduler.state_dict()
        else:
            opt_state_dict = [opt.state_dict() for opt in self.module.optimizer]
            lr_state_dict = [lr.state_dict() for lr in self.module.lr_scheduler]

        checkpoint = {
            "iteration": self.iteration,
            "epoch": self.epoch,

            "module_state_dict": self.module.state_dict(),
            "opt_state_dict": opt_state_dict,
            "lr_state_dict": lr_state_dict,

            "bad_valid_epochs": self.bad_valid_epochs,
            "best_valid_loss": self.best_valid_loss
        }

        self.writer.write_checkpoint(self._get_checkpoint_name(tag), checkpoint)

    def load_checkpoint(self, tag):
        checkpoint = self.writer.load_checkpoint(self._get_checkpoint_name(tag), self.module.device)

        self.iteration = checkpoint["iteration"]
        self.epoch = checkpoint["epoch"]

        self.module.load_state_dict(checkpoint["module_state_dict"])

        if self.module.num_optimizers == 1:
            self.module.optimizer.load_state_dict(checkpoint["opt_state_dict"])
            try:
                self.module.lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
            except KeyError:
                print("WARNING: Not setting lr scheduler state dict since it is not available in checkpoint.")
        else:
            for (optimizer, state_dict) in zip(self.module.optimizer, checkpoint["opt_state_dict"]):
                optimizer.load_state_dict(state_dict)
            try:
                for (lr_scheduler, state_dict) in zip(self.module.lr_scheduler, checkpoint["lr_state_dict"]):
                    lr_scheduler.load_state_dict(state_dict)
            except KeyError:
                print("WARNING: Not setting lr scheduler state dict since it is not available in checkpoint.")

        self.bad_valid_epochs = checkpoint["bad_valid_epochs"]
        self.best_valid_loss = checkpoint["best_valid_loss"]

        print(f"Loaded {self.module.model_type} checkpoint `{tag}' after epoch {self.epoch}")

    def get_all_loaders(self):
        return self.train_loader, self.valid_loader, self.test_loader

    def update_all_loaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.evaluator.valid_loader = valid_loader
        self.evaluator.test_loader = test_loader
