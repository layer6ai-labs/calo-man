import os
from distutils.dir_util import copy_tree
import torch
import math
from tqdm import tqdm

from ..datasets import get_embedding_loader, remove_drop_last
from .single_trainer import BaseTrainer


class BaseTwoStepTrainer:
    """
    Base class for training a two-step module

    NOTE: The de_trainer sent in here will be initialized with dummy dataloaders
    """
    def __init__(
            self,

            gae_trainer,
            de_trainer,

            writer,

            evaluator,

            checkpoint_load_list,
            
            cfg,

            pretrained_gae_path="",
            freeze_pretrained_gae=True,

            only_test=False,
            sample_showers=True,
    ):
        self.gae_trainer = gae_trainer
        self.de_trainer = de_trainer
        self.writer = writer
        self.evaluator = evaluator
        self.only_test = only_test
        self.cfg = cfg
        self.sample_showers = sample_showers

        if pretrained_gae_path:
            self._load_pretrained_gae(
                path=pretrained_gae_path,
                checkpoint_load_list=checkpoint_load_list,
                freeze_params=freeze_pretrained_gae
            )
        else:
            self.load_checkpoint(checkpoint_load_list)

    @property
    def gae(self):
        return self.gae_trainer.module

    @property
    def de(self):
        return self.de_trainer.module

    @property
    def module(self):
        return self.evaluator.module

    @property
    def epoch(self):
        if hasattr(self, "_epoch"):
            return self._epoch
        else:
            return self.gae_trainer.epoch + self.de_trainer.epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    def train(self):
        raise NotImplementedError("Implement train function in child classes")

    def _set_de_loaders(self):
        gae_loaders = self.gae_trainer.get_all_loaders()
        de_loaders = []

        for loader in gae_loaders:
            if loader == None:
                de_loaders.append(None)
                continue

            # NOTE: If we drop last in this step, we will lose training data in embeddings.
            #       However, we cannot simply change loader.drop_last to False after the loader
            #       is initialized. Thus, we create a new dataloader in remove_drop_last.
            loader_drop_last = loader.drop_last
            if loader_drop_last:
                loader = remove_drop_last(loader)

            with torch.no_grad():
                encoded_data = self.gae.encode(loader, None)  # second input is dummy thanks to @batch_or_dataloader()

            encoded_dataloader = get_embedding_loader(
                embeddings=encoded_data,
                batch_size=loader.batch_size,
                drop_last=loader_drop_last,
                role=loader.dataset.role,
                y=loader.dataset.y,
            )
            de_loaders.append(encoded_dataloader)

        self.de_trainer.update_all_loaders(*de_loaders)
        
    def _tqdm_progress_bar(self, iterable, desc, length, leave):
        return tqdm(
            iterable,
            desc=desc,
            total=length,
            bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
            leave=leave
        )

    def _load_pretrained_gae(self, path, checkpoint_load_list, freeze_params):
        module = self.gae_trainer.module
        model_type = module.model_type

        checkpoint_found = False
        old_checkpoint_dir = os.path.join(path, "checkpoints")
        for checkpoint in checkpoint_load_list:
            checkpoint_path = os.path.join(old_checkpoint_dir, f"gae_{model_type}_{checkpoint}.pt")
            if os.path.exists(checkpoint_path):
                checkpoint_found = True
                checkpoint_to_load = checkpoint
                break

        if not checkpoint_found:
            raise RuntimeError(f"Valid {model_type} checkpoint not found in {path}/checkpoints/")

        # Move the GAE checkpoints over so that they can be loaded on tests with the two-step module
        copy_tree(old_checkpoint_dir, self.writer._checkpoints_dir)
        self.gae_trainer.load_checkpoint(checkpoint_to_load)

        if freeze_params:
            for p in module.parameters():
                p.requires_grad = False

            self.gae_trainer.epoch = self.gae_trainer.max_epochs

            # We do not want to allow the GAE trainer to train if the parameters are frozen
            self.gae_trainer.only_test = True

        print(f"Loaded pretrained gae from {checkpoint_path}")

    def write_checkpoint(self, tag):
        self.gae_trainer.write_checkpoint(tag)
        self.de_trainer.write_checkpoint(tag)

    def load_checkpoint(self, checkpoint_load_list):
        for trainer, name in zip((self.gae_trainer, self.de_trainer), ("gae", "de")):
            for ckpt in checkpoint_load_list:
                try:
                    trainer.load_checkpoint(ckpt)
                    break
                except FileNotFoundError:
                    print(f"Did not find {ckpt} {name} checkpoint")


class SequentialTrainer(BaseTwoStepTrainer):
    """Class for fully training a GAE model and then a DE model"""
    def train(self):
        epl_module = self.evaluator.metric_kwargs.get("epl_module", None)
        if not self.only_test:
            # Simply call SingleTrainer train method
            self.gae_trainer.train()

            self._set_de_loaders()
            # Move SingleTrainer code here so that we can sample per epoch using the full model
            self.de_trainer.update_transform_parameters()
            while self.de_trainer.epoch < self.de_trainer.max_epochs and self.de_trainer.bad_valid_epochs < self.de_trainer.max_bad_valid_epochs:
                self.de_trainer.module.train()

                self.de_trainer.train_for_epoch()
                # Validate
                with torch.no_grad():
                    valid_results = self.de_trainer.evaluator.validate()
                    # Use two-step evaluator for ave_histogram_difference metric
                    two_step_valid_results = self.evaluator.validate()
                    joint_dict = {**valid_results, **two_step_valid_results}
                    self.record_dict("validate", joint_dict, self.de_trainer.epoch)
                    valid_loss = joint_dict.get(self.de_trainer.early_stopping_metric)
                    
                    print("Sampling showers to tensorboard for DE validation")
                    self._sample_showers_to_tensorboard(self.evaluator.valid_loader, epl_module=None)

                if self.de_trainer.early_stopping_metric:
                    if valid_loss < self.de_trainer.best_valid_loss:
                        self.de_trainer.bad_valid_epochs = 0
                        self.de_trainer.best_valid_loss = valid_loss
                        self.de_trainer.write_checkpoint("best_valid")

                        print(f"Best validation loss of {valid_loss} achieved on epoch {self.epoch}")

                    else:
                        self.de_trainer.bad_valid_epochs += 1

                        if self.de_trainer.bad_valid_epochs == self.de_trainer.max_bad_valid_epochs:
                            print(f"No validation improvement for {self.de_trainer.max_bad_valid_epochs}"
                                    + " epochs. Training halted.")
                            self.de_trainer.write_checkpoint("latest")

                            self.de_trainer.load_checkpoint("best_valid")
                            with torch.no_grad():
                                self.de_trainer._test()
                                test_results = self.evaluator.test()
                                self.record_dict("test", test_results, save=True)
                                print("Sampling showers to tensorboard for DE testing")
                                self._sample_showers_to_tensorboard(self.evaluator.test_loader, epl_module=epl_module)
                                if self.sample_showers:
                                    print("Sampling showers to hdf5 for DE testing")
                                    self._sample_to_hdf5(epl_module=epl_module)

                            return

            self.de_trainer.write_checkpoint("latest")
        else:
            self._set_de_loaders()

        if self.de_trainer.bad_valid_epochs < self.de_trainer.max_bad_valid_epochs:
            with torch.no_grad():
                self.de_trainer._test()
            print(f"Maximum epochs reached. Training of {self.de_trainer.module.model_type} complete.")

        with torch.no_grad():
            test_results = self.evaluator.test()
            self.record_dict("test", test_results, save=True)
            print("Sampling showers to tensorboard for two_step testing")
            self._sample_showers_to_tensorboard(self.de_trainer.test_loader, epl_module=epl_module)
            if self.sample_showers:
                print("Sampling showers to hdf5 for two_step testing")
                self._sample_to_hdf5(epl_module=epl_module)

    def sample_and_record(self):
        return BaseTrainer.sample_and_record(self)
    
    def _sample_showers_to_tensorboard(self, loader, num_samples=5000, epl_module=None):
        return BaseTrainer._sample_showers_to_tensorboard(self, loader, num_samples=num_samples, epl_module=epl_module)
    
    def _sample_to_hdf5(self, eval=True, epl_module=None, epl_cfg=None):
        return BaseTrainer._sample_showers_to_hdf5(self, eval=eval, epl_module=epl_module, epl_cfg=epl_cfg)

    def record_dict(self, tag_prefix, value_dict, save=False):
        return BaseTrainer.record_dict(self, tag_prefix, value_dict, step=self.epoch, save=save)

    def write_scalar(self, *args, **kwargs):
        return BaseTrainer.write_scalar(self, *args, **kwargs)

    # NOTE: Here, almost all of the training functionality is handled by the underlying
    #       <x>_trainer instances. Saving and loading of models, however, is managed by the parent class.


class BaseAlternatingTrainer(BaseTrainer, BaseTwoStepTrainer):
    """
    Class for alternating between training a GAE model and a DE model every epoch
    """
    def __init__(
            self,

            two_step_module,

            gae_trainer,
            de_trainer,

            train_loader,
            valid_loader,
            test_loader,

            writer,

            max_epochs,

            early_stopping_metric,
            max_bad_valid_epochs,
            max_grad_norm,

            evaluator,

            checkpoint_load_list,
            
            cfg,

            only_test=False,
            sample_showers=False,

            pretrained_gae_path=None,
            freeze_pretrained_gae=True,
    ):
        assert two_step_module is evaluator.module, "Evaluator must point to same module"
        self.ckpt_prefix = "two_step"

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

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

        self.only_test = only_test
        self.sample_showers = sample_showers

        BaseTwoStepTrainer.__init__(
            self=self,
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            writer=writer,
            evaluator=evaluator,
            checkpoint_load_list=checkpoint_load_list,
            cfg=cfg,
            pretrained_gae_path=pretrained_gae_path,
            freeze_pretrained_gae=freeze_pretrained_gae,
            sample_showers=self.sample_showers,
        )

    def train(self):
        BaseTrainer.train(self)

    def write_checkpoint(self, tag):
        BaseTwoStepTrainer.write_checkpoint(self, tag)

        checkpoint = {
            "epoch": self.epoch,
            "iteration": self.iteration,

            "bad_valid_epochs": self.bad_valid_epochs,
            "best_valid_loss": self.best_valid_loss
        }

        self.writer.write_checkpoint(self._get_checkpoint_name(tag), checkpoint)

    def load_checkpoint(self, checkpoint_load_list):
        BaseTwoStepTrainer.load_checkpoint(self, checkpoint_load_list)

        checkpoint_found = False
        for ckpt in checkpoint_load_list:
            try:
                checkpoint = self.writer.load_checkpoint(self._get_checkpoint_name(ckpt), self.module.device)
                tag = ckpt
                checkpoint_found = True
                break
            except FileNotFoundError:
                print(f"Did not find two step checkpoint `{ckpt}'")

        if not checkpoint_found: return

        self.epoch = checkpoint["epoch"]
        self.iteration = checkpoint["iteration"]

        self.bad_valid_epochs = checkpoint["bad_valid_epochs"]
        self.best_valid_loss = checkpoint["best_valid_loss"]

        print(f"Loaded two step checkpoint `{tag}' after epoch {self.epoch}")


class AlternatingEpochTrainer(BaseAlternatingTrainer):
    """
    Class for alternating between training a GAE model and a DE model every epoch
    """
    def train_for_epoch(self):
        self.gae_trainer.train_for_epoch()

        self._set_de_loaders()
        self.de_trainer.update_transform_parameters()

        self.de_trainer.train_for_epoch()

        self.iteration = self.de_trainer.iteration
        self.epoch += 1

    def update_transform_parameters(self):
        # This is invoked at the start of the epoch, so wait until
        # de training starts above to invoke self.de_trainer.update_transform_parameters()
        self.gae_trainer.update_transform_parameters()


class AlternatingIterationTrainer(BaseAlternatingTrainer):
    """
    Class for alternating between training a GAE model and a DE model every iteration
    """
    def train_single_batch(self, batch, y):
        batch = batch.to(self.gae.device)
        y_gae = y
        if y is not None:
            y_gae = y.to(self.gae.device)
        gae_loss_dict = self.gae_trainer.train_single_batch(batch, y_gae)

        encoded_batch = self.module.generalized_autoencoder.encode(batch).detach()
        encoded_batch = encoded_batch.to(self.de.device)
        y_de = y
        if y is not None:
            y_de = y.to(self.de.device)

        de_loss_dict = self.de_trainer.train_single_batch(encoded_batch, y_de)

        if self.iteration % self._STEPS_PER_LOSS_WRITE == 0:
            for k, v in gae_loss_dict.items():
                self.gae_trainer.write_scalar("train/"+k, v, self.iteration+1)
            for k, v in de_loss_dict.items():
                self.de_trainer.write_scalar("train/"+k, v, self.iteration+1)

        self.iteration += 1

        loss_dict = {}
        for k, v in gae_loss_dict.items():
            loss_dict[f"gae_{k}"] = v
        for k, v in de_loss_dict.items():
            loss_dict[f"de_{k}"] = v

        return loss_dict

    def update_transform_parameters(self):
        self.gae_trainer.update_transform_parameters()
        self._set_de_loaders()
        self.de_trainer.update_transform_parameters()
