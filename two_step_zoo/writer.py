# NOTE: The below file is modified from commit `aeaf5fd` of
#       https://github.com/jrmcornish/cif/blob/master/cif/writer.py
import os
import datetime
import json
import sys

import numpy as np
import torch
import h5py
from tensorboardX import SummaryWriter


class Tee:
    """This class allows for redirecting of stdout and stderr"""
    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


class Writer:
    _STDOUT = sys.stdout
    _STDERR = sys.stderr

    def __init__(self, logdir, make_subdir, tag_group):
        if make_subdir:
            os.makedirs(logdir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(logdir, timestamp)

        self._writer = SummaryWriter(logdir=logdir)

        assert logdir == self._writer.logdir
        self.logdir = logdir

        self._tag_group = tag_group

        sys.stdout = Tee(
            primary_file=self._STDOUT,
            secondary_file=open(os.path.join(logdir, "stdout"), "a")
        )

        sys.stderr = Tee(
            primary_file=self._STDERR,
            secondary_file=open(os.path.join(logdir, "stderr"), "a")
        )

    def write_scalar(self, tag, scalar_value, global_step=None):
        self._writer.add_scalar(self._tag(tag), scalar_value, global_step=global_step)

    def write_image(self, tag, img_tensor, global_step=None):
        self._writer.add_image(self._tag(tag), img_tensor, global_step=global_step)

    def write_figure(self, tag, figure, global_step=None):
        self._writer.add_figure(self._tag(tag), figure, global_step=global_step)

    def write_hparams(self, hparam_dict=None, metric_dict=None):
        self._writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    def write_json(self, tag, data):
        text = json.dumps(data, indent=4)

        self._writer.add_text(
            self._tag(tag),
            4*" " + text.replace("\n", "\n" + 4*" ") # Indent by 4 to ensure codeblock formatting
        )

        json_path = os.path.join(self.logdir, f"{tag}.json")

        with open(json_path, "w") as f:
            f.write(text)

    def write_textfile(self, tag, text):
        path = os.path.join(self.logdir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

    def write_numpy(self, tag, arr):
        path = os.path.join(self.logdir, f"{tag}.npy")
        np.save(path, arr)
        print(f"Saved array to {path}")
        
    def write_csv(self, tag, arr, header=""):
        path = os.path.join(self.logdir, f"{tag}.csv")
        np.savetxt(path, arr, delimiter=",", header=header)
        print(f"Saved array to {path}")

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = self._checkpoint_path(tag)

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # replace is atomic, so we guarantee our checkpoints are always good
        os.replace(tmp_checkpoint_path, checkpoint_path)

    def write_hdf5(self, tag, data_dict):
        path = os.path.join(self.logdir, f"{tag}.hdf5")
        file = h5py.File(path, 'w')
        for key, val in data_dict.items():
            file.create_dataset(key,
			    data=val,
			    compression='gzip')
        file.close()
        print(f"Saved datasets to {path}")

    def load_checkpoint(self, tag, device):
        return torch.load(self._checkpoint_path(tag), map_location=device)

    def _checkpoint_path(self, tag):
        return os.path.join(self._checkpoints_dir, f"{tag}.pt")

    @property
    def _checkpoints_dir(self):
        return os.path.join(self.logdir, "checkpoints")

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"


def get_writer(cmd_line_args, write_cfg=True, **kwargs):
    three_step = ("epl_cfg" in kwargs)
    two_step = ("shared_cfg" in kwargs)

    if cmd_line_args.load_dir and not (two_step and cmd_line_args.load_pretrained_gae):
        # NOTE: In this case, operate in the existing directory
        writer = Writer(
            logdir=cmd_line_args.load_dir,
            make_subdir=False,
            tag_group=cmd_line_args.dataset
        )
    else:
        cfg = kwargs["shared_cfg"] if two_step else kwargs["cfg"]
        writer = Writer(
            logdir=cfg["logdir_root"],
            make_subdir=True,
            tag_group=cmd_line_args.dataset,
        )

    if write_cfg:
        if three_step:
            writer.write_json(tag="epl_config", data=kwargs["epl_cfg"])
        if two_step:
            writer.write_json(tag="gae_config", data=kwargs["gae_cfg"])
            writer.write_json(tag="de_config", data=kwargs["de_cfg"])
            writer.write_json(tag="shared_config", data=kwargs["shared_cfg"])
        else:
            writer.write_json(tag="config", data=kwargs["cfg"])

    return writer
