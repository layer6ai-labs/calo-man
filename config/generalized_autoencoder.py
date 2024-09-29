def get_base_config(dataset, standalone, **kwargs):
    if standalone:
        standalone_info = {
            "train_batch_size": 512,
            "valid_batch_size": 512,
            "test_batch_size": 512,

            "make_valid_loader": True,

            "data_root": "data/",
            "logdir_root": "runs/"
        }
    else:
        standalone_info = {}

    physics_dataset_info = {}
    if dataset in ["photons1", "pions1", "electrons2", "electrons3"]:
        max_deposited_ratio = {
            "photons1": 3.1,
            "pions1": 7.0,
            "electrons2": 2.0,
            "electrons3": 2.0,
            }
        raw_shapes = {
            "photons1": [-1, 368],
            "pions1": [-1, 533],
            "electrons2": [-1, 45, 16, 9],
            "electrons3": [-1, 45, 50, 18],
        }
        energy_min_max = {
            "photons1": [0.256, 4194.304],
            "pions1": [0.256, 4194.304],
            "electrons2": [1.0, 1000.0],
            "electrons3": [1.0, 1000.0],
        }
        normalize_logspace = {
            "photons1": [False, False, False, False, False],
            "pions1": [False, False, False, False, False],
            "electrons2": [False, False, False, False, True],
            "electrons3": [False, False, False, False, True],
        }
        physics_dataset_info = {
            "dataset_name": dataset,
            "max_deposited_ratio": max_deposited_ratio[dataset],
            "raw_shape": raw_shapes[dataset],
            "energy_min": energy_min_max[dataset][0],
            "energy_max": energy_min_max[dataset][1],

            "normalize_to_epl": normalize_logspace[dataset][0],
            "normalized_deposited_energy": normalize_logspace[dataset][1],
            "deposited_energy_per_layer": normalize_logspace[dataset][2],
            "logspace_incident_energies": normalize_logspace[dataset][3],
            "logitspace_voxels": normalize_logspace[dataset][4],

            "preprocess_physics_data": True,
            "caloflow_epl": False,
        }


    return {
        "flatten": False,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": False,
        "whitening_transform": False,
        "use_labels": False,
        "sample_showers": True,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": True,
        "max_epochs": 300,
        "max_grad_norm": 10,

        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 20,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],

        **standalone_info,
        **physics_dataset_info,
    }



def get_ae_config(dataset, standalone, **kwargs):
    if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"]:
        net = "mlp"
    elif dataset in ["svhn", "cifar10"]:
        net = "cnn"
    else:
        net = "residual"

    ae_base = {
        "latent_dim": 35,
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False
        }

    elif net == "residual":
        net_config = {
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False
        }

    return {
        **ae_base,
        **net_config,
    }


def get_avb_config(dataset, standalone, **kwargs):
    if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"]:
        net = "mlp"
    elif dataset in ["svhn", "cifar10"]:
        net = "cnn"
    else:
        net = "residual"

    avb_base = {
        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 10,

        "latent_dim": 35,

        "discriminator_hidden_dims": [256, 256],

        "single_sigma": True,

        "input_sigma": 3.,
        "prior_sigma": 1.,

        "lr": None,
        "disc_lr": 0.001,
        "nll_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_nll_lr_scheduler": True,
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 50,
            "noise_dim": 256,
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 100,
            "noise_dim": 256,
        }

    elif net == "residual":
        net_config = {
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False,
            "max_epochs": 100,
            "noise_dim": 256,
        }

    return {
        **avb_base,
        **net_config,
    }


def get_bigan_config(dataset, standalone, **kwargs):
    if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"]:
        net = "mlp"
    elif dataset in ["svhn", "cifar10"]:
        net = "cnn"
    else:
        net = "residual"

    bigan_base = {
        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 50,

        "latent_dim": 35,

        "discriminator_hidden_dims": [256, 256],
        "num_discriminator_steps": 2,
        "wasserstein": True,
        "clamp": 0.01,
        "gradient_penalty": True,
        "lambda": 10.0,
        "recon_weight": 1.0,

        "optimizer": 'adam',
        "lr": None,
        "disc_lr": 0.0001,
        "ge_lr": 0.0001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_ge_lr_scheduler": True,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 200,
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 200,
        }

    elif net == "residual":
        net_config = {
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False,
            "max_epochs": 100,
        }


    return {
        **bigan_base,
        **net_config,
    }


def get_vae_config(dataset, standalone, **kwargs):
    if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"]:
        net = "mlp"
    elif dataset in ["svhn", "cifar10"]:
        net = "cnn"
    else:
        net = "residual"

    vae_base = {
        "latent_dim": 35,
        "k": 1,
        "beta": 1,
        "single_sigma": True,
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [512, 512, 512],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [512, 512, 512],

            "flatten": True
        }

    elif net == "cnn":
        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False
        }

    elif net == "residual":
        net_config = {
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False,
            "max_epochs": 100,
        }

    return {
        **vae_base,
        **net_config,
    }


def get_wae_config(dataset, standalone, **kwargs):
    if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"]:
        net = "mlp"
    elif dataset in ["svhn", "cifar10"]:
        net = "cnn"
    else:
        net = "residual"

    wae_base = {
        "latent_dim": 35,

        "max_epochs": 300,

        "discriminator_hidden_dims": [256, 256],

        "_lambda": 10.,
        "sigma": 1.,

        "lr": None,
        "disc_lr": 0.0005,
        "rec_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": False,
        "use_rec_lr_scheduler": False,

        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 30,

        "valid_metrics": ["l2_reconstruction_error"],
    }

    if net == "mlp":
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True
        }

    elif net == "cnn":
        enc_hidden_channels = [64, 64, 32, 32]
        enc_kernel = [3, 3, 3, 3]
        enc_stride = [1, 1, 1, 1]

        net_config = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": enc_hidden_channels,
            "encoder_kernel_size": enc_kernel,
            "encoder_stride": enc_stride,

            "decoder_net": "cnn",
            "decoder_hidden_channels": enc_hidden_channels[::-1],
            "decoder_kernel_size": enc_kernel[::-1],
            "decoder_stride": enc_stride[::-1],

            "flatten": False
        }

    elif net == "residual":
        net_config = {
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False,
            "max_epochs": 100,
        }

    return {
        **wae_base,
        **net_config,
    }


GAE_CFG_MAP = {
    "base": get_base_config,
    "ae": get_ae_config,
    "avb": get_avb_config,
    "bigan": get_bigan_config,
    "vae": get_vae_config,
    "wae": get_wae_config
}
