def get_base_config(dataset, standalone, epl=False):
    if standalone:
        standalone_info = {
            "train_batch_size": 128,
            "valid_batch_size": 128,
            "test_batch_size": 128,

            "make_valid_loader": True,

            "data_root": "data/",
            "logdir_root": "runs/"
        }
        scale_data = False
        whitening_transform = False
    else:
        standalone_info = {}
        scale_data = True
        whitening_transform = False

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
        physics_standalone = {}
        if standalone:
            physics_standalone = {
                "valid_metrics": ["ave_histogram_difference"],
                "test_metrics": ["ave_histogram_difference", "sample_timing"],
                "metric_kwargs": {
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

                    "conditional_on_epl": False,
                    "data_size": 100000,
                    "batch_size": 1000,
                    "hidden_dims": [512, 512, 512],
                    "lr": 5e-4,
                    "max_epochs": 300,
                    "max_bad_valid_epochs": 20,
                },
                "early_stopping_metric": "ave_histogram_difference",
                "max_bad_valid_epochs": 20,
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

            **physics_standalone,
        }

    if epl:
        caloflow_epl = True,
        epl_info = {
            "sample_showers": False,
            "sample_eperlayer": True,
            "max_epochs": 300,
            "hidden_units": 8,
            "caloflow_epl": caloflow_epl,
            "valid_metrics": ["loss"],
            "test_metrics": ["log_likelihood", "ave_epl_histogram_difference"],
            "metric_kwargs": {
                    "dataset_name": dataset,
                    "max_deposited_ratio": max_deposited_ratio[dataset],
                    "raw_shape": raw_shapes[dataset],
                    "conditional_on_epl": True,
                    "caloflow_epl": caloflow_epl,
                },
            "early_stopping_metric": "loss",
            "max_bad_valid_epochs": 20,
            }
    else:
        caloflow_epl = False,
        epl_info = {}

    return {
        "flatten": True,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": scale_data,
        "whitening_transform": whitening_transform,
        "use_labels": True,
        "sample_showers": True,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": False,
        "max_epochs": 200,
        "max_grad_norm": 10,

        "early_stopping_metric": "loss", #"ave_histogram_difference"
        "max_bad_valid_epochs": 20,

        # NOTE: A validation metric should indicate better performance as it decreases.
        #       Thus, log_likelihood is not an appropriate validation metric.
        "valid_metrics": ["loss"], #,"ave_histogram_difference"
        "test_metrics": ["log_likelihood"],

        **standalone_info,
        **physics_dataset_info,
        **epl_info,
    }


def get_arm_config(dataset, standalone, **kwargs):
    arm_base = {
        "k_mixture": 10,

        "flatten": False,
        "early_stopping_metric": "loss",
        "max_bad_valid_epochs": 10,
        "use_lr_scheduler": True
    }

    if standalone:
        hidden_size = 256
        num_layers = 2
    else:
        hidden_size = 128
        num_layers = 1

    net_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    return{
        **arm_base,
        **net_config
    }


def get_avb_config(dataset, standalone, **kwargs):
    return {
        "max_epochs": 100,

        "noise_dim": 128,
        "latent_dim": 20,
        "encoder_net": "mlp",
        "decoder_net": "mlp",
        "encoder_hidden_dims": [256],
        "decoder_hidden_dims": [256],
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


def get_ebm_config(dataset, standalone, **kwargs):
    if standalone:
        net = "mlp" if dataset in ["mnist", "fashion-mnist", "photons1", "pions1"] else "cnn"
        lr = 0.0003
        x_lims = (0, 1)
        loss_alpha = 1.0
        spectral_norm = True
        if net == "mlp":
            energy_func_hidden_dims = [256, 128]
    else:
        net = "mlp"
        x_lims = (-1, 1)
        energy_func_hidden_dims = [64, 32]
        lr = 0.001
        loss_alpha = 0.1
        spectral_norm = False

    ebm_base = {
        "max_length_buffer": 8192,
        "x_lims": x_lims,
        "ld_steps": 60,
        "ld_step_size": 10,
        "ld_eps_new": 0.05,
        "ld_sigma": 0.005,
        "ld_grad_clamp": 0.03,
        "loss_alpha": loss_alpha,

        "scale_data": True,
        "whitening_transform": False,
        "spectral_norm": spectral_norm,

        "lr": lr,
        "max_grad_norm": 1.0,
    }

    if net == "mlp":
        net_config = {
            "net": "mlp",
            "energy_func_hidden_dims": energy_func_hidden_dims
        }

    elif net == "cnn":
        net_config = {
            "net": "cnn",
            "energy_func_hidden_channels": [64, 64, 32, 32],
            "energy_func_kernel_size": [3, 3, 3, 3],
            "energy_func_stride": [1, 1, 1, 1],

            "flatten": False
        }

    return {
        **ebm_base,
        **net_config
    }


def get_flow_config(dataset, standalone, **kwargs):
    if standalone:
        hidden_units = 128
        lr = 0.0005
        standalone_info = {
            "early_stopping_metric": "loss",
            "max_bad_valid_epochs": 30,
            "valid_metrics": ["loss"],
            "test_metrics": ["loss"],
        }
    else:
        hidden_units = 64
        lr = 0.001
        standalone_info = {}
    flow_config = {
        "scale_data": True,
        "whitening_transform": False,

        "transform": "simple_nsf",
        "hidden_units": hidden_units,
        "num_layers": 4,
        "num_blocks_per_layer": 3,
        "num_bins":8,
        "tail_bound":1,
        "lr": lr,
        "random_mask":False,
        "use_residual_blocks":True,
        "permutation":"reverse",
        "dropout_probability":0.0,
    }
    return {
        **flow_config,
        **standalone_info,
    }


def get_vae_config(dataset, standalone, **kwargs):
    vae_base = {
        "latent_dim": 20,
        "k": 1,
        "beta": 1,
        "use_lr_scheduler": False,

        "single_sigma": True,
    }

    if standalone and dataset in ["electrons2", "electrons3"]:
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

    else:
        net_config = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [512, 512, 512],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [512, 512, 512],

            "flatten": True
        }

    return {
        **vae_base,
        **net_config,
    }


DE_CFG_MAP = {
    "base": get_base_config,
    "arm": get_arm_config,
    "avb": get_avb_config,
    "ebm": get_ebm_config,
    "flow": get_flow_config,
    "vae": get_vae_config
}
