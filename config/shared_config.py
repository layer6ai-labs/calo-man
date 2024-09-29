def get_shared_config(dataset):
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
            "photons1": [True, False, False, False, False],
            "pions1": [True, False, False, False, False],
            "electrons2": [True, False, False, False, True],
            "electrons3": [True, False, False, False, True],
        }
        test_set_size = {
            "photons1": 121000,
            "pions1": 120230,
            "electrons2": 512,
            "electrons3": 512,
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
                "max_epochs": 60,
                "max_bad_valid_epochs": 10,
                "caloflow_epl": False,
            },
        }

    return {
        "dataset": dataset,

        "sequential_training": True,
        "alternate_by_epoch": False,
        "sample_showers": True,

        "max_epochs": 100,
        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,
        "max_grad_norm": None,

        "make_valid_loader": True,

        "data_root": "data/",
        "logdir_root": "runs/",

        "train_batch_size": 512,
        "valid_batch_size": 512,
        "test_batch_size": test_set_size[dataset],

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],

        **physics_dataset_info,
    }
