def get_dim_config(dataset, estimator):
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
            
            "preprocess_physics_data": False,
            "caloflow_epl": False,
        }
    generated_dataset_info = {}
    if "random" in dataset.lower():
        generated_dataset_info = {
            "shape": [50000, 368],
        }
    estimator_info = {}
    if estimator == 'ess':
        estimator_info['ver'] = 'a'
        estimator_info['d'] = 1


    return {
        "train_batch_size": 1024,
        "valid_batch_size": 1024,
        "test_batch_size": 1024,
        "make_valid_loader": False,
        "data_root": "data/",
        "logdir_root": "runs/",

        # Dimension estimator configuration
        "seed": 1234,
        "max_num_samples": -1,
        'n_jobs': 1,

        **physics_dataset_info,
        **generated_dataset_info,
        **estimator_info,
    }
