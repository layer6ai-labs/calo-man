<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# CaloMan - Density estimation on learned manifolds for calorimeter simulation

This is the codebase accompanying the work ["CaloMan: Fast generation of calorimeter showers with density estimation on learned manifolds"](https://arxiv.org/abs/2211.15380), which was submitted to the ["Fast Calorimeter Simulation Challenge 2022"](https://calochallenge.github.io/homepage/). It builds on a fork of [the Two-Step Zoo codebase](https://github.com/layer6ai-labs/two_step_zoo) from the paper ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172).
Here we discuss how to run the experiments in the paper and give a general overview of the codebase.

Before running on calorimeter data you must download the datasets from the links provided at [calochallenge.github.io/homepage](https://calochallenge.github.io/homepage/), and place them in the `data` folder.

## Python Environment Setup

The main prerequisite is to set up the python environment.
The command

    conda env create -f environment.yml

will create a `conda` environment called `caloman`.
Launch this environment using the command

    conda activate caloman

before running any experiments.

The file `env-lock.yml` contains strict versions of each of the packages.

## Datasets and Dimension Estimation - `dimension-main.py`

This codebase is designed for experimentation on the datasets from the ["Fast Calorimeter Simulation Challenge 2022"](https://calochallenge.github.io/homepage/), which contain simulated calorimeter showers produced by incident photons, pions, and electrons.

We aim to model the distributions of these datasets taking into account their low-dimensional manifold structure. Our modelling techniques take the dimension of the manifold as input, but this is unknown for the challenge's datasets. Hence, we first estimate the intrinsic dimension using statistical estimators.

The dimension estimation module is based on ["The Intrinsic Dimension of Images and Its Impact on Learning"](https://openreview.net/forum?id=XJk19XzGq2J) by Pope et al. and builds on their [code](https://github.com/ppope/dimensions).

Dimension estimation experiments are run from the script 'dimension_main.py' via a call like

    python dimension_main.py --dataset <dataset> --config <optional-argument>=<argument-value>

where:

- `<dataset>` is the dataset
  - For the Fast Calorimeter Simulation Challenge 2022 the available datasets are `photons1`, `pion1`, `electrons2`, and `electrons3`.
- '<optional-argument>' overrides the default configuration in 'config/intrinsic_dimension.py'
  - Useful arguments for modifying the datasets are the booleans `preprocess_physics_data`, `normalized_deposited_energy`, `logspace_voxels`, `logspace_incident_energies`,
  - `k` is a hyperparameter of the dimension estimator,
  - `max_number_samples` specifies how many datapoints to use in the estimation. The argument `-1` uses all the data,
  - `class_ind` specifies a particular class label to estimate the dimension of. In this context the class labels are the incident energies. The argument `-1` uses all the classes,
  - `asymptotically_unbiased` is a boolean that specifies whether to correct the maximum likelihood estimate to make it asymptotically unbiased.

An example experiment is:

    python dimension_main.py --dataset photons1 --config max_number_samples=-1 --config logspace_incident_energies=True

## Usage - `main.py`

The main script for running experiments is unsurprisingly `main.py`.
This script runs two-step experiments, wherein a **Generalized Autoencoder (GAE)** is trained to embed the data in some lower-dimensional space, and a **Density Estimator (DE)** is trained to estimate the density of these embeddings.
The basic usage is as follows:

    python main.py --dataset <dataset> --gae-model <gae-model> --de-model <de-model>

where:

- `<dataset>` is the dataset
  - Our work is based on the datasets  `photons1`, `pion1`, `electrons2`, and `electrons3`. Machine learning benchmark datasets are also available with `mnist`, `fashion-mnist`, `svhn`, `cifar10`.
- `<gae-model>` is the generalized autoencoder
  - Currently, we support any of the following: `ae`, `avb`, `bigan`, `vae`, and `wae`
- `<de-model>` is the density estimator
  - Currently, we support any of the following: `arm`, `avb`, `ebm`, `flow`, and `vae`

### Dynamic Updating of Config Values

Model and training hyperparameters are loaded from the config files in the directory `config` at runtime.
However, it is also possible to update the hyperparameters on the command line using any of the flags `--shared-config` (for shared hyperparameters), `--gae-config` (for GAE hyperparameters), and/or `--de-config` (for DE hyperparameters).
For each hyperparameter `<key>` that one wants to set to a new value `<value>`, given one of the config flags above as `--<flag>`, add the following to the command line:

    --<flag> <key>=<value>

We can do this multiple times for multiple configs. In particular, say we want to change the run directory to `new_runs`, we want to change the DE optimizer to SGD with learning rate `0.01`, and we want to change the GAE latent dimension to `5`.
The options to do this would appear as follows:

    --shared-config logdir_root=new_runs --de-config optimizer=sgd --de-config lr=0.01 --gae-config latent_dim=5

A full list of config values is visible in the respective files in the `config` directory.

### Run Directories

By default, the `main` command above will create a directory of the form `runs/<date>_<hh>-<mm>-<ss>`, e.g. `Apr26_09-38-37`, to store information about the run, including:

- `torch` model checkpoints
- Experiment metrics / results as `json`
- `tensorboard` files
- Config files as `json`
- `stderr` / `stdout` logs

If the config option `sample_showers` is set to `True`, then calorimeter showers will be generated and saved to this directory in the format of the Challenge submission.

We provide the ability to reload a saved run with run directory `<dir>` via the command:

    ./main.py --load-dir <dir>

which will restart the training of the model (if not completed) and then perform testing.

Adding the flag `--load-best-valid-first` attempts to load `best_valid` checkpoints saved by early stopping before loading the `latest` checkpoints (loading `latest` is default behaviour).

Furthermore, adding `--max-epochs-loaded` (resp. `--max-epochs-loaded-gae`, `--max-epochs-loaded-de`) to the command with some integer argument changes the maximum number of epochs of shared (resp. GAE, DE) training; this may be useful if a loaded model has already hit the originally-specified maximum number of epochs but further training is desired.

The flag `--only-test` is also available here and only performs testing on the loaded model, no training. This is useful for post hoc testing of metrics not included on the original run.

### Inspecting Results with Tensorboard

Assuming runs are being stored in the default directory, some training curves and additional model samples can be viewed via `tensorboard` using the command

    tensorboard --logdir runs

Tensorboard also shows the config used to produce each run.

### Loading Pretrained GAEs

It is possible to load pretrained GAE models (trained using `single_main.py`; details below) and then train a density estimator on the embeddings from this pretrained GAE, with the option to further train the weights of the GAE.
Assuming the pretrained GAE has run directory `runs/<gae-dir>`, this can be accomplished via:

    ./main.py --load-dir runs/<gae-dir> --de-model <de> --load-pretrained-gae

where `<de>` is any of the DE models listed above.

**IMPORTANT NOTE:** The pretrained GAE *must* have been trained with the `--is-gae` flag if initially run by `single_main.py` so that the `main.py` command in this section properly picks up the config `gae_config.json`.

We also often use the following tags described below:
- `--load-best-valid-first`: Attempt to load the `best_valid` checkpoint from early stopping before the `latest` checkpoint.
- `--freeze-pretrained-gae`: Freeze the weights of the pretrained GAE, i.e. *do not* train the pretrained GAE further.

### Metrics

In this section we give a brief overview of the available metrics, which are specified in `two_step_zoo/evaluators/metrics.py`.
The following choices are available:

- `ave_histogram_difference`: For calorimeter shower datasets, computes the chi-squared value between histograms of real and generated shower features, and returns the average over several features. Only for two-step modules and density estimator modules.
- `shower_classifier_auc`: For calorimeter shower datasets, trains a classifier on real and generated images, and returns the ROC-AUC of classifier predictions on a test set. Only for two-step modules and density estimator modules.
- `fid`: For image datasets, and all modules which are generative models
- `precision_recall_density_coverage`: For image datasets, and all modules which are generative models
- `log_likelihood`: For two-step modules and density estimators with a `log_prob` method
- `l2_reconstruction_error`: For two-step modules and generalized autoencoders with a `rec_error` method
- `loss`: For all density estimator and generalized autoencoder modules, besides those trained using multiple loss functions such as AVB or WAE
- `likelihood_ood_acc`: Reports the accuracy of OOD detection using likelihoods in the ambient space. Available for all two-step modules and density estimators
- `likelihood_ood_acc_low_dim`: For two-step modules, reports the accuracy of OOD detection using likelihoods in the latent space

Test and validation metrics (besides OOD detection) should always be specified in a list.
In the config files, this is easily specified as e.g. below:

    {
        ...
        "test_metrics": ["loss", "ave_histogram_difference", "l2_reconstruction_error"],
        ...
    }

where `...` refers to the rest of the config arguments.
At the command line it is a bit more challenging, as we need to be careful about parsing list arguments.
Doing something as e.g. below does the trick, as we ensure that we test `ave_histogram_difference` and `l2_reconstruction_error` for the two-step module:

    ./main.py ... --shared-config test_metrics="['ave_histogram_difference', 'l2_recontruction_error']" ...


## Usage - `single_main.py`

We also provide functionality for training standard, single-step deep generative models in `single_main.py`.
These generally act as baselines for our two-step approaches.
The usage is as follows:

    ./single_main.py --model <model> --dataset <dataset> [--is-gae]

where:

- `model` is any of the following: `ae`, `arm`, `avb`, `bigan`, `ebm`, `flow`, `vae`, or `wae`
- `dataset` is any of the datasets listed in the `main.py` usage
- `--is-gae` is a flag that should be included when the model is a GAE, as it allows indexing the correct set of config files and saves the config as JSON under the name `gae_config.json`

As with `main.py`, launching this command will produce a run directory containing the same elements as before.
Also like `main.py`, `single_main.py` maintains the same behaviour for the following command line flags:

- `--load-dir`
- `--max-epochs-loaded`
- `--load-best-valid-first`
- `--only-test`
- `--test-ood`

Lastly, a similar behaviour for updating config arguments at the command line is achieved using the flag `--config <key>=<value>`, where `<key>` is a hyperparameter name and `<value>` is the value to update it to.

## Testing

We also provide tests for our codebase using the `pytest` framework.
We have two separate types of test, all coded within the `tests` directory.

The first type tests basic functionality of individual components of the code and can be called simply using the command

    pytest

from this directory. These tests should be fairly quick to run, on the order of a handful of seconds.

The second type tests overall code functionality, including training several types of models for several epochs, along with saving/loading/checkpointing.
These tests, which live in the file `test_cmds.py`, can be run via

    pytest -m cmd

again from this directory. These tests may take several minutes to run.

## Notebooks and `load_run.py`

We have included two `jupyter` notebooks with this repository within the `notebooks` directory.

The first, entitled __TODO__, performs the 2D circle example from the paper end-to-end.

The second, entitled `ood_histogram.ipynb`, demonstrates how to obtain the OOD detection histograms from the paper.
We have provided this functionality in a notebook since the histograms require some manual specification.
This notebook also demonstrates how to load a previously-trained module into a python script via the `load_run` function within `load_run.py`.
This may be useful for post-hoc inspection beyond just producing OOD detection histograms.

## Worked Examples

The examples below show how to accomplish some of the behaviour described above.

### Train a single-step GAE model

__Example__: VAE on photon showers

    python single_main.py --dataset photons1 --model vae --is-gae

### Train a single-step DE model

__Example__: NF on pion showers

    python single_main.py --dataset pions1 --model flow

### Train a two-step model

__Example__: NF+VAE+NF on photon showers (Dataset 1)

    python main.py --dataset photons1 --epl_model=flow --gae-model vae --de-model flow

### Load a pre-trained GAE, and train a DE to create a two-step model

__Example__: AVB+VAE on FMNIST

First train the AVB using `single_main.py` as in the first example:

    python single_main.py --dataset fashion-mnist --model avb --is-gae

Then locate the corresponding folder in `runs/` and use its name as an argument below.

Train a DE on the latents from a fixed GAE:

    python main.py --load-dir runs/<avb model folder> --load-pretrained-gae --load-best-valid-first --freeze-pretrained-gae --de-model vae

### Test OOD detection on a trained model

__Example__: AVB on FMNIST

Locate the AVB folder trained using `single_main.py` as in the first example:

    python single_main.py --load-dir runs/<avb model folder> --load-best-valid-first --only-test --test-ood

__Example__: AVB+VAE on FMNIST

Locate the AVB+VAE folder trained using `main.py`:

    python main.py --load-dir runs/<avb vae model folder> --load-best-valid-first --only-test --test-ood

## BibTeX

    @article{cresswell2022caloman,
        title={{CaloMan}: Fast generation of calorimeter showers with density estimation on learned manifolds},
        author={Cresswell, Jesse C and Ross, Brendan Leigh and Loaiza-Ganem, Gabriel and Reyes-Gonzalez, Humberto and Letizia, Marco and Caterini, Anthony L},
        journal={Machine Learning and the Physical Sciences Workshop at NeurIPS 2022},
        year={2022}
    }
