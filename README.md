# Hydra-SMAC-Sweeper

This plugin enables [Hydra](https://github.com/facebookresearch/hydra) applications to use 
[SMAC](https://github.com/automl/SMAC3) for hyperparameter optimization.

SMAC (Sequential Model-Based Algorithm Configuration) allows to optimize arbitrary algorithms.
Its core consists of Bayesian Optimization in combination with an aggressive racing mechanism 
to efficiently decide which of two configurations performs better.
SMAC is a *minimizer*.

The Hydra SMAC sweeper can parallelize the evaluations of the hyperparameters on your
local machine or on a slurm cluster.
The sweeper supports every SMAC facade.

## Installation
First, make sure to install the newest SMAC and checkout branch `development`:
```bash
git clone git@github.com:automl/SMAC3.git
cd SMAC3
git checkout development
pip install .
```

For the Hydra-SMAC-Sweeper please clone the repository first:
```bash
git clone git@github.com:automl/hydra-smac-sweeper.git
cd hydra-smac-sweeper
```
In your virtual environment, install via pip:
```bash
pip install -e .
```

Please find standard approaches for configuring hydra plugins
[here](https://hydra.cc/docs/patterns/configuring_plugins/).


## How the Hydra-SMAC-Sweeper works and Setting Up the Cluster
If you want to optimize your hydra application with the Hydra-SMAC-Sweeper, hydra and SMAC starts locally on your machine.
Then, depending on your dask client setup, it will either run locally, possibly using multi-processing, or on a cluster.
SMAC will create jobs/processes for the specified number of workers and will keep them open for the specified time frames.
Then dask can schedule smaller jobs on the created workers. 
This is especially useful if we have a lot of cheap function evaluations which would otherwise affect job priorities on the cluster.


### Run on Cluster (Slurm Example)
In order to run SMAC's function evaluations on the cluster, we need to setup the dask client and dask cluster.

For the setup, we need to add the dask client configuration to the `smac_kwargs` like so:
```yaml
hydra:
  sweeper:
    smac_kwargs:
      dask_client:
        _target_: dask.distributed.Client
        address: ${create_cluster:${cluster},${hydra.sweeper.scenario.n_workers}}
``` 

The cluster is automatically created from the config node `cluster` and the number of workers defined in the scenario.
This is an example configuration for the cluster itself, found in [examples/configs/hpc.yaml](examples/configs/hpc.yaml).

```yaml
# @package _global_
cluster:
  _target_: dask_jobqueue.SLURMCluster
  queue: cpu_short
  #  account: myaccount
  cores: 1
  memory: 1 GB
  walltime: 00:30:00
  processes: 1
  log_directory: tmp/smac_dask_slurm
```
You can specify any kwargs available in `dask_jobqueue.SLURMCluster`.

### Run Local
You can also run it locally by specifying the dask client to be `null`, e.g.
```bash
python examples/mlp.py hydra.sweeper.smac_kwargs.dask_client=null -m
```

Or in the config file:
```yaml
hydra:
  sweeper:
    smac_kwargs:
      dask_client: null
``` 


## Usage
In your yaml-configuration file, set `hydra/sweeper` to `SMAC`:
```yaml
defaults:
  - override hydra/sweeper: SMAC
```
You can also add `hydra/sweeper=SMAC` to your command line.

## Hyperparameter Search Space
SMAC offers to optimize several types of hyperparameters: uniform floats, integers, categoricals
and can even manage conditions and forbiddens.
The definition of the hyperparameters is based on [ConfigSpace](https://github.com/automl/ConfigSpace/).
The syntax of the hyperparameters is according to ConfigSpace's json serialization.
Please see their [user guide](https://automl.github.io/ConfigSpace/master/User-Guide.html)
for more information on how to configure hyperparameters.

You can provide the search space either as a path to a json file stemming from ConfigSpace's [serialization](https://automl.github.io/ConfigSpace/main/api/serialization.html) or you can directly specify your search space in your yaml configuration files.

Your yaml-configuration file must adhere to following syntax:
```yaml
hydra:
  sweeper:
    ...
    search_space:
      hyperparameters:  # required
        hyperparameter_name_0:
          ...
        hyperparameter_name_1:
          ...
        ...
      conditions:  # optional
        - ...
        - ...
      forbiddens:  # optional
        - ...
        - ...
      
```
The fields `conditions` and `forbiddens` are optional. Please see this 
[example](https://github.com/automl/hydra-smac-sweeper/blob/main/examples/configs/mlp.yaml)
for the exemplary definition of conditions.

Defining a uniform integer parameter is easy:
```yaml
n_neurons:
  type: uniform_int  # or have a float parameter by specifying 'uniform_float'
  lower: 8
  upper: 1024
  log: true  # optimize the hyperparameter in log space
  default_value: ${n_neurons}  # you can set your default value to the one normally used in your config
```
Same goes for categorical parameters:
```yaml
activation:
  type: categorical
  choices: [logistic, tanh, relu]
  default_value: ${activation}
```

See below for two exemplary search spaces.


## Examples
You can find examples in this [directory](https://github.com/automl/hydra-smac-sweeper/tree/main/examples).

### Branin (Synthetic Function)
The first example is optimizing (minimizing) a synthetic function (`branin.py` with
the yaml-config `configs/branin.yaml`).
Branin has two (hyper-)parameters, `x0` and `x1` which we pass via the hydra config.
For the hyperparameter optimization (or sweep) we can easily define the search
space for the uniform hyperparameters in the config file:
```yaml
hydra:
  sweeper:
    ...
    search_space:
      hyperparameters:
        x0:
          type: uniform_float
          lower: -5
          upper: 10
          log: false
        x1:
          type: uniform_float
          lower: 0
          upper: 15
          log: false
          default_value: 2
```

To optimize Branin's hyperparameters, call
```bash
python examples/branin.py --multirun
```

### Optimizing an MLP
Example for optimizing a Multi-Layer Perceptron (MLP) using multiple budgets
(`mlp.py` with the yaml-config `configs/mlp.yaml`).
The search space is hierarchical: Some options are only available if other categorical
choices are selected.
The budget variable is set by the intensifier for each run and can be specified in the sweeper config.
```yaml
hydra:
  sweeper:
    budget_variable: max_epochs
    ...
    search_space:
      hyperparameters:
        ...
      conditions:
        - child: batch_size  # only adapt the batch size if we use sgd or adam as a solver
          parent: solver
          type: IN
          values: [sgd, adam]
        - child: learning_rate  # only adapt the learning_rate if use sgd as a solver
          parent: solver
          type: EQ
          value: sgd
        - child: learning_rate_init
          parent: solver
          type: IN
          values: [sgd, adam]
```

TODO: slurm config


## Necessary Configuration Keys
In order to let SMAC successfully interact with your hydra main function, you need to use following configuration keys in your main DictConfig:

- seed: SMAC will set DictConfig.seed and pass it to your main function.

## Multi-Fidelity Optimization
In order to use multi-fidelity, you need to use `cfg.budget` to set your budget at the DictConfig's root level.
You can find an example in `examples/mlp.py` and `examples/configs/mlp.yaml` to see how we set the budget variable.

## Using Instances
In order to use instances, you need to use `cfg.instance` to set your instance in your main function.



## Notes
python examples/mlp.py hydra.sweeper.smac_kwargs.dask_client=null -m


