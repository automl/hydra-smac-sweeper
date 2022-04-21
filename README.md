# hydra-smac-sweeper

This plugin enables [Hydra](https://github.com/facebookresearch/hydra) applications to use 
[SMAC](https://github.com/automl/SMAC3) for hyperparameter optimization.

SMAC (Sequential Model-Based Algorithm Configuration) allows to optimize arbitrary algorithms.
Its core consists of Bayesian Optimization in combination with an aggressive racing mechanism 
to efficiently decide which of two configurations performs better.
SMAC is a *minimizer*.

The Hydra SMAC sweeper can parallelize the evaluations of the hyperparameters on slurm.
The sweeper currently only supports the multi-fidelity facade (`SMAC4MF`). TODO

## Installation
Please clone the repository first:
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

TODO: move repo to automl
TODO: add license / MIT License right now

## Usage
In your yaml-configuration file, set `hydra/sweeper` to `SMAC`:
```yaml
defaults:
  - override hydra/sweeper: SMAC
```
You can also add `hydra/sweeper=SMAC` to your command line.

## Hyperparameter Search Space
TODO add info that search space can be json file

SMAC offers to optimize several types of hyperparameters: uniform floats, integers, categoricals
and can even manage conditions and forbiddens.
The definition of the hyperparameters is based on [ConfigSpace](https://github.com/automl/ConfigSpace/).
The syntax of the hyperparameters is according to ConfigSpace's json serialization.
Please see their [user guide](https://automl.github.io/ConfigSpace/master/User-Guide.html)
for more information on how to configure hyperparameters.

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
python branin.py --multirun
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
