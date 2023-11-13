"""
Warning: If the tests fail because no slurm output could be written to disk, change the pytest directory,
e.g. by appending --basetemp=./tmp/pytest to your pytest command.
"""
from typing import Union

import json
import os
from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from examples.blackbox_branin import branin
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import chdir_plugin_root, run_python_script
from hydra.utils import get_class
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import (
    search_space_to_config_space,
)
from hydra_plugins.hydra_smac_sweeper.smac_sweeper import SMACSweeper
from hydra_plugins.hydra_smac_sweeper.smac_sweeper_backend import SMACSweeperBackend
from omegaconf import DictConfig, OmegaConf
from pytest import mark
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)

chdir_plugin_root()


def test_discovery() -> None:
    assert SMACSweeper.__name__ in [x.__name__ for x in Plugins.instance().discover(Sweeper)]


def create_configspace_a() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(lower=-512, upper=512, default_value=-3, log=False, name="x0"),
            UniformFloatHyperparameter(lower=335, upper=512, default_value=400, log=True, name="x1"),
        ]
    )
    return cs


def create_configspace_a_with_defaultdefaults() -> ConfigurationSpace:
    cs = create_configspace_a()
    cs["x0"].default_value = 0.0
    cs["x1"].default_value = 414.1497313774
    return cs


@mark.parametrize(
    "input, expected",
    [
        # Test parsing json-Configuration Space
        ("tests/configspace_a.json", create_configspace_a()),
        # Test creation of ConfigurationSpace from dict
        (
            DictConfig(
                content={
                    "hyperparameters": {  # same structure as in yaml file
                        "x0": {"type": "uniform_float", "lower": -512.0, "upper": 512.0, "default": -3.0},
                        "x1": {"type": "uniform_float", "log": True, "lower": 335, "upper": 512.0, "default": 400},
                    }
                }
            ),
            create_configspace_a(),
        ),
        # Test passing ConfigurationSpace
        (create_configspace_a(), create_configspace_a()),
        # Test if default value is set correctly
        (
            DictConfig(
                content={
                    "hyperparameters": {  # same structure as in yaml file
                        "x0": {
                            "type": "uniform_float",
                            "lower": -512.0,
                            "upper": 512.0,
                        },
                        "x1": {
                            "type": "uniform_float",
                            "log": True,
                            "lower": 335,
                            "upper": 512.0,
                        },
                    }
                }
            ),
            create_configspace_a_with_defaultdefaults(),
        ),
    ],
)
def test_search_space_parsing(input: Union[str, DictConfig], expected: ConfigurationSpace) -> None:
    actual = search_space_to_config_space(search_space=input, seed=48574)
    assert actual == expected


def test_search_space_parsing_value_error() -> None:
    some_list = list()
    try:
        _ = search_space_to_config_space(some_list)
        assert False
    except ValueError:
        assert True


@mark.parametrize(
    "kwargs",
    [
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": None,
                "search_space": "tests/configspace_a.json",
                "scenario": {"seed": 33, "deterministic": "true", "n_trials": 12, "n_workers": 1},
            }
        ),
        DictConfig(
            content={
                "smac_class": "smac.facade.hyperparameter_optimization_facade.HyperparameterOptimizationFacade",
                "smac_kwargs": None,
                "search_space": "tests/configspace_a.json",
                "scenario": {},
            }
        ),
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": None,
                "search_space": "tests/configspace_a.json",
                "scenario": {},
            }
        ),
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": {
                    "intensifier": "smac.intensifier.intensifier.Intensifier",
                    "intensifier_kwargs": {},
                },
                "search_space": "tests/configspace_a.json",
                "scenario": {},
            }
        ),
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": {
                    "intensifier": "${get_method:smac.facade.hyperparameter_"
                    "optimization_facade.HyperparameterOptimizationFacade.get_intensifier}",
                    "intensifier_kwargs": {},
                },
                "search_space": "tests/configspace_a.json",
                "scenario": {},
            }
        ),
    ],
)
def test_smac_sweeper_args(kwargs: DictConfig):
    sweeper = SMACSweeperBackend(**kwargs)
    config = OmegaConf.create({"hydra": {"sweep": {"dir": "./tmp"}}})
    sweeper.config = config
    sweeper.task_function = branin
    smac = sweeper.setup_smac()

    target_smac_class = kwargs.get("smac_class")
    if target_smac_class is None:
        assert type(smac) == HyperparameterOptimizationFacade
    else:
        assert type(smac) == get_class(target_smac_class)


@mark.parametrize(
    "kwargs",
    [
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": None,
                "search_space": "tests/configspace_a.json",
                "scenario": {},
            }
        ),
    ],
)
def test_smac_sweeper_sweep_arguments_error(kwargs: DictConfig):
    sweeper = SMACSweeperBackend(**kwargs)
    config = OmegaConf.create({"hydra": {"sweep": {"dir": "./tmp"}}})
    sweeper.config = config
    sweeper.task_function = branin
    sweeper.launcher = "dummy"
    sweeper.hydra_context = "dummy"
    with pytest.warns():
        sweeper.sweep(arguments=["nothing", "should", "go", "in", "here"])


@mark.parametrize(
    "kwargs",
    [
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": None,
                "search_space": "tests/configspace_a.json",
                "scenario": {
                    "trial_walltime_limit": 10,
                },
            }
        ),
    ],
)
def test_smac_sweeper_sweep_resourcelimitation_error(kwargs: DictConfig):
    sweeper = SMACSweeperBackend(**kwargs)
    config = OmegaConf.create({"hydra": {"sweep": {"dir": "./tmp"}}})
    sweeper.config = config
    sweeper.task_function = branin
    sweeper.launcher = "dummy"
    sweeper.hydra_context = "dummy"
    with pytest.raises(ValueError):
        sweeper.setup_smac()


@mark.parametrize("n_workers", [1, 2])
def test_smac_example(tmpdir: Path, n_workers: int) -> None:
    print(tmpdir)
    seed = 123
    cmd = [
        "examples/blackbox_branin.py",
        "hydra.run.dir=" + str(tmpdir),
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.sweeper.scenario.n_trials=10",
        f"hydra.sweeper.scenario.seed={seed}",
        "+hydra.sweeper.scenario.name=testrun",
        f"hydra.sweeper.scenario.n_workers={n_workers}",
        "hydra.sweeper.smac_kwargs.dask_client=null",  # execute local
        "--multirun",
    ]
    run_python_script(cmd, allow_warnings=True)
    smac_dir = os.path.join(tmpdir, "smac3_output", "testrun")
    runhistory_fn = os.path.join(smac_dir, str(seed), "runhistory.json")
    with open(runhistory_fn, "r") as file:
        runhistory = json.load(file)
    stats = runhistory["stats"]
    # Check if 10 runs have finished
    assert stats["finished"] == 10
