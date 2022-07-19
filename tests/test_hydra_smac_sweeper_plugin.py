"""
Warning: If the tests fail because no slurm output could be written to disk, change the pytest directory,
e.g. by appending --basetemp=./tmp/pytest to your pytest command.
"""
import os
import glob

from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union, Dict
import json

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import (
    TSweepRunner,
    chdir_plugin_root,
    run_python_script,
)
from omegaconf import DictConfig, OmegaConf
from pytest import mark, warns

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from hydra_plugins.hydra_smac_sweeper.smac_sweeper import SMACSweeper
from hydra_plugins.hydra_smac_sweeper.smac_sweeper_backend import SMACSweeperBackend
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import search_space_to_config_space
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import SMACLocalLauncher

from smac.facade.smac_hpo_facade import SMAC4HPO

chdir_plugin_root()


def test_discovery() -> None:
    assert SMACSweeper.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ]


def create_configspace_a() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        UniformFloatHyperparameter(lower=-512, upper=512, default_value=-3, log=False, name="x0"),
        UniformFloatHyperparameter(lower=335, upper=512, default_value=400, log=True, name="x1")
    ])
    return cs


@mark.parametrize(
    "input, expected",
    [
        (
            "tests/configspace_a.json",
            create_configspace_a()
        ),
        (
            DictConfig(content={
                "hyperparameters": {  # same structure as in yaml file
                    "x0": {
                        "type": "uniform_float",
                        "lower": -512.0,
                        "upper": 512.0,
                        "default": -3.0
                    },
                    "x1": {
                        "type": "uniform_float",
                        "log": True,
                        "lower": 335,
                        "upper": 512.0,
                        "default": 400
                    }
                }
            }),
            create_configspace_a()
        )
    ]
)
def test_search_space_parsing(input: Union[str, DictConfig], expected: ConfigurationSpace) -> None:
    actual = search_space_to_config_space(search_space=input)
    assert actual == expected


@mark.parametrize(
    "kwargs",
    [
        DictConfig(
            content={
                "smac_class": None,
                "smac_kwargs": None,
                # "hydra": {
                #     "sweep": {
                #         "dir": "./tmp"
                #     }
                # }
            }
        ),
        DictConfig(
            content={
                "smac_class": "smac.facade.smac_hpo_facade.SMAC4HPO",
                "smac_kwargs": None,
            }
        ),
    ]
)
def test_smac_sweeper_args(kwargs: DictConfig):
    # kwargs = OmegaConf.to_container(cfg=kwargs, resolve=True)
    search_space = search_space_to_config_space("tests/configspace_a.json")
    search_space = "tests/configspace_a.json"
    default_kwargs = dict(
        search_space=search_space,
        n_trials=10,
        n_jobs=1,
        seed=333,
    )
    kwargs.update(default_kwargs)
    # kwargs = OmegaConf.create(kwargs)
    sweeper = SMACSweeperBackend(**kwargs)
    sweeper.config = kwargs
    sweeper.launcher = SMACLocalLauncher()
    sweeper.hydra_context = 1
    sweeper.sweep(arguments=[])
    # TODO: test with Scenario kwargs provided
    # TODO: test without Scenario kwargs provided
    # TODO: test with SMAC class provided
    # TODO: test without SMAC class provided
    # pass


@mark.parametrize("with_commandline", (True,))
def test_smac_example(with_commandline: bool, tmpdir: Path) -> None:
    study_name = "test-smac-example"
    print(tmpdir)
    cmd = [
        "examples/branin.py",
        "hydra.run.dir=" + str(tmpdir),
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.sweeper.n_trials=10",
        "hydra.sweeper.seed=123",
        "+hydra/launcher=submitit_smac_local",
        "hydra.sweeper.n_jobs=1",
        "--multirun",
    ]
    run_python_script(cmd, allow_warnings=True)
    smac_dir = glob.glob(os.path.join(tmpdir, "run_*"))[0]
    traj_fn = os.path.join(smac_dir, "traj.json")
    with open(traj_fn, "r") as file:
        lines = file.readlines()
    trajectory = [json.loads(s) for s in lines]
    last_cost = trajectory[-1]["cost"]

    # 10 trials
    assert last_cost <= 5.3719
