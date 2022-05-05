import os

from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Union

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
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import search_space_to_config_space

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


def test_smac_sweeper_args():
    # TODO: test with Scenario kwargs provided
    # TODO: test without Scenario kwargs provided
    # TODO: test with SMAC class provided
    # TODO: test without SMAC class provided
    pass


@mark.parametrize("with_commandline", (True,))
def test_smac_example(with_commandline: bool, tmpdir: Path) -> None:
    study_name = "test-smac-example"
    cmd = [
        "examples/branin.py",
        "hydra.run.dir=" + str(tmpdir),
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.sweeper.scenario.runcount_limit=10",
        "hydra.sweeper.seed=123",
        "hydra.launcher.partition=cpu_short",
        # "hydra/launcher=joblib",
        # "~hydra.launcher.partition",
        "--multirun",
    ]
    run_python_script(cmd, allow_warnings=True)
    # returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")
    # study = optuna.load_study(storage=storage, study_name=study_name)
    # best_trial = study.best_trial
    # assert isinstance(returns, DictConfig)
    # assert returns.name == "optuna"
    # assert returns["best_params"]["x"] == best_trial.params["x"]
    # assert returns["best_value"] == best_trial.value
    # # Check the search performance of the TPE sampler.
    # # The threshold is the 95th percentile calculated with 1000 different seed values
    # # to make the test robust against the detailed implementation of the sampler.
    # # See https://github.com/facebookresearch/hydra/pull/1746#discussion_r681549830.
    # assert returns["best_value"] <= 2.27


