from typing import cast, List, Optional

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.plugins import Plugins
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import (
    search_space_to_config_space,
)

from hydra_plugins.hydra_smac_sweeper.submitit_runner import SubmititRunner
from smac.scenario.scenario import Scenario
from smac.facade.smac_mf_facade import SMAC4MF


class SMACSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        scenario: DictConfig,
        n_jobs: int,
        seed: Optional[int] = None,
        intensifier: Optional[DictConfig] = None,
        budget_variable: Optional[str] = None,
    ) -> None:
        self.cs = search_space_to_config_space(search_space, seed)
        self.scenario = scenario
        self.intensifier_kwargs = intensifier
        self.n_jobs = n_jobs
        self.seed = seed
        self.budget_variable = budget_variable
        self.rng = np.random.RandomState(self.seed)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        scenario = Scenario(
            dict(
                cs=self.cs,
                output_dir=self.config.hydra.sweep.dir,
                **cast(
                    dict,
                    OmegaConf.to_container(
                        self.scenario, resolve=True, enum_to_str=True
                    ),
                ),
            )
        )
        smac = SMAC4MF(
            scenario=scenario,
            intensifier_kwargs=self.intensifier_kwargs,
            rng=self.rng,
            tae_runner=SubmititRunner,
            tae_runner_kwargs=dict(
                n_jobs=self.n_jobs,
                launcher=self.launcher,
                budget_variable=self.budget_variable,
                ta=self.task_function,
            ),
        )
        incumbent = smac.optimize()
        return incumbent
