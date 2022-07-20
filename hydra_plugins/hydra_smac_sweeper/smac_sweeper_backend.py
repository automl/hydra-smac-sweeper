from typing import List, Optional, cast

import logging

import numpy as np
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_class
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import (
    search_space_to_config_space,
)
from hydra_plugins.hydra_smac_sweeper.submitit_runner import SubmititRunner
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import (
    SubmititSmacLauncherMixin,
)
from hydra_plugins.hydra_smac_sweeper.utils.smac import silence_smac_loggers
from omegaconf import DictConfig, OmegaConf
from smac.configspace import Configuration, ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_class", get_class, replace=True)


class SMACSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig,
        n_trials: int,
        n_jobs: int,
        seed: Optional[int] = None,
        smac_class: Optional[str] = None,
        smac_kwargs: Optional[DictConfig] = None,
        budget_variable: Optional[str] = None,
    ) -> None:
        # TODO document parameters
        self.configspace: Optional[ConfigurationSpace] = None
        self.search_space = search_space
        self.smac_class = smac_class
        self.smac_kwargs = smac_kwargs
        self.n_jobs = n_jobs
        self.seed = seed
        self.n_trials = n_trials
        self.budget_variable = budget_variable
        self.rng = np.random.RandomState(self.seed)

        self.task_function: Optional[TaskFunction] = None
        self.sweep_dir: Optional[str] = None

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

    def setup_smac(self):
        # Select SMAC Facade
        if self.smac_class is not None:
            smac_class = get_class(self.smac_class)
        else:
            smac_class = "smac.facade.smac_ac_facade.SMAC4AC"
            smac_class = get_class(smac_class)

        # Setup other SMAC kwargs
        smac_kwargs = {}
        if self.smac_kwargs is not None:
            smac_kwargs = OmegaConf.to_container(self.smac_kwargs, resolve=True, enum_to_str=True)

        # Instantiate Scenario
        if self.configspace is None:
            self.configspace = search_space_to_config_space(search_space=self.search_space, seed=self.seed)
        scenario_kwargs = dict(
            cs=self.configspace,
            output_dir=self.config.hydra.sweep.dir,
            ta_run_limit=self.n_trials,
        )
        scenario = smac_kwargs.get("scenario", None)
        if scenario is not None:
            scenario_kwargs.update(scenario)

        scenario = Scenario(scenario=scenario_kwargs)
        smac_kwargs["scenario"] = scenario

        smac = smac_class(
            rng=self.rng,
            tae_runner=SubmititRunner,
            tae_runner_kwargs=dict(
                n_jobs=self.n_jobs,
                launcher=self.launcher,
                budget_variable=self.budget_variable,
                ta=self.task_function,
                stats=Stats(scenario=scenario),
            ),
            **smac_kwargs,
        )
        silence_smac_loggers()

        return smac

    def sweep(self, arguments: List[str]) -> Optional[Configuration]:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        cast(SubmititSmacLauncherMixin, self.launcher).global_overrides = arguments
        log.info(f"Sweep overrides: {' '.join(arguments)}")

        smac = self.setup_smac()

        incumbent = smac.optimize()
        smac.solver.stats.print_stats()
        log.info("Final Incumbent: %s", smac.solver.incumbent)
        if smac.solver.incumbent and smac.solver.incumbent in smac.solver.runhistory.get_all_configs():
            log.info("Estimated cost of incumbent: %f", smac.solver.runhistory.get_cost(smac.solver.incumbent))
        return incumbent
