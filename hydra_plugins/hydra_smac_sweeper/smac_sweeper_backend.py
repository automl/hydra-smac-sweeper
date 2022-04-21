import logging
from typing import List, Optional, cast
import numpy as np
from omegaconf import DictConfig, OmegaConf

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction

from hydra_plugins.hydra_smac_sweeper.search_space_encoding import \
    search_space_to_config_space
from hydra_plugins.hydra_smac_sweeper.submitit_runner import SubmititRunner
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import SubmititSmacLauncher
from hydra_plugins.hydra_smac_sweeper.utils.smac import silence_smac_loggers

from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario
from ConfigSpace import Configuration

log = logging.getLogger(__name__)


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
        if intensifier is None:
            intensifier = {
                "initial_budget": 1,
                "max_budget": 1,
            }
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
        self.task_function = task_function  # TODO define in init
        self.sweep_dir = config.hydra.sweep.dir  # TODO define in init

    def sweep(self, arguments: List[str]) -> Optional[Configuration]:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None
        
        cast(SubmititSmacLauncher, self.launcher).global_overrides = arguments
        log.info(f"Sweep overrides: {' '.join(arguments)}")

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
        smac = SMAC4MF(  # TODO make variable
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
        silence_smac_loggers()
        incumbent = smac.optimize()
        smac.solver.stats.print_stats()
        log.info("Final Incumbent: %s", smac.solver.incumbent)
        if smac.solver.incumbent and smac.solver.incumbent in smac.solver.runhistory.get_all_configs():
            log.info("Estimated cost of incumbent: %f",
                     smac.solver.runhistory.get_cost(smac.solver.incumbent))
        return incumbent
