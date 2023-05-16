from __future__ import annotations

from typing import Any

import logging
import warnings
from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace  # type: ignore[import]
from dask_jobqueue import JobQueueCluster  # type: ignore[import]
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_class, get_method, instantiate
from hydra_plugins.hydra_smac_sweeper.search_space_encoding import (
    search_space_to_config_space,
)
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from smac.facade.abstract_facade import AbstractFacade
from smac.runner import DaskParallelRunner
from smac.scenario import Scenario

log = logging.getLogger(__name__)


def create_cluster(cluster_cfg: DictConfig, n_workers: int = 1) -> JobQueueCluster:
    """Create Dask cluster to schedule jobs on

    Parameters
    ----------
    cluster_cfg : DictConfig
        Configuration for the cluster
    n_workers : int, optional
        Number of workers, by default 1

    Returns
    -------
    JobQueueCluster
        Dask cluster
    """
    cluster = instantiate(cluster_cfg)
    cluster.scale(jobs=n_workers)
    return cluster


OmegaConf.register_new_resolver("get_class", get_class, replace=True)
OmegaConf.register_new_resolver("get_method", get_method, replace=True)
OmegaConf.register_new_resolver("create_cluster", create_cluster, replace=True)


class TargetFunction(object):
    def __init__(self, task_function: callable, config: DictConfig) -> None:
        self.task_function = task_function
        self.config = config
    
    def __call__(
            self, config: Configuration, seed: int | None = None, budget: int | None = None, instance: str | None = None
        ) -> Any:
        # Translate SMAC's function signature back to hydra DictConfig
        cfg = self.config  # hydra config
        for k, v in dict(config).items():
            cfg[k] = v
        OmegaConf.update(cfg, "seed", seed, force_add=True)
        if "budget_variable" in cfg:
            OmegaConf.update(cfg, cfg.budget_variable, budget, force_add=True)
        OmegaConf.update(cfg, "instance", instance, force_add=True)

        return self.task_function(cfg=cfg)  # type: ignore[misc]


class SMACSweeperBackend(Sweeper):
    def __init__(
        self,
        search_space: DictConfig | str | ConfigurationSpace,
        scenario: DictConfig,
        smac_class: str | None = None,
        smac_kwargs: DictConfig | None = None,
    ) -> None:
        """
        Backend for the SMAC sweeper. Instantiate and launch SMAC's optimization.

        Parameters
        ----------
        search_space: DictConfig | str | ConfigurationSpace
            The search space, either a DictConfig from a hydra yaml config file, or a path to a json configuration space
            file in the format required of ConfigSpace, or already a ConfigurationSpace config space.
        scenario: DictConfig
            Kwargs for scenario
        smac_class: str | None
            Optional string defining the smac class, e.g. "smac.facade.smac_ac_facade.SMAC4AC".
        smac_kwargs: DictConfig | None
            Kwargs for the smac class from the yaml config file.

        Returns
        -------
        None

        """
        self.configspace: ConfigurationSpace | None = None
        self.search_space = search_space
        self.smac_class = smac_class
        self.smac_kwargs = smac_kwargs
        self.scenario = scenario
        self.seed = self.scenario.get("seed", None)

        self.task_function: TaskFunction | None = None
        self.sweep_dir: str | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """
        Setup launcher.

        Parameters
        ----------
        hydra_context: HydraContext
        task_function: TaskFunction
        config: DictConfig

        Returns
        -------
        None

        """
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.task_function = task_function
        self.sweep_dir = config.hydra.sweep.dir

    def setup_smac(self) -> AbstractFacade:
        """
        Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns
        -------
        SMAC4AC
            Instance of a SMAC facade.

        """
        assert self.task_function is not None
        # Select SMAC Facade
        if self.smac_class is not None:
            smac_class = get_class(self.smac_class)
        else:
            smac_class = "smac.facade.hyperparameter_optimization_facade.HyperparameterOptimizationFacade"
            smac_class = get_class(smac_class)

        if (
            smac_class == get_class("smac.facade.multi_fidelity_facade.MultiFidelityFacade")
            and "budget_variable" not in self.config
        ):
            raise ValueError(
                "In order to use the MultiFidelityFacade you need to provide `budget_variable` at "
                "your configs root level indicating which variable of your config is the fidelity "
                "and controls the budget."
            )

        # Setup other SMAC kwargs
        smac_kwargs = {}
        if self.smac_kwargs is not None:
            smac_kwargs = OmegaConf.to_container(self.smac_kwargs, resolve=True, enum_to_str=True)

        # Instantiate Scenario
        if self.configspace is None:
            self.configspace = search_space_to_config_space(search_space=self.search_space, seed=self.seed)
        scenario_kwargs = dict(
            configspace=self.configspace,
            output_directory=Path(self.config.hydra.sweep.dir)
            / "smac3_output",  # TODO document that output directory is automatically set
        )
        # We always expect scenario kwargs from the user
        _scenario_kwargs = OmegaConf.to_container(self.scenario, resolve=True)
        scenario_kwargs.update(_scenario_kwargs)

        scenario = Scenario(**scenario_kwargs)

        if scenario.trial_walltime_limit is not None or scenario.trial_memory_limit is not None:
            raise ValueError(
                "The hydra smac sweeper currently does not support resource "
                "limitation (scenario.trial_walltime_limit and "
                "scenario.trial_memory_limit should be None) due to "
                "pickling issues with multiprocessing."
            )
        smac_kwargs["scenario"] = scenario

        # If we have a custom intensifier we need to instantiate ourselves
        # because the helper methods in the facades expect a scenario.
        # Here it is easier to instantiate than completely via the yaml file.
        if "intensifier" in smac_kwargs and "intensifier_kwargs" in smac_kwargs:
            # Get, delete and update intensifier kwargs
            intensifier_kwargs = smac_kwargs["intensifier_kwargs"]
            del smac_kwargs["intensifier_kwargs"]
            intensifier_kwargs["scenario"] = scenario
            # Build intensifier
            intensifier = smac_kwargs["intensifier"]
            if isinstance(intensifier, str):
                intensifier = get_class(smac_kwargs["intensifier"])
            smac_kwargs["intensifier"] = intensifier(**intensifier_kwargs)

        if "initial_design" in smac_kwargs:
            initial_design = smac_kwargs["initial_design"]
            if isinstance(initial_design, str):
                initial_design = get_class(smac_kwargs["initial_design"])
            initial_design_kwargs = smac_kwargs.get("initial_design_kwargs", {})
            del smac_kwargs["initial_design_kwargs"]
            smac_kwargs["initial_design"] = initial_design(scenario=scenario, **initial_design_kwargs)

        printr(smac_class, smac_kwargs)

        target_function = TargetFunction(task_function=self.task_function, config=self.config)

        smac = smac_class(
            target_function=target_function,
            **smac_kwargs,
        )

        return smac

    def sweep(self, arguments: list[str]) -> Configuration | None:
        """
        Run optimization with SMAC.

        Parameters
        ----------
        arguments: list[str]
            Hydra overrides for the sweep. Must be empty.

        Returns
        -------
        Configuration | None
            Incumbent (best) configuration.

        Raises
        ------
        Warning
            When providing overriding arguments, override arguments do not have any effect.

        """
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        printr("Config", self.config)
        printr("Hydra context", self.hydra_context)

        if len(arguments) > 0:
            warnings.warn(f"Override arguments might not have an effect if they are a sweep. {arguments}")

        smac = self.setup_smac()

        incumbent = smac.optimize()
        if isinstance(smac._runner, DaskParallelRunner):
            smac._runner.close(force=True)
        smac._optimizer.print_stats()
        log.info(f"Final Incumbent: {incumbent}")
        if incumbent is not None:
            incumbent_cost = smac.runhistory.get_cost(incumbent)
            log.info(f"Estimated cost of incumbent: {incumbent_cost}")
        # if smac.solver.incumbent and smac.solver.incumbent in smac.solver.runhistory.get_all_configs():
        #     log.info("Estimated cost of incumbent: %f", smac.solver.runhistory.get_cost(smac.solver.incumbent))
        return incumbent
