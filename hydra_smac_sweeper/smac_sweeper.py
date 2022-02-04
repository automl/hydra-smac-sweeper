from typing import Any, List, Optional

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from .config import SamplerConfig

import optuna

class SMACSweeper(Sweeper):
    """Class to interface with Optuna"""

    def __init__(
        self,
        sampler: SamplerConfig,
        direction: Any,
        storage: Optional[str],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        search_space: Optional[DictConfig],
    ) -> None:
        from .smac_sweeper_backend import SMACSweeperBackend

        self.sweeper = SMACSweeperBackend(
            sampler, direction, storage, study_name, n_trials, n_jobs, search_space
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
