from __future__ import annotations

from typing import Any

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig


class SMACSweeper(Sweeper):
    def __init__(self, *args: Any, **kwargs: dict[Any, Any]) -> None:
        from .smac_sweeper_backend import SMACSweeperBackend

        self.sweeper = SMACSweeperBackend(*args, **kwargs)  # type: ignore[arg-type]

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
        self.sweeper.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments: list[str]) -> Any:
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

        """
        return self.sweeper.sweep(arguments)
