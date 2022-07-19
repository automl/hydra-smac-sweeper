from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf, SlurmQueueConf, LocalQueueConf


@dataclass
class SMACSweeperConfig:
    _target_: str = (
        "hydra_plugins.hydra_smac_sweeper.smac_sweeper.SMACSweeper"
    )
    search_space: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    n_trials: Optional[int] = None
    n_jobs: int = 1
    smac_class: Optional[str] = None
    smac_kwargs: Optional[Dict] = None
    budget_variable: Optional[str] = None  # TODO document budget var


ConfigStore.instance().store(group="hydra/sweeper",
                             name="SMAC", node=SMACSweeperConfig, provider="hydra_smac_sweeper")


@dataclass
class LauncherConfigMixin(BaseQueueConf):
    progress: str = "rich"  # possible values: basic, rich
    progress_slurm_refresh_interval: int = 15 # in seconds


@dataclass
class SlurmLauncherConfig(LauncherConfigMixin, SlurmQueueConf):
    _target_: str = (
        "hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher.SMACSlurmLauncher"
    )


ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_smac_slurm",
    node=SlurmLauncherConfig(),
    provider="hydra_smac_sweeper",
)


@dataclass
class LocalLauncherConfig(LauncherConfigMixin, LocalQueueConf):
    _target_: str = (
        "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SMACLocalLauncher"
    )


ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_smac_local",
    node=LocalLauncherConfig(),
    provider="hydra_smac_sweeper",
)
