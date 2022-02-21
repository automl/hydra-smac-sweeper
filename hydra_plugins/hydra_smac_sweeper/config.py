from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf


@dataclass
class SMACSweeperConfig:
    _target_: str = (
        "hydra_plugins.hydra_smac_sweeper.smac_sweeper.SMACSweeper"
    )
    search_space: Dict[str, Any] = field(default_factory=dict)
    scenario: Dict[str, Any] = field(default_factory=dict)
    n_jobs: int = 1
    seed: Optional[int] = None
    budget_variable: str = "budget"
    intensifier: Optional[Dict[str, Any]] = None


ConfigStore.instance().store(group="hydra/sweeper",
                             name="SMAC", node=SMACSweeperConfig, provider="hydra_smac_sweeper")


@dataclass
class LauncherConfig(SlurmQueueConf):
    _target_: str = (
        "hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher.SubmititSmacLauncher"
    )

    progress: str = "interactive"  # basic, interactive
    progress_slurm_refresh_interval: int = 2

ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_smac_slurm",
    node=LauncherConfig(),
    provider="hydra_smac_sweeper",
)
