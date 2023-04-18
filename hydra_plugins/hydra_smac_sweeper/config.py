from typing import Any, Dict, Optional

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import (
    BaseQueueConf,
    LocalQueueConf,
    SlurmQueueConf,
)


@dataclass
class SMACSweeperConfig:
    _target_: str = "hydra_plugins.hydra_smac_sweeper.smac_sweeper.SMACSweeper"
    search_space: Dict[str, Any] = field(default_factory=dict)
    scenario: Dict[str, Any] = field(default_factory=dict)
    smac_class: Optional[str] = None
    smac_kwargs: Optional[Dict] = None
    budget_variable: Optional[str] = None  # TODO document budget var


ConfigStore.instance().store(group="hydra/sweeper", name="SMAC", node=SMACSweeperConfig, provider="hydra_smac_sweeper")
