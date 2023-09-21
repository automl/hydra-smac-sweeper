from typing import Any, Dict, Optional

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class TargetFunctionConfig:
    _target_: str = "hydra_plugins.hydra_smac_sweeper.smac_sweeper_backend.TargetFunction"
    _partial_: bool = True

@dataclass
class SMACSweeperConfig:
    _target_: str = "hydra_plugins.hydra_smac_sweeper.smac_sweeper.SMACSweeper"
    target_function: TargetFunctionConfig = TargetFunctionConfig
    search_space: Dict[str, Any] = field(default_factory=dict)
    scenario: Dict[str, Any] = field(default_factory=dict)
    smac_class: Optional[str] = None
    smac_kwargs: Optional[Dict] = None



ConfigStore.instance().store(group="hydra/sweeper", name="SMAC", node=SMACSweeperConfig, provider="hydra_smac_sweeper")
