from typing import Any, List, Optional

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf

from ConfigSpace import ConfigurationSpace


def search_space_to_config_space(search_space: DictConfig) -> ConfigurationSpace:


    return cs


class SMACSweeperBackend(object):
    pass
    # def __init__(
    #     self,
    #     sampler: SamplerConfig,
    #     direction: Any,
    #     storage: Optional[str],
    #     study_name: Optional[str],
    #     n_trials: int,
    #     n_jobs: int,
    #     search_space: Optional[DictConfig],
    # ) -> None:
    #
    #     # Convert search space into configuration space


if __name__ == "__main__":
    fname = "../examples/configs/optimization.yaml"
    config = OmegaConf.load(fname)
