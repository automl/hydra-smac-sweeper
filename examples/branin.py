"""
Branin
^^^^^^
"""

import logging
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

__copyright__ = "Copyright 2022, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="branin")
def branin(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    x0 = cfg.x0
    x1 = cfg.x1
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    ret = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s

    return ret


if __name__ == "__main__":
    branin()
