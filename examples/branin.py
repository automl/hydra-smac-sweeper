"""
Branin
^^^^^^

This file is a wrapper used by SMAC to optimize parameters on the branin function.
To run this example in the terminal, execute:

.. code-block:: bash

    cd examples/commandline
    python ../../scripts/smac.py --scenario branin/scenario.txt


Inside the scenario, this file and also ``configspace.pcs`` is referenced and therefore used
for the optimization. A full call by SMAC looks like this:

.. code-block:: bash

    <algo>           <instance> <instance specific> <cutoff time>  <runlength> <seed> <parameters>
    python branin.py 0          0                   9999999        0           12345  -x1 0 -x2 0


Since SMAC processes results from the commandline, print-statements are
crucial. The format of the results must be the following to ensure correct usage:

.. code-block:: bash

    Result for SMAC: <STATUS>, <runtime>, <runlength>, <quality>, <seed>, <instance-specifics>

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
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x1 - b * x0 ** 2 + c * x0 - r) ** 2 + \
        s * (1 - t) * np.cos(x0) + s

    return ret


if __name__ == '__main__':
    branin()
