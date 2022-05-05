"""
MLP with Multi-Fidelity
^^^^^^^^^^^^^^^^^^^^^^^
Same as mlp example, except, that nested configs & arrays are supported.
"""
__copyright__ = "Copyright 2022, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import warnings
import numpy as np

from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier

import hydra
from omegaconf import DictConfig, OmegaConf

digits = load_digits()


# Target Algorithm
@hydra.main(config_path="configs", config_name="nested")
def mlp_from_cfg(cfg: DictConfig):
    """
    Creates a MLP classifier from sklearn and fits the given data on it.

    Parameters
    ----------
    cfg: Configuration
        configuration chosen by smac
    seed: int or RandomState
        used to initialize the rf's random generator
    budget: float
        used to set max iterations for the MLP

    Returns
    -------
    float
    """
    log.info(OmegaConf.to_yaml(cfg))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        mlp = MLPClassifier(
            hidden_layer_sizes=cfg.model.neuorons,
            solver=cfg.optimizer.solver,
            batch_size=cfg.batch_size,
            activation=cfg.model.activation,
            learning_rate=cfg.optimizer.learning_rate,
            learning_rate_init=cfg.optimizer.learning_rate_init,
            max_iter=int(np.ceil(cfg.max_epochs)),
            random_state=cfg.seed,
            alpha=cfg.model.alpha
        )

        # returns the cross validation accuracy
        cv = StratifiedKFold(
            n_splits=cfg.cv.n_splits, random_state=cfg.seed, shuffle=cfg.cv.shuffle
        )  # to make CV splits consistent
        score = cross_val_score(
            mlp, digits.data, digits.target, cv=cv, error_score="raise"
        )

    return 1 - np.mean(score)


if __name__ == "__main__":
    mlp_from_cfg()
