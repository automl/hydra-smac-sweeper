from hydra_plugins.hydra_smac_sweeper.smac_sweeper_backend import SMACSweeperBackend
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import SMACLocalLauncher
from omegaconf import OmegaConf


def target_function(cfg, seed, budget):
    return seed + budget


if __name__ == "__main__":
    global_cfg = OmegaConf.load("/home/benjamin/Dokumente/code/tmp/hydra-smac-sweeper/examples/configs/branin.yaml")
    global_cfg.hydra.sweep.dir = "tmp"
    cfg = global_cfg.hydra.sweeper
    backend = SMACSweeperBackend(scenario=cfg.scenario, search_space=cfg.search_space)
    backend.config = global_cfg
    backend.task_function = target_function
    backend.launcher = SMACLocalLauncher()
    smac = backend.setup_smac()
