import logging
import os
from pathlib import Path
from typing import Any, List, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from omegaconf import OmegaConf

from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher
from submitit import Job

log = logging.getLogger(__name__)


class SubmititSmacLauncher(SlurmLauncher):
    global_overrides: List[str] = []

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> List[Job[JobReturn]]:
        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        submitit_smac_launcher_keys = {'progress', 'progress_slurm_refresh_interval'}
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys and x not in submitit_smac_launcher_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys and x not in submitit_smac_launcher_keys
        }
        executor.update_parameters(**params)

        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            if self.params['progress'] == 'basic':
                lst = " ".join(filter_overrides(overrides))
                log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides) + self.global_overrides,
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        jobs = executor.map_array(self, *zip(*job_params))
        return jobs
