from typing import List

from smac.runhistory.runhistory import RunInfo
from submitit.core.core import Job


class JobInfo:
    def __init__(self, idx: int, job: Job, overrides: List[str], run_info: RunInfo):
        """
        Job Info

        Parameters
        ----------
        idx: int
            Index of job.
        job: Job
            Job instance.
        overrides: List[str]
            Overrides / hydra args of this job.
        run_info: RunInfo
            SMAC's run information.
        """
        self.idx = idx
        self.job = job
        self.overrides = overrides
        self.run_info = run_info

    def done(self) -> bool:
        """
        Check if a job is done

        Returns
        -------
        bool
            Whether the job has finished.

        """
        return self.job.done()
