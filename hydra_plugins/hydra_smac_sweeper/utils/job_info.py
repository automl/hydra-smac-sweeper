from typing import List

from smac.runhistory import TrialInfo
from submitit.core.core import Job


class JobInfo:
    def __init__(self, idx: int, job: Job, overrides: List[str], trial_info: TrialInfo):
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
        trial_info: TrialInfo
            SMAC's trail information.
        """
        self.idx = idx
        self.job = job
        self.overrides = overrides
        self.trial_info = trial_info

    def done(self) -> bool:
        """
        Check if a job is done

        Returns
        -------
        bool
            Whether the job has finished.

        """
        return self.job.done()
