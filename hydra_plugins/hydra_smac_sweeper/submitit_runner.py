import time
import typing

from hydra.core.utils import JobStatus
from omegaconf import OmegaConf
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType
from smac.tae.base import BaseRunner
from submitit import Job
import pandas as pd
from hydra_plugins.hydra_smac_sweeper.utils.job_info import JobInfo
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import SubmititSmacLauncher

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def flatten_dict(d):
    return list(pd.json_normalize(d).T.to_dict().values())[0]


class SubmititRunner(BaseRunner):

    def __init__(
        self,
        ta: typing.Callable,
        launcher: SubmititSmacLauncher,
        n_jobs: int,
        budget_variable: typing.Optional[str] = None,
        output_directory: typing.Optional[str] = None,
        **kwargs
    ):
        single_worker = ExecuteTAFuncDict(ta=ta, **kwargs)
        super().__init__(
            ta=single_worker.ta,
            stats=single_worker.stats,
            run_obj=single_worker.run_obj,
            par_factor=single_worker.par_factor,
            cost_for_crash=single_worker.cost_for_crash,
            abort_on_first_run_crash=single_worker.abort_on_first_run_crash,
        )

        # The single worker, which is replicated on a need
        # basis to every compute node
        self.single_worker = single_worker

        self.output_directory = output_directory

        self.launcher = launcher
        self.n_jobs = n_jobs
        self.job_idx = 0
        self.running_job_info = []
        self.results: typing.List[JobInfo] = []
        self.base_cfg_flat = flatten_dict(OmegaConf.to_container(launcher.config, enum_to_str=True))
        self.budget_variable = budget_variable

        if launcher.params['progress'] == 'rich':
            # TODO: add rich to requirements
            from .utils.rich_progress import RichProgress
            self.progress_handler = RichProgress()
        else:
            self.progress_handler = None

    def submit_run(self, run_info: RunInfo) -> None:
        """This function submits a configuration
        embedded in a run_info object, and uses one of the workers
        to produce a result locally to each worker.
        The execution of a configuration follows this procedure:
        1.  SMBO/intensifier generates a run_info
        2.  SMBO calls submit_run so that a worker launches the run_info
        3.  submit_run internally calls self.run(). it does so via a call to self.run_wrapper()
        which contains common code that any run() method will otherwise have to implement, like
        capping check.
        Child classes must implement a run() method.
        All results will be only available locally to each worker, so the
        main node needs to collect them.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it
        """
        # Check for resources or block till one is available
        assert self.launcher.config
        while not self._workers_available():
            self.wait()
            self._extract_completed_runs_from_futures()
        overrides = self._diff_overrides(run_info)
        if self.budget_variable is not None:
            overrides = [
                override + (f"{self.budget_variable}={run_info.budget}",) for override in overrides]
        jobs = self.launcher.launch(overrides, self.job_idx)

        for i, (override, job) in enumerate(zip(overrides, jobs)):
            idx = self.job_idx + i
            job_info = JobInfo(idx=idx, job=job, overrides=override, run_info=run_info)
            self.running_job_info.append(job_info)

        self.job_idx += len(jobs)

    def _diff_overrides(self, run_info: RunInfo):
        run_info_cfg_flat = flatten_dict(run_info.config.get_dictionary())
        diff_overrides = [tuple(f"{key}={val1}" for key, val1 in run_info_cfg_flat.items(
        ) if val1 != self.base_cfg_flat[key])]
        return diff_overrides

    def _workers_available(self) -> bool:
        if len(self.running_job_info) < self.n_jobs:
            return True
        return False

    def get_finished_runs(self) -> typing.List[typing.Tuple[RunInfo, RunValue]]:
        """This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results list is emptied and all RunValues produced by running run() are
        returned.
        Returns
        -------
            List[RunInfo, RunValue]: A list of RunValues (and respective RunInfo), that is,
                the results of executing a run_info
            a submitted configuration
        """

        # Proactively see if more configs have finished
        self._extract_completed_runs_from_futures()

        results_list: typing.List[typing.Tuple[RunInfo, RunValue]] = []
        while self.results:
            # run_info, job = self.results.pop()
            job_info = self.results.pop()
            run_info, job = job_info.run_info, job_info.job
            ret = job.result()
            endtime = time.time()
            run_value = RunValue(cost=ret.return_value, time=endtime - job._start_time, status=StatusType.SUCCESS if ret.status == JobStatus.COMPLETED else StatusType.CRASHED,
                                 starttime=job._start_time, endtime=endtime, additional_info=None)
            results_list.append((run_info, run_value))
        return results_list

    def _extract_completed_runs_from_futures(self) -> None:
        """
        A run is over, when a future has done() equal true.
        This function collects the completed futures and move
        them from self.futures to self.results.
        We make sure futures never exceed the capacity of
        the scheduler
        """
        jobs_done = [job for job in self.running_job_info if job.done()]
        for job_info in jobs_done:
            # for future in futures:
            self.results.append(job_info)
            self.running_job_info.remove(job_info)

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.
        This class waits until 1 run completes
        """
        if len(self.running_job_info):

            if self.progress_handler is not None:
                progress_slurm_refresh_interval = self.launcher.params['progress_slurm_refresh_interval']
                job_idx, jobs, job_overrides = zip(* [(j.idx, j.job, j.overrides) for j in self.running_job_info])
                self.progress_handler.loop(
                    job_idx, jobs, job_overrides, 
                    auto_stop=False,
                    progress_slurm_refresh_interval=progress_slurm_refresh_interval,
                    return_first_finished=True
                    )
            else:
                while True:
                    if any([[f.done() for f in self.running_job_info]]):
                        return
                    
                    time.sleep(1)

    def pending_runs(self) -> bool:
        """
        Whether or not there are configs still running. Generally if the runner is serial,
        launching a run instantly returns it's result. On parallel runners, there might
        be pending configurations to complete.
        """
        # If there are futures available, it translates
        # to runs still not finished/processed
        return len(self.running_job_info) > 0

    def run(
        self, config: Configuration,
        instance: str,
        cutoff: typing.Optional[float] = None,
        seed: int = 12345,
        budget: typing.Optional[float] = None,
        instance_specific: str = "0",
    ) -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """
        This method only complies with the abstract parent class. In the parallel case,
        we call the single worker run() method
        Parameters
        ----------
            config : Configuration
                dictionary param -> value
            instance : string
                problem instance
            cutoff : float, optional
                Wallclock time limit of the target algorithm. If no value is
                provided no limit will be enforced.
            seed : int
                random seed
            budget : float, optional
                A positive, real-valued number representing an arbitrary limit to the target
                algorithm. Handled by the target algorithm internally
            instance_specific: str
                instance specific information (e.g., domain file or solution)
        Returns
        -------
            status: enum of StatusType (int)
                {SUCCESS, TIMEOUT, CRASHED, ABORT}
            cost: float
                cost/regret/quality (float) (None, if not returned by TA)
            runtime: float
                runtime (None if not returned by TA)
            additional_info: dict
                all further additional run information
        """
        return self.single_worker.run(
            config=config,
            instance=instance,
            cutoff=cutoff,
            seed=seed,
            budget=budget,
            instance_specific=instance_specific,
        )

    def num_workers(self) -> int:
        return self.n_jobs
