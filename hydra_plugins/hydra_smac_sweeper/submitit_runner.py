from __future__ import annotations

from typing import Any, Callable, Iterator

import time

import pandas as pd
from hydra.core.utils import JobStatus
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import (
    SMACLocalLauncher,
    SMACSlurmLauncher,
)
from hydra_plugins.hydra_smac_sweeper.utils.job_info import JobInfo
from omegaconf import OmegaConf
from ConfigSpace import Configuration
from smac.runhistory import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType
from smac.runner.abstract_runner import AbstractRunner
from smac.runner import TargetFunctionRunner
from smac.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def flatten_dict(d):
    return list(pd.json_normalize(d).T.to_dict().values())[0]


class SubmititRunner(AbstractRunner):  # TODO check if correct class to inherit from
    def __init__(
        self,
        target_function: Callable,
        scenario: Scenario,
        launcher: SMACLocalLauncher | SMACSlurmLauncher,
        budget_variable: str | None,
        required_arguments: list[str] = [],
        n_jobs: int = 1,
        output_directory: str | None = None,
        **kwargs: dict[Any, Any],
    ) -> None:
        """
        Interface class to handle the execution of SMAC' configurations via submitit.


        Parameters
        ----------
        ta: Callable
            Target algorithm (the method we want to optimize) as a callable. Must receive
            as an argument a omegaconf.DictConfig.
        launcher: SMACLocalLauncher | SMACSlurmLauncher
            Launches the jobs, either locally (SMACLocalLauncher) or on a slurm cluster (SMACSlurmLauncher).
        n_jobs: int
            Number of parallel jobs/workers.
        budget_variable: str | None
            Name of the variable controlling the budget, e.g. max_epochs. Only relevant for multi-fidelity methods.
        output_directory: str | None
            Path to output directory. Not used right now.  # TODO check why we have output directory here
        kwargs: Dict[Any, Any]
            Kwargs for smac.tae.execute_func.ExecuteTAFuncDict.

        Returns
        -------
        None

        """
        single_worker = TargetFunctionRunner(
            scenario=scenario,
            target_function=target_function,
            required_arguments=required_arguments
        )
        # super().__init__(
        #     ta=single_worker.ta,
        #     stats=single_worker.stats,
        #     run_obj=single_worker.run_obj,
        #     par_factor=single_worker.par_factor,
        #     cost_for_crash=single_worker.cost_for_crash,
        #     abort_on_first_run_crash=single_worker.abort_on_first_run_crash,
        # )

        super().__init__(
            scenario=single_worker._scenario,
            required_arguments=single_worker._required_arguments,
        )

        # The single worker, which is replicated on a need
        # basis to every compute node
        self.single_worker = single_worker

        self.output_directory = output_directory

        self.launcher = launcher
        self.n_jobs = n_jobs
        self.job_idx = 0
        self.running_job_info = []
        self.results: list[JobInfo] = []
        self.base_cfg_flat = flatten_dict(OmegaConf.to_container(launcher.config, enum_to_str=True))
        self.budget_variable = budget_variable

        # TODO don't hardcode
        if "progress" not in launcher.params:
            launcher.params["progress"] = "rich"
        if launcher.params["progress"] == "rich":
            # TODO: add rich to requirements
            from .utils.rich_progress import RichProgress

            self.progress_handler = RichProgress()
        else:
            self.progress_handler = None
        del launcher.params["progress"]

    def submit_trial(self, trial_info: TrialInfo) -> None:
        """This function submits a configuration embedded in a TrialInfo object, and uses one of the
        workers to produce a result (such result will eventually be available on the `self._results_queue`
        FIFO).

        This interface method will be called by SMBO, with the expectation
        that a function will be executed by a worker. What will be executed is dictated by trial_info, and "how" will it
        be executed is decided via the child class that implements a run() method.

        Because config submission can be a serial/parallel endeavor, it is expected to be implemented by a child class.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.

        Return
        ------
        None
        """
        # Check for resources or block till one is available
        assert self.launcher.config
        while not self._workers_available():
            self.wait()
            self._extract_completed_runs_from_futures()
        overrides = self._diff_overrides(trial_info)
        if self.budget_variable is not None:
            overrides = [override + (f"{self.budget_variable}={trial_info.budget}",) for override in overrides]
        jobs = self.launcher.launch(overrides, self.job_idx)

        for i, (override, job) in enumerate(zip(overrides, jobs)):
            idx = self.job_idx + i
            job_info = JobInfo(idx=idx, job=job, overrides=override, trial_info=trial_info)
            self.running_job_info.append(job_info)

        self.job_idx += len(jobs)

    def _diff_overrides(self, run_info: TrialInfo):
        run_info_cfg_flat = flatten_dict(run_info.config.get_dictionary())

        diff_overrides = []
        for key, val in run_info_cfg_flat.items():
            if key in self.base_cfg_flat.keys():
                if val != self.base_cfg_flat[key]:
                    diff_overrides.append(f"{key}={val}")

            else:
                # list valued key?
                components = [i if not i.isnumeric() else int(i) for i in key.split(".")]
                key_intermediate = ".".join(components[:-1])
                index = components[-1]
                val1 = self.base_cfg_flat[key_intermediate][index]
                if val != val1:
                    diff_overrides.append(f"{key}={val1}")

        # diff_overrides = [
        #     tuple(f"{key}={val1}"
        #           for key, val1 in run_info_cfg_flat.items()
        #           if key in self.base_cfg_flat.keys() and val1 != self.base_cfg_flat[key])]

        return [tuple(diff_overrides)]

    def _workers_available(self) -> bool:
        """
        Check if workers are available

        Returns
        -------
        bool
            Yes/No: At least one worker is available.

        """
        if len(self.running_job_info) < self.n_jobs:
            return True
        return False

    def get_finished_runs(self) -> list[tuple[TrialInfo, TrialValue]]:
        """
        Return finisihed configurations/runs

        This method returns any finished configuration, and returns a list with
        the results of exercising the configurations. This class keeps populating results
        to self.results until a call to get_finished runs is done. In this case, the
        self.results list is emptied and all TrialValues produced by running run() are
        returned.

        Returns
        -------
        list[TrialInfo, TrialValue]: A list of TrialValues (and respective TrialInfo), that is,
            the results of executing a run_info and a submitted configuration.
        """

        # Proactively see if more configs have finished
        self._extract_completed_runs_from_futures()

        results_list: list[tuple[TrialInfo, TrialValue]] = []
        while self.results:
            # run_info, job = self.results.pop()
            job_info = self.results.pop()
            run_info, job = job_info.run_info, job_info.job
            ret = job.result()
            endtime = time.time()
            run_value = TrialValue(
                cost=ret.return_value,
                time=endtime - job._start_time,
                status=StatusType.SUCCESS if ret.status == JobStatus.COMPLETED else StatusType.CRASHED,
                starttime=job._start_time,
                endtime=endtime,
                additional_info=None,
            )
            results_list.append((run_info, run_value))
        return results_list

    def _extract_completed_runs_from_futures(self) -> None:
        """
        Manage active runs

        A run is over, when a future has done() equal true.
        This function collects the completed futures and move
        them from self.futures to self.results.
        We make sure futures never exceed the capacity of
        the scheduler.
        """
        jobs_done = [job for job in self.running_job_info if job.done()]
        for job_info in jobs_done:
            # for future in futures:
            self.results.append(job_info)
            self.running_job_info.remove(job_info)

    def wait(self) -> None:
        """
        Wait for runs to finish

        SMBO/intensifier might need to wait for runs to finish before making a decision.
        This class waits until 1 run completes.
        """
        if len(self.running_job_info):

            if self.progress_handler is not None:
                progress_slurm_refresh_interval = self.launcher.params["progress_slurm_refresh_interval"]
                job_idx, jobs, job_overrides = zip(*[(j.idx, j.job, j.overrides) for j in self.running_job_info])
                self.progress_handler.loop(
                    job_idx,
                    jobs,
                    job_overrides,
                    auto_stop=False,
                    progress_slurm_refresh_interval=progress_slurm_refresh_interval,
                    return_first_finished=False
                    # TODO this is not working currently because jobs can be finished in between
                )

            else:
                while True:
                    if any([[f.done() for f in self.running_job_info]]):
                        return

                    time.sleep(1)

    def is_running(self) -> bool:
        """
        Check if trails are running

        Whether or not there are configs still running. Generally if the runner is serial,
        launching a run instantly returns it's result. On parallel runners, there might
        be pending configurations to complete.
        """
        # If there are futures available, it translates
        # to runs still not finished/processed
        return len(self.running_job_info) > 0

    def run(
        self,
        config: Configuration,
        instance: str,
        cutoff: float | None = None,
        seed: int = 12345,
        budget: float | None = None,
        instance_specific: str = "0",
    ) -> tuple[StatusType, float, float, dict]:
        """
        Run configuration on target algorithm

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

    def count_available_workers(self) -> int:
        """
        Get the number of workers / jobs.

        Returns
        -------
        int
            Number of workers/jobs
        """
        # TODO find out why n_jobs needs to be double the amount
        # TODO find out why we have too little number of trials
        return self.n_jobs - len(self.running_job_info)


    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        """This method returns any finished configuration, and returns a list with the
        results of exercising the configurations. This class keeps populating results
        to ``self._results_queue`` until a call to ``get_finished`` trials is done. In this case,
        the `self._results_queue` list is emptied and all trial values produced by running
        `run` are returned.

        Returns
        -------
        Iterator[tuple[TrialInfo, TrialValue]]:
            A list of TrialInfo/TrialValue tuples, all of which have been finished.
        """
        # Proactively see if more configs have finished
        self._extract_completed_runs_from_futures()

        while self.results:
            # trial_info, job = self.results.pop()
            job_info = self.results.pop()
            trial_info, job = job_info.trial_info, job_info.job
            ret = job.result()
            endtime = time.time()
            status = StatusType.SUCCESS if ret.status == JobStatus.COMPLETED else StatusType.CRASHED
            run_value = TrialValue(
                cost=ret.return_value,
                time=endtime - job._start_time,
                status=status,
                starttime=job._start_time,
                endtime=endtime,
                additional_info=None,
            )
            yield (trial_info, run_value)
