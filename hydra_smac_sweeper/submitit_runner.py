import time
import typing

from hydra.core.utils import JobReturn
from omegaconf import OmegaConf

from smac.configspace import Configuration
from smac.runhistory.runhistory import RunInfo, RunValue
from smac.tae import StatusType
from smac.tae.base import BaseRunner
from submitit import Job
import pandas as pd

from hydra_smac_sweeper.submitit_smac_launcher import SubmititSmacLauncher

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def flatten_dict(d):
    return list(pd.json_normalize(d).T.to_dict().values())[0]


class SubmititRunner(BaseRunner):

    def __init__(
        self,
        single_worker: BaseRunner,
        launcher: SubmititSmacLauncher,
        n_jobs: int,
        output_directory: typing.Optional[str] = None,
    ):

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
        self.futures: typing.List[Job[JobReturn]] = []
        self.run_infos: typing.List[RunInfo] = []
        self.results: typing.List[typing.Tuple[RunInfo, Job]] = []
        self.base_cfg_flat = flatten_dict(OmegaConf.to_container(
            launcher.config, resolve=True, enum_to_str=True))

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

        overrides = self._diff_overrides(run_info)
        overrides.append(
            tuple(f"smac.{name}={val}" for name, val in run_info[1:]))
        overrides.append((f"smac.budget_variable={run_info.budget}"))
        jobs = self.launcher.launch(overrides, self.job_idx)
        self.run_infos.extend([run_info] * len(jobs))
        self.futures.extend(jobs)
        self.job_idx += len(jobs)

    def _diff_overrides(self, run_info: RunInfo):
        run_info_cfg_flat = flatten_dict(run_info.config)
        diff_overrides = []
        for key, val1 in run_info_cfg_flat.items():
            val2 = self.base_cfg_flat[key]
            if val1 != val2:
                diff_overrides.append((f"{key}={val1}",))
        return diff_overrides
                
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
            run_info, job = self.results.pop()
            ret = job.result()
            endtime = time.time()
            run_value = RunValue(cost=ret.return_value, time=endtime - job._start_time, status=ret.status,
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
        done_futures = [f for f in self.futures if f.done()]
        for i, future in enumerate(done_futures):
            run_info = self.run_infos[i]
            self.results.append((run_info, future))
            self.run_infos.remove(run_info)
            self.futures.remove(future)

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.
        This class waits until 1 run completes
        """
        if self.futures:
            while True:
                for job in self.futures:
                    if job.done():
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
        return len(self.futures) > 0

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
