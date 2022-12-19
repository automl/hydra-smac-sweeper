# Add below as a WA for
# https://github.com/dask/distributed/issues/4168
import multiprocessing.popen_spawn_posix  # noqa
import time
import unittest.mock
from unittest.mock import patch

import dask  # noqa
from hydra_plugins.hydra_smac_sweeper.submitit_runner import SubmititRunner
from hydra_plugins.hydra_smac_sweeper.submitit_smac_launcher import SMACLocalLauncher
from omegaconf import DictConfig, OmegaConf
from smac.utils.configspace import ConfigurationSpace, UniformFloatHyperparameter
from smac.runhistory import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType
from smac.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def create_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(lower=-10, upper=10, default_value=-3, log=False, name="x"),
        ]
    )
    return cs


def target(x, seed, instance):
    return x**2, {"key": seed, "instance": instance}


def target_delayed(x, seed, instance):
    time.sleep(1)
    return x**2, {"key": seed, "instance": instance}


def target_wrapper(target_function) -> callable:
    def target(cfg: DictConfig):
        return target_function(x=cfg.x, seed=cfg.seed, instance=cfg.instance)

    return target


class TestSubmititRunner(unittest.TestCase):
    def setUp(self):
        self.cs = create_configspace()
        self.scenario = Scenario(**{"configspace": self.cs, "output_directory": ""})

    def get_runner(self, ta, n_jobs: int = 2) -> SubmititRunner:
        ta = target_wrapper(ta)
        launcher = SMACLocalLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"sweep": {"dir": "./tmp"}},
                "x": 3,
                "instance": 0,
                "seed": 55,
            }
        )
        launcher.config = config
        launcher.params["progress"] = "rich"
        launcher.params["progress_slurm_refresh_interval"] = 1
        launcher.params["submitit_folder"] = "./tmp"
        return SubmititRunner(target_function=ta, scenario=self.scenario, n_jobs=n_jobs, launcher=launcher, budget_variable="")

    #@patch("SMACLocalLauncher.", return_value=9)
    def test_run(self):
        """Makes sure that we are able to run a configuration and
        return the expected values/types"""
        runner = self.get_runner(ta=target)

        # We use the funcdict as a mechanism to test Parallel Runner
        self.assertIsInstance(runner, SubmititRunner)

        trial_info = TrialInfo(
            config=self.cs.sample_configuration(),
            instance="test",
            seed=0,
            budget=0.0,
        )

        # submit runs! then get the value
        runner.submit_trial(trial_info)
        run_values = runner.get_finished_runs()
        # Run will not have finished so fast
        self.assertEqual(len(run_values), 0)
        runner.wait()
        run_values = runner.get_finished_runs()
        self.assertEqual(len(run_values), 1)
        self.assertIsInstance(run_values, list)
        self.assertIsInstance(run_values[0][0], TrialInfo)
        self.assertIsInstance(run_values[0][1], TrialValue)
        self.assertEqual(run_values[0][1].cost, 4)
        self.assertEqual(run_values[0][1].status, StatusType.SUCCESS)

    def test_parallel_runs(self):
        """Make sure because there are 2 workers, the runs are launched
        closely in time together"""

        # We use the funcdict as a mechanism to test Runner
        runner = self.get_runner(ta=target_delayed)

        trial_info = TrialInfo(
            config=self.cs.sample_configuration(),
            instance="test",
            seed=0,
            budget=0.0,
        )
        runner.submit_trial(trial_info)
        trial_info = TrialInfo(
            config=self.cs.sample_configuration(),
            instance="test",
            seed=0,
            budget=0.0,
        )
        runner.submit_trial(trial_info)

        # At this stage, we submitted 2 jobs, that are running in remote
        # workers. We have to wait for each one of them to complete. The
        # runner provides a wait() method to do so, yet it can only wait for
        # a single job to be completed. It does internally via dask wait(<list of futures>)
        # so we take wait for the first job to complete, and take it out
        runner.wait()
        run_values = runner.get_finished_runs()

        # To be on the safe side, we don't check for:
        # self.assertEqual(len(run_values), 1)
        # In the ideal world, two runs were launched which take the same time
        # so waiting for one means, the second one is completed. But some
        # overhead might cause it to be delayed.

        # Above took the first run results and put it on run_values
        # But for this check we need the second run we submitted
        # Again, the runs might take slightly different time depending if we have
        # heterogeneous workers, so calling wait 2 times is a guarantee that 2
        # jobs are finished
        runner.wait()
        run_values.extend(runner.get_finished_runs())
        self.assertEqual(len(run_values), 2)

        # To make it is parallel, we just make sure that the start of the second
        # run is earlier than the end of the first run
        # Results are returned in left to right
        self.assertLessEqual(int(run_values[0][1].starttime), int(run_values[1][1].endtime))

    def test_num_workers(self):
        """Make sure we can properly return the number of workers"""

        # We use the funcdict as a mechanism to test Runner
        runner = self.get_runner(ta=target_delayed, n_jobs=2)
        self.assertEqual(runner.n_jobs, 2)

        # Reduce the number of workers
        # have to give time for the worker to be killed
        # time.sleep(2)
        # self.assertEqual(runner.num_workers(), 1)

    # def test_file_output(self):
    #     tmp_dir = tempfile.mkdtemp()
    #     single_worker_mock = unittest.mock.Mock()
    #     parallel_runner = SubmititRunner(  # noqa F841
    #         single_worker=single_worker_mock, n_workers=1, output_directory=tmp_dir
    #     )
    #     self.assertTrue(os.path.exists(os.path.join(tmp_dir, ".dask_scheduler_file")))

    # def test_do_not_close_external_client(self):
    #     tmp_dir = tempfile.mkdtemp()
    #
    #     single_worker_mock = unittest.mock.Mock()
    #     client = Client()
    #     parallel_runner = SubmititRunner(
    #         single_worker=single_worker_mock,
    #         dask_client=client,
    #         n_workers=1,
    #         output_directory=tmp_dir,
    #     )  # noqa F841
    #     del parallel_runner
    #     self.assertFalse(os.path.exists(os.path.join(tmp_dir, ".dask_scheduler_file")))
    #     self.assertEqual(client.status, "running")
    #     parallel_runner = SubmititRunner(
    #         single_worker=single_worker_mock,
    #         dask_client=client,
    #         n_workers=1,
    #         output_directory=tmp_dir,
    #     )  # noqa F841
    #     del parallel_runner
    #     self.assertEqual(client.status, "running")
    #     client.shutdown()

    def test_additional_info_crash_msg(self):
        """
        We want to make sure we catch errors as additional info,
        and in particular when doing multiprocessing runs, we
        want to make sure we capture dask exceptions
        """

        def target_nonpickable(x, seed, instance):
            return x**2, {"key": seed, "instance": instance}

        runner = self.get_runner(ta=target_nonpickable)

        trial_info = TrialInfo(
            config=self.cs.sample_configuration(),
            instance="test",
            seed=0,
            budget=0.0,
        )
        runner.submit_trial(trial_info)
        runner.wait()
        trial_info, result = runner.get_finished_runs()[0]

        # Make sure the traceback message is included
        self.assertIn("traceback", result.additional_info)
        self.assertIn(
            # We expect the problem to occur in the run wrapper
            # So traceback should show this!
            "target_nonpickable",
            result.additional_info["traceback"],
        )

        # Make sure the error message is included
        self.assertIn("error", result.additional_info)
        self.assertIn("Can't pickle local object", result.additional_info["error"])


if __name__ == "__main__":
    unittest.main()
