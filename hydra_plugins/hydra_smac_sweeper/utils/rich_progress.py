from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from hydra.core.utils import filter_overrides, JobStatus
import time
import logging
from functools import lru_cache


log = logging.getLogger(__name__)

@lru_cache(None)
def warn_once(msg: str):
    log.warning(msg)

class RichProgress:

    def __init__(self) -> None:
        
        self.progress = Progress(
            "{task.fields[job_id]}",
            "{task.fields[status]}",
            SpinnerColumn(),
            TimeElapsedColumn(),
            # "{task.fields[status_text]}",
            "{task.description}",            
        )
        self.idx_to_progress_task_id = {}
        self.started = False
        self.all_running_jobs = {}

    def get_progress_task(self, idx):
        
        if idx in self.idx_to_progress_task_id:
            progress_task_id = self.idx_to_progress_task_id[idx]
        else:
            progress_task_id = self.progress.add_task("", total=1, start=False, job_id=f"#{idx}", status=' ', status_text='')

            self.idx_to_progress_task_id[idx] = progress_task_id

        return self.progress._tasks[progress_task_id]

    def start(self):
        self.started = True
        self.progress.start()

    def stop(self):
        self.started = False
        self.progress.stop()

    def stop_task(self, idx):
        progress_task = self.get_progress_task(idx)
        self.progress.update(progress_task.id, completed=1)
        if idx in self.all_running_jobs:
            del self.all_running_jobs[idx]

    def refresh(self, job_idx, jobs, job_overrides):

        for (idx, job, overrides) in zip(job_idx, jobs, job_overrides):
            if idx not in self.all_running_jobs:
                self.all_running_jobs[idx] = (idx, job, overrides)

        for (idx, job, overrides) in list(self.all_running_jobs.values()):
            progress_task = self.get_progress_task(idx)
            s = job.state.upper()

            status_icon = '🕛'
            done = job.done()

            if (s == 'RUNNING' or done) and not progress_task.started:
                self.progress.start_task(progress_task.id) # start_time is not set to the exact start time because this information not included in slurm job info

            if done:

                if not progress_task.finished:
                    # self.progress.update(progress_task.id, completed=1)
                    self.stop_task(idx)

                try:
                    r = job.result()
                    hydra_status = r.status

                    if hydra_status == JobStatus.COMPLETED:
                        s = 'SUCCESS' # 
                        status_icon = '🏁'

                    elif hydra_status == JobStatus.FAILED:
                        s = 'FAILED' # 💥
                        status_icon = '💥'
                except:
                    s = 'FAILED' # 💥
                    status_icon = '💥'
            
            if s == 'RUNNING':
                status_icon = '🏃'

            lst = " ".join(filter_overrides(overrides))
            progress_task.description = lst
            progress_task.fields['status'] = status_icon
            progress_task.fields['status_text'] = s

    def loop(self, job_idx, jobs, job_overrides, progress_slurm_refresh_interval = 15, auto_start=True, auto_stop=True, return_first_finished=False):

        if progress_slurm_refresh_interval < 15:
            warn_once('WARNING: progress_slurm_refresh_interval should not be smaller than 15 seconds otherwise slurm will be queried too often') 

        if auto_start:
            self.start()

        last_status_check = time.time()
        
        while True:
            last_check_delta = time.time() - last_status_check
            if last_check_delta >= progress_slurm_refresh_interval:
                jobs[0].get_info(mode="force")

            self.refresh(job_idx=job_idx, jobs=jobs, job_overrides=job_overrides)

            num_done = sum([1 if job.done()  else 0 for job in jobs])

            if return_first_finished and num_done > 0:
                break
            if num_done == len(jobs):
                break

            time.sleep(1)

        if auto_stop:
            self.stop()