class JobInfo:
    def __init__(self, idx, job, overrides, run_info):
        self.idx = idx
        self.job = job
        self.overrides = overrides
        self.run_info = run_info

    def done(self):
        return self.job.done()
