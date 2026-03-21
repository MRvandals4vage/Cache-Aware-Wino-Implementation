import os
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

class MulticoreScheduler:
    """
    Implements static and guided scheduling for Winograd tiles.
    Uses Python threading since tasks release GIL in C-extensions natively.
    """
    def __init__(self, mode="single", num_threads=1, scheduling_type="static"):
        self.mode = mode # 'single' or 'multi'
        self.num_threads = num_threads if mode == "multi" else 1
        self.scheduling_type = scheduling_type # 'static' or 'guided'
        
        self.executor = None
        if self.mode == "multi":
            self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        
        self._set_thread_affinity()

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=False)

    def _set_thread_affinity(self):
        """Bind process to specific cores if available."""
        if self.mode == "multi" and hasattr(os, 'sched_setaffinity'):
            cores = psutil.cpu_count()
            if cores and self.num_threads <= cores:
                os.sched_setaffinity(0, list(range(self.num_threads)))

    def _static_schedule(self, tasks, worker_func):
        """Partition tasks equally among threads."""
        chunk_size = max(1, len(tasks) // self.num_threads)
        chunks = [tasks[x:x+chunk_size] for x in range(0, len(tasks), chunk_size)]
        
        results = []
        futures = [self.executor.submit(self._worker_batch, chunk, worker_func) for chunk in chunks]
        for future in as_completed(futures):
            results.extend(future.result())
        return results

    def _guided_schedule(self, tasks, worker_func):
        """Dynamic scheduling: tasks are picked from a queue."""
        results = []
        future_to_task = {self.executor.submit(worker_func, task): task for task in tasks}
        for future in as_completed(future_to_task):
            results.append(future.result())
        return results

    def _worker_batch(self, chunk, func):
        res = []
        for c in chunk:
            res.append(func(c))
        return res

    def execute_tasks(self, tasks, worker_func):
        if self.mode == "single":
            return [worker_func(t) for t in tasks]
            
        if self.scheduling_type == "static":
            return self._static_schedule(tasks, worker_func)
        else:
            return self._guided_schedule(tasks, worker_func)

if __name__ == "__main__":
    def dummy_worker(task):
        import time
        # time.sleep(0.001)
        return task * 2

    # Provide tasks as simple numbers
    tasks = list(range(100))
    scheduler = MulticoreScheduler(mode="multi", num_threads=4, scheduling_type="guided")
    print(scheduler.execute_tasks(tasks, dummy_worker)[:10])
