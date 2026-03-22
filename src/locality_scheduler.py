import os
import json
import psutil

class LocalityScheduler:
    def __init__(self, trace_file="artifacts/scheduler_trace.json"):
        self.trace_file = trace_file
        self.trace = []

    def generate_tile_tasks(self, h, w, tile_dim, c_in, c_out):
        """
        Generate atomic task descriptors for each output tile that needs computation.
        """
        m = tile_dim - 2 # output tile size assuming r=3
        h_tiles = (h + m - 1) // m
        w_tiles = (w + m - 1) // m
        
        tasks = []
        for yo in range(h_tiles):
            for xo in range(w_tiles):
                # For each channel block (compute in blocks of 16 for locality)
                c_block = 16
                for co in range(0, c_out, c_block):
                    task = {
                        "id": f"t_{yo}_{xo}_{co}",
                        "y_tile": yo,
                        "x_tile": xo,
                        "c_out_start": co,
                        "c_out_end": min(co + c_block, c_out),
                        "c_in_total": c_in,
                        "tile_dim": tile_dim
                    }
                    tasks.append(task)
        return tasks

    def group_tasks_by_channel_locality(self, tasks):
        """
        Order tasks to prioritize channel reuse. Keep c_out fixed while iterating over spatial tiles.
        Returns a sorted list of tasks.
        """
        # Sort by channel block first, then by spatial tiles (y, x)
        sorted_tasks = sorted(tasks, key=lambda t: (t["c_out_start"], t["y_tile"], t["x_tile"]))
        return sorted_tasks

    def schedule_single_core(self, tasks):
        """
        Execute tasks sequentially.
        """
        execution_plan = []
        for i, task in enumerate(tasks):
            execution_plan.append({"core": 0, "order": i, "task": task})
        return execution_plan

    def schedule_multi_core(self, tasks, num_cores=None):
        """
        Distribute tasks across cores.
        Assign different channel blocks to different cores to avoid false sharing.
        """
        if num_cores is None:
            num_cores = psutil.cpu_count(logical=False) or 1
            
        execution_plan = []
        
        # Group tasks by channel block
        from collections import defaultdict
        blocks = defaultdict(list)
        for t in tasks:
            blocks[t["c_out_start"]].append(t)
            
        # Distribute channel blocks to cores
        core_assignments = {i: [] for i in range(num_cores)}
        
        for i, (c_start, block_tasks) in enumerate(blocks.items()):
            core_id = i % num_cores
            core_assignments[core_id].extend(block_tasks)
            
        for core_id, core_tasks in core_assignments.items():
            for order, task in enumerate(core_tasks):
                execution_plan.append({"core": core_id, "order": order, "task": task})
                
        return execution_plan

    def run_schedule(self, execution_plan):
        """
        Simulate the schedule execution and log the trace.
        """
        self.trace.append({
            "event": "schedule_executed",
            "plan": execution_plan,
            "tasks_count": len(execution_plan)
        })
        self._save_trace()
        return len(execution_plan)

    def _save_trace(self):
        os.makedirs(os.path.dirname(self.trace_file), exist_ok=True)
        with open(self.trace_file, "w") as f:
            json.dump(self.trace, f, indent=2)

if __name__ == "__main__":
    scheduler = LocalityScheduler()
    tasks = scheduler.generate_tile_tasks(14, 14, 4, 64, 64)
    ordered_tasks = scheduler.group_tasks_by_channel_locality(tasks)
    plan = scheduler.schedule_multi_core(ordered_tasks, num_cores=4)
    scheduler.run_schedule(plan)
    print(f"Scheduled {len(plan)} tasks.")
