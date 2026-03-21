import os
import subprocess
import time

try:
    import psutil
except ImportError:
    psutil = None

class HardwareTelemetry:
    """
    Unified telemetry collector. Detects what is available (perf, vcgencmd, psutil).
    """
    def __init__(self, use_perf=True):
        self.use_perf = use_perf
        self.perf_process = None
        self.has_vcgencmd = False
        
        # Check if vcgencmd is available
        try:
            subprocess.check_output(["which", "vcgencmd"], stderr=subprocess.STDOUT)
            self.has_vcgencmd = True
        except Exception:
            self.has_vcgencmd = False

        self.start_metrics = {}
        self.end_metrics = {}

    def start(self):
        # 1. Start Perf if available
        if self.use_perf:
            cmd = ["perf", "stat", "-e", "L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss", "-p", str(os.getpid())]
            try:
                self.perf_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            except Exception:
                self.perf_process = None

        # 2. Record start psutil metrics
        if psutil:
            self.start_metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            self.start_metrics["mem_used"] = psutil.virtual_memory().used
            
    def stop(self):
        results = {}
        
        # 1. Stop Perf
        if self.perf_process:
            self.perf_process.terminate()
            stdout, stderr = self.perf_process.communicate()
            results["perf_stat"] = stderr
        else:
            results["perf_stat"] = None

        # 2. End psutil metrics
        if psutil:
            results["cpu_percent"] = psutil.cpu_percent(interval=None)
            results["mem_used_end"] = psutil.virtual_memory().used
            
        # 3. Collect vcgencmd metrics (one shot snapshot at the end of execution block)
        if self.has_vcgencmd:
            try:
                temp_out = subprocess.check_output(["vcgencmd", "measure_temp"], universal_newlines=True)
                results["pi_temp"] = temp_out.strip()
                
                clock_out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], universal_newlines=True)
                results["pi_clock_arm"] = clock_out.strip()
                
                throttled_out = subprocess.check_output(["vcgencmd", "get_throttled"], universal_newlines=True)
                results["pi_throttled"] = throttled_out.strip()
            except Exception:
                results["pi_temp"] = "Error"
        
        return results
