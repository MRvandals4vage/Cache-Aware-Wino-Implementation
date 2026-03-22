# pyre-ignore-all-errors
import os
import subprocess
import time
from typing import Any, Dict, Optional

try:
    import psutil # pyre-ignore[21]
except ImportError:
    psutil = None # pyre-ignore

class HardwareTelemetry:
    """
    Unified telemetry collector. Detects what is available (perf, vcgencmd, psutil).
    """
    def __init__(self, use_perf: bool = True) -> None:
        self.use_perf = use_perf
        self.perf_process: Optional[subprocess.Popen[str]] = None
        self.has_vcgencmd: bool = False
        
        # Check if vcgencmd is available
        try:
            subprocess.check_output(["which", "vcgencmd"], stderr=subprocess.STDOUT)
            self.has_vcgencmd = True
        except Exception:
            self.has_vcgencmd = False

        self.start_metrics: Dict[str, Any] = {}
        self.end_metrics: Dict[str, Any] = {}

    def start(self) -> None:
        # 1. Start Perf if available
        if self.use_perf:
            cmd = ["perf", "stat", "-e", "L1-dcache-loads,L1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.demand_data_rd_miss", "-p", str(os.getpid())]
            try:
                self.perf_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
            except Exception:
                self.perf_process = None

        # 2. Record start psutil metrics
        if psutil:
            self.start_metrics["cpu_percent"] = psutil.cpu_percent(interval=None) # type: ignore
            self.start_metrics["mem_used"] = psutil.virtual_memory().used # type: ignore
            
    def stop(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        
        # 1. Stop Perf
        if self.perf_process is not None:
            self.perf_process.terminate() # pyre-ignore
            stdout, stderr = self.perf_process.communicate() # pyre-ignore
            results["perf_stat"] = stderr
        else:
            results["perf_stat"] = None

        # 2. End psutil metrics
        if psutil:
            results["cpu_percent"] = psutil.cpu_percent(interval=None) # type: ignore
            results["mem_used_end"] = psutil.virtual_memory().used # type: ignore
            
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
