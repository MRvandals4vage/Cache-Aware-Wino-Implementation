#!/usr/bin/env python3
import os
import subprocess
import json
import psutil

class CounterCollector:
    def __init__(self, log_dir="artifacts/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.supported_counters = self._detect_supported_counters()
        
    def _detect_supported_counters(self):
        supported = {
            "perf": False,
            "vcgencmd": False,
            "powermetrics": False
        }
        
        # Check perf (Linux)
        try:
            subprocess.run(["perf", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            supported["perf"] = True
        except Exception:
            pass
            
        # Check vcgencmd (Raspberry Pi)
        try:
            subprocess.run(["vcgencmd", "measure_temp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            supported["vcgencmd"] = True
        except Exception:
            pass
            
        # Check powermetrics (macOS)
        try:
            if os.uname().sysname == "Darwin":
                # Check root or existence, powermetrics needs root usually, but we check if command exists
                subprocess.run(["which", "powermetrics"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                supported["powermetrics"] = True
        except Exception:
            pass
            
        return supported

    def collect(self):
        summary = {
            "supported": self.supported_counters,
            "logs": {}
        }
        
        # CPU percentages (Universal)
        summary["logs"]["cpu_percent"] = psutil.cpu_percent(interval=1.0)
        
        # 1. perf
        if self.supported_counters["perf"]:
            try:
                # We do a tiny sleep just to collect some perf stats
                out = subprocess.check_output(
                    ["perf", "stat", "-e", "instructions,cycles,cache-misses", "sleep", "0.1"],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                with open(os.path.join(self.log_dir, "perf_raw.log"), "w") as f:
                    f.write(out)
                summary["logs"]["perf"] = "Logged to perf_raw.log"
            except Exception as e:
                summary["logs"]["perf_error"] = str(e)
        else:
            summary["logs"]["perf"] = "Unsupported or not found"
            
        # 2. vcgencmd
        if self.supported_counters["vcgencmd"]:
            try:
                temp = subprocess.check_output(["vcgencmd", "measure_temp"], universal_newlines=True).strip()
                clock = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], universal_newlines=True).strip()
                summary["logs"]["vcgencmd"] = {
                    "temperature": temp,
                    "clock": clock
                }
            except Exception as e:
                summary["logs"]["vcgencmd_error"] = str(e)
        else:
            summary["logs"]["vcgencmd"] = "Unsupported or not found"
            
        # 3. powermetrics
        if self.supported_counters["powermetrics"]:
            summary["logs"]["powermetrics"] = "Skipped (likely requires sudo)"
            
        # Write summary
        summary_path = os.path.join(self.log_dir, "telemetry_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(json.dumps(summary, indent=2))
        return summary_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Hardware Counters")
    parser.add_argument("--log-dir", default="artifacts/logs")
    args = parser.parse_args()
    
    collector = CounterCollector(args.log_dir)
    collector.collect()
