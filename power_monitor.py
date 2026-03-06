import os
import time
import threading

class JetsonPowerMonitor:
    """
    Monitor Jetson Nano hardware power consumption using INA3221 sensors across multiple rails.
    Rails:
    in_power0_input: VDD_IN (Total Board Power)
    in_power1_input: VDD_CPU
    in_power2_input: VDD_GPU
    """
    BASE_PATH = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device"
    RAILS = {
        "total": "in_power0_input",
        "cpu": "in_power1_input",
        "gpu": "in_power2_input"
    }
    
    def __init__(self, sample_interval=0.01):
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.samples = {"total": [], "cpu": [], "gpu": []}
        self._monitor_thread = None

    def _read_rail(self, rail_name):
        """Reads a specific power rail in milliwatts."""
        path = os.path.join(self.BASE_PATH, self.RAILS[rail_name])
        if not os.path.exists(path):
            # Simulation fallback
            sim_values = {"total": 3500.0, "cpu": 2500.0, "gpu": 100.0}
            return sim_values[rail_name]
        
        try:
            with open(path, 'r') as f:
                return float(f.read().strip())
        except Exception:
            return 0.0

    def read_all_power(self):
        """Returns instantaneous power for all rails in mW."""
        return {
            "total_power_mW": self._read_rail("total"),
            "cpu_power_mW": self._read_rail("cpu"),
            "gpu_power_mW": self._read_rail("gpu")
        }

    def _sampling_loop(self):
        while self.is_monitoring:
            powers = self.read_all_power()
            self.samples["total"].append(powers["total_power_mW"])
            self.samples["cpu"].append(powers["cpu_power_mW"])
            self.samples["gpu"].append(powers["gpu_power_mW"])
            time.sleep(self.sample_interval)

    def start_monitoring(self):
        """Starts background power sampling every 10ms."""
        self.is_monitoring = True
        for key in self.samples:
            self.samples[key] = []
        self._monitor_thread = threading.Thread(target=self._sampling_loop)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stops monitoring and returns the average power (mW) for all rails."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        
        averages = {}
        for key, vals in self.samples.items():
            averages[key] = sum(vals) / len(vals) if vals else 0.0
        return averages

if __name__ == "__main__":
    monitor = JetsonPowerMonitor()
    print("Testing multi-rail power monitor...")
    monitor.start_monitoring()
    time.sleep(1)
    avgs = monitor.stop_monitoring()
    print(f"Average Power (mW): {avgs}")
