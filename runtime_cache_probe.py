import os
import glob
import subprocess
import json

class RuntimeCacheProbe:
    """
    Detects cache hierarchy information at runtime or install time.
    Reads from /sys/devices/system/cpu/cpu0/cache and falls back to lscpu.
    """
    def __init__(self):
        self.cache_info = {
            "l1d_size_bytes": None,
            "l2_size_bytes": None,
            "line_size_bytes": 64,  # default
            "l1d_associativity": None,
            "l2_associativity": None,
            "num_cores": os.cpu_count() or 1
        }
        self._probe()

    def _probe_sysfs(self):
        base_path = "/sys/devices/system/cpu/cpu0/cache"
        if not os.path.exists(base_path):
            return False
            
        success = False
        try:
            for index_dir in glob.glob(os.path.join(base_path, "index*")):
                with open(os.path.join(index_dir, "level"), "r") as f:
                    level = int(f.read().strip())
                with open(os.path.join(index_dir, "type"), "r") as f:
                    ctype = f.read().strip()
                
                size_str = "0"
                with open(os.path.join(index_dir, "size"), "r") as f:
                    size_str = f.read().strip()
                
                size_bytes = int(size_str[:-1]) * 1024 if size_str.endswith('K') else int(size_str)
                
                if level == 1 and ctype == "Data":
                    self.cache_info["l1d_size_bytes"] = size_bytes
                    with open(os.path.join(index_dir, "coherency_line_size"), "r") as f:
                        self.cache_info["line_size_bytes"] = int(f.read().strip())
                    with open(os.path.join(index_dir, "ways_of_associativity"), "r") as f:
                        self.cache_info["l1d_associativity"] = int(f.read().strip())
                    success = True
                elif level == 2:
                    self.cache_info["l2_size_bytes"] = size_bytes
                    if os.path.exists(os.path.join(index_dir, "ways_of_associativity")):
                        with open(os.path.join(index_dir, "ways_of_associativity"), "r") as f:
                            self.cache_info["l2_associativity"] = int(f.read().strip())
                    success = True
        except Exception:
            return False
        return success

    def _probe_lscpu(self):
        try:
            out = subprocess.check_output(["lscpu", "-J"], universal_newlines=True)
            data = json.loads(out)
            for item in data.get("lscpu", []):
                field = item.get("field", "").lower()
                val = item.get("data", "")
                
                if "l1d cache" in field or "l1d" in field:
                    self.cache_info["l1d_size_bytes"] = self._parse_lscpu_size(val)
                elif "l2 cache" in field or "l2" in field:
                    self.cache_info["l2_size_bytes"] = self._parse_lscpu_size(val)
        except Exception:
            pass

    def _parse_lscpu_size(self, val_str):
        val_str = val_str.strip().upper()
        if val_str.endswith("K") or val_str.endswith("KIB"):
            return int(float(val_str.replace("KIB", "").replace("K", "")) * 1024)
        if val_str.endswith("M") or val_str.endswith("MIB"):
            return int(float(val_str.replace("MIB", "").replace("M", "")) * 1024 * 1024)
        try:
            return int(val_str)
        except:
            return None

    def _probe(self):
        import platform
        import datetime
        
        self.cache_info["platform"] = platform.system()
        self.cache_info["cpu_model"] = platform.processor()
        self.cache_info["date_time"] = datetime.datetime.now().isoformat()
        
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
            self.cache_info["git_commit"] = commit
        except Exception:
            self.cache_info["git_commit"] = "unknown"
            
        if not self._probe_sysfs():
            self._probe_lscpu()
            
        # Fallbacks if still None (especially on Mac where lscpu/sysfs fails)
        if self.cache_info["l1d_size_bytes"] is None:
            # On mac, try sysctl
            if platform.system() == "Darwin":
                try:
                    l1d = subprocess.check_output(["sysctl", "-n", "hw.l1dcachesize"], universal_newlines=True).strip()
                    self.cache_info["l1d_size_bytes"] = int(l1d)
                    l2 = subprocess.check_output(["sysctl", "-n", "hw.l2cachesize"], universal_newlines=True).strip()
                    self.cache_info["l2_size_bytes"] = int(l2)
                    line = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], universal_newlines=True).strip()
                    self.cache_info["line_size_bytes"] = int(line)
                except Exception:
                    pass
            if self.cache_info["l1d_size_bytes"] is None:
                self.cache_info["l1d_size_bytes"] = 32768  # fallback
                self.cache_info["l2_size_bytes"] = 2097152 # fallback
            
    def get_info(self):
        return self.cache_info

    def save_descriptor(self, path="artifacts/platform_descriptor.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.cache_info, f, indent=2)

if __name__ == "__main__":
    probe = RuntimeCacheProbe()
    probe.save_descriptor("platform_descriptor.json")
    print(json.dumps(probe.get_info(), indent=2))
