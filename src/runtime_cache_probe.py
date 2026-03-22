import os
import glob
import subprocess
import json
import platform
import datetime
import psutil
import sys

def get_git_commit_hash():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
        return commit
    except Exception:
        return "unknown"

def read_sysfs_cache():
    base_path = "/sys/devices/system/cpu/cpu0/cache"
    if not os.path.exists(base_path):
        return None
        
    cache_data = {}
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
                cache_data["l1d_size_bytes"] = size_bytes
                with open(os.path.join(index_dir, "coherency_line_size"), "r") as f:
                    cache_data["line_size_bytes"] = int(f.read().strip())
                with open(os.path.join(index_dir, "ways_of_associativity"), "r") as f:
                    cache_data["l1d_associativity"] = int(f.read().strip())
            elif level == 2:
                cache_data["l2_size_bytes"] = size_bytes
                if os.path.exists(os.path.join(index_dir, "ways_of_associativity")):
                    with open(os.path.join(index_dir, "ways_of_associativity"), "r") as f:
                        cache_data["l2_associativity"] = int(f.read().strip())
    except Exception:
        return None
        
    if "l1d_size_bytes" not in cache_data:
        return None
    return cache_data

def read_lscpu_cache():
    cache_data = {}
    try:
        out = subprocess.check_output(["lscpu", "-J"], universal_newlines=True)
        data = json.loads(out)
        for item in data.get("lscpu", []):
            field = item.get("field", "").lower()
            val = item.get("data", "")
            
            if "l1d cache" in field or "l1d" in field:
                cache_data["l1d_size_bytes"] = _parse_lscpu_size(val)
            elif "l2 cache" in field or "l2" in field:
                cache_data["l2_size_bytes"] = _parse_lscpu_size(val)
    except Exception:
        return None
    
    if "l1d_size_bytes" not in cache_data:
        return None
    return cache_data

def _parse_lscpu_size(val_str):
    val_str = val_str.strip().upper()
    if val_str.endswith("K") or val_str.endswith("KIB"):
        return int(float(val_str.replace("KIB", "").replace("K", "")) * 1024)
    if val_str.endswith("M") or val_str.endswith("MIB"):
        return int(float(val_str.replace("MIB", "").replace("M", "")) * 1024 * 1024)
    try:
        return int(val_str)
    except:
        return None

def detect_platform():
    info = {
        "os": platform.system(),
        "architecture": platform.machine(),
        "cpu_model": platform.processor(),
        "python_version": sys.version,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "l1d_size_bytes": None,
        "l2_size_bytes": None,
        "line_size_bytes": 64,
        "l1d_associativity": None,
        "l2_associativity": None,
        "is_raspberry_pi": False,
        "pi_model": None,
        "has_neon": False
    }
    
    # OS specific detection
    if info["os"] == "Linux":
        # Raspberry Pi / ARM checking
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().strip('\x00').strip()
                    info["pi_model"] = model
                    if "Raspberry Pi" in model:
                        info["is_raspberry_pi"] = True
            except Exception:
                pass
                
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "neon" in cpuinfo.lower() or "asimd" in cpuinfo.lower():
                        info["has_neon"] = True
            except Exception:
                pass
                
        # Cache checking
        sysfs_cache = read_sysfs_cache()
        if sysfs_cache:
            info.update(sysfs_cache)
        else:
            lscpu_cache = read_lscpu_cache()
            if lscpu_cache:
                info.update(lscpu_cache)
                
    elif info["os"] == "Darwin":
        # macOS sysctl fallback
        try:
            l1d = subprocess.check_output(["sysctl", "-n", "hw.l1dcachesize"], universal_newlines=True).strip()
            info["l1d_size_bytes"] = int(l1d)
        except Exception:
            pass
        try:
            l2 = subprocess.check_output(["sysctl", "-n", "hw.l2cachesize"], universal_newlines=True).strip()
            info["l2_size_bytes"] = int(l2)
        except Exception:
            pass
        try:
            line = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], universal_newlines=True).strip()
            info["line_size_bytes"] = int(line)
        except Exception:
            pass
            
    # Universal fallback if caches still unknown
    if info["l1d_size_bytes"] is None:
        info["l1d_size_bytes"] = 32768
    if info["l2_size_bytes"] is None:
        info["l2_size_bytes"] = 2097152
        
    return info

def build_platform_descriptor():
    return detect_platform()

def save_platform_descriptor(descriptor, path="artifacts/platform_descriptor.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(descriptor, f, indent=2)
    return path

if __name__ == "__main__":
    desc = build_platform_descriptor()
    print(json.dumps(desc, indent=2))
    save_platform_descriptor(desc)
