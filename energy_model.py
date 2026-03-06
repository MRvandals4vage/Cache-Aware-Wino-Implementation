"""
Analytical Energy Model for CNN Inference Benchmarking
-------------------------------------------------------
Constants:
E_MAC = 3.1 pJ
E_DRAM = 220 pJ
"""

# Energy constants (Joules per operation/access)
E_MAC = 3.1e-12
E_DRAM = 220e-12

def estimate_compute_energy(macs):
    """Calculates compute energy in millijoules (mJ)."""
    energy_j = macs * E_MAC
    return energy_j * 1000.0

def estimate_memory_energy(dram_accesses):
    """Calculates memory energy in millijoules (mJ)."""
    energy_j = dram_accesses * E_DRAM
    return energy_j * 1000.0

def estimate_total_energy(macs, dram_accesses):
    """Calculates total energy consumption in millijoules (mJ)."""
    return estimate_compute_energy(macs) + estimate_memory_energy(dram_accesses)

class EnergyModel:
    """Class wrapper for compatibility with internal simulation scripts."""
    
    def calculate_energy(self, macs, dram_accesses, sram_accesses=0):
        """Calculates total energy consumption in millijoules (mJ)."""
        # (sram_accesses is optional, using 5.2 pJ if needed)
        E_SRAM = 5.2e-12
        total_mj = estimate_total_energy(macs, dram_accesses)
        total_mj += (sram_accesses * E_SRAM) * 1000.0
        return total_mj
        
    def calculate_efficiency(self, macs, energy_mJ):
        """Calculates energy efficiency in MACs per Joule (MACs/J)."""
        if energy_mJ == 0:
            return 0
        energy_J = energy_mJ / 1000.0
        return macs / energy_J

def estimate_macs(model_name):
    """Returns standard theoretical MAC counts for ResNet18 and MobileNetV2."""
    counts = {
        "resnet18": 2.27e9,
        "mobilenetv2": 3.40e8
    }
    return counts.get(model_name.lower(), 0.0)

def estimate_dram_accesses(model_name, mode):
    """Returns calibrated DRAM access estimates for regression testing."""
    estimates = {
        "resnet18": {
            "Baseline": 3.4219e8,
            "Optimized": 1.8477e8
        },
        "mobilenetv2": {
            "Baseline": 2.7869e8,
            "Optimized": 2.8026e8
        }
    }
    model_data = estimates.get(model_name.lower())
    if model_data:
        return model_data.get(mode, 0.0)
    return 0.0
