"""
Measurement-Driven Energy Model for CNN Inference Benchmarking
-------------------------------------------------------
Energy is derived from measured hardware power and runtime latency.
"""

class EnergyModel:
    """
    Measurement-Driven Energy Model.
    Calculates energy and efficiency based on direct hardware readings.
    """
    
    def calculate_energy(self, average_power_mw, latency_ms):
        """
        Calculates energy consumption in millijoules (mJ).
        Formula: Energy (mJ) = Average Power (W) * Latency (s) * 1000
        Which simplifies to: Energy (mJ) = Power (mW) * Latency (ms) / 1000
        """
        return (average_power_mw * latency_ms) / 1000.0
        
    def calculate_efficiency(self, macs, energy_mj):
        """
        Calculates energy efficiency in MACs per Joule (MACs/J).
        """
        if energy_mj == 0:
            return 0
        energy_j = energy_mj / 1000.0
        return macs / energy_j

def compute_dynamic_macs(model, input_size=(1, 3, 224, 224)):
    """
    Uses thop to compute MAC operations directly from the model graph.
    """
    try:
        from thop import profile
        import torch
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return macs
    except ImportError:
        print("Warning: thop not installed, MAC counting may be inaccurate.")
        return 0
