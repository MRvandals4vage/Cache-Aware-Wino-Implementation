import os
import subprocess
import sys

def run_experiments():
    """Runs the main benchmarking experiment and produces all outputs."""
    print("Starting Edge Convolution Benchmarking Experiments...")
    
    # Path to main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(current_dir, "main.py")
    
    # Execute the benchmarking command
    try:
        # Run main.py and stream output to console
        subprocess.check_call([sys.executable, main_path])
        print("\nExperiments completed. All results saved in the project directory.")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred during experiment execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_experiments()
