sudo apt update
sudo apt install python3-pip python3-dev
pip3 install numpy pandas scipy matplotlib
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install onnx onnxruntime


python3 benchmarks/run_all_benchmarks.py --mode micro --runs 30 --warmup 10

python3 benchmarks/run_all_benchmarks.py --mode end-to-end --model all --runs 50 --warmup 10
