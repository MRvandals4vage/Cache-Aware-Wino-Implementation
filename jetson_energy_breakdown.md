# CNN Inference Energy Breakdown (Analytical Mode)

| Model          | Strategy             | Time (ms) | MACs         | DRAM Access | Compute Energy | Memory Energy | Total Energy |
| :------------- | :------------------- | :-------: | :----------: | :---------: | :------------: | :-----------: | :----------: |
| resnet18       | PyTorch Baseline     |       8.3 |        2.27B |       16.7M |           7.04 |          3.67 |        10.71 |
| resnet18       | ONNX Runtime         |       7.6 |        2.27B |       16.7M |           7.04 |          3.67 |        10.71 |
| resnet18       | TVM Model            |       7.0 |        2.27B |       15.6M |           7.04 |          3.43 |        10.47 |
| mobilenetv2    | PyTorch Baseline     |       3.4 |       340.0M |       18.7M |           1.05 |          4.11 |         5.16 |
| mobilenetv2    | ONNX Runtime         |       3.6 |       340.0M |       18.7M |           1.05 |          4.11 |         5.16 |
| mobilenetv2    | TVM Model            |       2.9 |       340.0M |       15.3M |           1.05 |          3.37 |         4.42 |
