# CNN Inference Energy Breakdown (Analytical Mode)

| Model          | Strategy             | Time (ms) | MACs         | DRAM Access | Compute Energy | Memory Energy | Total Energy |
| :------------- | :------------------- | :-------: | :----------: | :---------: | :------------: | :-----------: | :----------: |
| resnet18       | PyTorch Baseline     |       8.3 |        2.27B |       16.7M |           7.04 |          3.67 |        10.71 |
| resnet18       | ONNX Runtime         |       7.6 |        2.27B |       16.7M |           7.04 |          3.67 |        10.71 |
| resnet18       | TVM Model            |       7.0 |        2.27B |       15.6M |           7.04 |          3.43 |        10.47 |
