# Convolution Strategy Benchmarking Results

| Model          | Mode                | Time (ms) | MACs         | DRAM Accesses | Energy (mJ) | MACs/J    |
| :------------- | :------------------ | :-------: | :----------: | :-----------: | :---------: | :-------: |
| resnet18       | Baseline            |    507.26 | 2,269,170,432 |   341,927,104 |       82.26 |  2.76e+10 |
| resnet18       | Naive Winograd      |    445.13 | 1,658,899,200 |   725,443,776 |      164.74 |  1.01e+10 |
| resnet18       | Cache-Aware         |    360.99 | 1,658,899,200 |   186,969,280 |       46.28 |  3.58e+10 |
| resnet18       | TVM Model           |    360.65 | 1,658,899,200 |   184,766,144 |       45.79 |  3.62e+10 |
