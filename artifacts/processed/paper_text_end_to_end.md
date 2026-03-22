## End-to-End Workload Evaluation

The end-to-end benchmarking results reveal a clear performance hierarchy across the evaluated CNN architectures natively running on macOS (Apple Silicon). As expected, the achieved throughput strictly inversely correlates with the overall architectural complexity of each model.

AlexNet proved to be the fastest in our evaluation, achieving a measured mean latency of 3.91 ms and supporting a high inference throughput of 264.18 Frames Per Second (FPS). In contrast, the deeper residual networks scaled proportionately with their depth: ResNet-18 processed inferences at an average of 8.31 ms (123.52 FPS), while ResNet-34 exhibited nearly double the latency at 15.77 ms (66.11 FPS).

VGG16, reflecting its considerably larger parameter count and dense computational requirements, was the slowest model evaluated. It averaged 53.59 ms per inference, strictly limiting its throughput to 19.20 FPS. These workload differences conservatively validate that while modern edge architectures can easily support real-time execution for shallower or optimized residual networks, traditionally heavy architectures like VGG16 still present heavy bottlenecks in unoptimized runtime deployments.
