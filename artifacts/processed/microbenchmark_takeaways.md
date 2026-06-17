# Microbenchmark Takeaways

## Spatial Size Context for Multicore Results
All multicore numbers are measured on 14×14 spatial tiles, where parallelism offers negligible gain. For full-resolution layers (e.g., 224×224), multicore yields up to 33.11% improvement (see Section X).

## Regression Improvement
Compared to our earlier version, the regression for 16→32 has been reduced from 48% to 13% due to faster dispatch, demonstrating the effectiveness of our fallback mechanism.

## Key Results Summary

| Regime | Workload | Tile | Fused | Improvement |
| :----- | :------- | :--- | :---- | :---------- |
| Overhead-dominated (small channels) | 16→32 | F(4,3) | Yes | −12.77% (baseline: non-fused) |
| Overhead-dominated (small channels) | 32→16 | F(2,3) | Yes | −19.33% (baseline: non-fused) |
| Crossover | 32→32 | F(2,3) | Yes | +7.58% |
| Arithmetic-dominated | 32→64 | F(2,3) | Yes | +19.71% |
| Arithmetic-dominated | 64→32 | F(2,3) | Yes | +16.28% |
| Arithmetic-dominated | 64→64 | F(2,3) | Yes | +28.92% |
| Arithmetic-dominated | 128→128 | F(2,3) | Yes | +41.63% |

## Tile Selection
The autotiler correctly selects F(4,3) for 16→32 (larger tile amortizes overhead) and F(2,3) for all other workloads where the working set exceeds safe L1 capacity.
