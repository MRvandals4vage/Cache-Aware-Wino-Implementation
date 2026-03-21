import pandas as pd

df = pd.read_csv("artifacts/processed/microbenchmark_results.csv")
print("### 3.1 Measured Core Latency\n")
print("Platform descriptor: `artifacts/platform_descriptor.json`\n")
print("| Platform | C_in | C_out | Fused | Multi | Mean Latency (ms) | 95% CI | Test Used | p-value | Effect |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for _, r in df.iterrows():
    print(f"| {r['Platform']} | {r['C_in']} | {r['C_out']} | {r['Fused']} | {r['MultiCore']} | {r['Mean_Latency_ms']:.2f} | ±{r['Conf_Interval_95']:.3f} | {r['Test_Used']} | {r['P_Value_vs_Baseline']:.1e} | {r['Effect_Direction']} |")
