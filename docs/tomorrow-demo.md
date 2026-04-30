# Tomorrow Demo

## 1. macOS sanity check
```bash
bash scripts/install_macos.sh
bash scripts/run_mac_benchmark.sh
```

## 2. Jetson Nano run
```bash
bash scripts/install_jetson_nano.sh
bash scripts/run_jetson_nano_benchmark.sh
```

If ARM Compute Library is installed through a separate wrapper:
```bash
export ARMCL_COMMAND='/path/to/run_armcl_wrapper.sh {c_in} {c_out} {height} {width} {runs}'
bash scripts/run_jetson_nano_benchmark.sh
```

## 3. Raspberry Pi run
```bash
bash scripts/install_raspberry_pi.sh
bash scripts/run_raspberry_pi_benchmark.sh
```

If ARM Compute Library is installed through a separate wrapper:
```bash
export ARMCL_COMMAND='/path/to/run_armcl_wrapper.sh {c_in} {c_out} {height} {width} {runs}'
bash scripts/run_raspberry_pi_benchmark.sh
```

## 4. What gets produced
- `artifacts/raw/`: raw microbenchmark samples.
- `artifacts/processed/`: processed tables.
- `artifacts/plots/`: figures.
- `artifacts/comparisons/`: direct backend comparison CSV/Markdown/JSON reports.
