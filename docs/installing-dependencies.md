# Installing Dependencies

## macOS
```bash
bash scripts/install_macos.sh
```

## Jetson Nano
```bash
bash scripts/install_jetson_nano.sh
```

## Raspberry Pi
```bash
bash scripts/install_raspberry_pi.sh
```

## Manual Python Setup
If you want to manage the virtual environment yourself:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Optional Backends
- `TVM` and `AutoTVM` are not installed by the default scripts. Install them in the same environment if you want those comparisons.
- `ARMCL` is integrated through an external wrapper command exposed via `ARMCL_COMMAND`.
