#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_model import get_alexnet_full, get_resnet18_full, get_resnet34_full, get_vgg16_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export torchvision models to ONNX files under models/onnx.")
    parser.add_argument(
        "--models",
        default="resnet18,resnet34,alexnet,vgg16",
        help="Comma-separated model list",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models" / "onnx"),
        help="Where to write ONNX models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    available_models = {
        "resnet18": get_resnet18_full,
        "resnet34": get_resnet34_full,
        "alexnet": get_alexnet_full,
        "vgg16": get_vgg16_full,
    }

    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        if name not in available_models:
            raise SystemExit(f"Unsupported model: {name}")
        model = available_models[name]().eval()
        out_path = output_dir / f"{name}.onnx"
        print(f"Exporting {name} -> {out_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


if __name__ == "__main__":
    main()
