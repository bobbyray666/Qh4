# SPDX-License-Identifier: Apache-2.0
"""Serverless real NVIDIA GPU runner via Modal (https://modal.com).

Usage:
  pip install modal
  modal setup
  modal run qh4-experimental/scripts/modal_cuda_triton_test.py --batch-size 256 --seed 2026
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import modal  # type: ignore
except ImportError as exc:  # pragma: no cover - optional remote cloud runner
    raise SystemExit(
        "Install Modal to launch this serverless NVIDIA GPU test: pip install modal && modal setup"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ninja-build", "g++")
    .pip_install("torch", "triton", "numpy")
    .add_local_dir(str(REPO_ROOT), remote_path="/root/qh4-experimental")
)

app = modal.App("qh4-actual-cuda-triton-verify")


@app.function(image=cuda_image, gpu="T4", timeout=600)
def run_remote_gpu_suite(batch_size: int = 256, seed: int = 2026) -> dict:
    import sys

    sys.path.insert(0, "/root/qh4-experimental")
    from scripts.run_actual_cuda_triton_test import run_suite

    return run_suite(batch_size=batch_size, seed=seed, target_sm=75)


@app.local_entrypoint()
def main(batch_size: int = 256, seed: int = 2026) -> None:
    report = run_remote_gpu_suite.remote(batch_size=batch_size, seed=seed)
    print(json.dumps(report, indent=2))
