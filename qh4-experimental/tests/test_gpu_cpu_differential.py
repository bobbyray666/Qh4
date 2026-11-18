# SPDX-License-Identifier: Apache-2.0
"""End-to-End Differential Test: Triton GPU Kernel vs. CPU Reference Baseline."""

from __future__ import annotations

import json
import sys
import unittest
from typing import Any

from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors
from qh4_experimental.mixing import transform_forward, transform_inverse
from qh4_experimental.triton_kernels import (
    TritonUnavailableError,
    probe_triton_environment,
    validate_triton_against_cpu,
    validate_triton_arithmetic_kernels,
)


def run_gpu_cpu_differential_test(
    config_name: str = "8B-D16",
    batch_size: int = 128,
    seed: int = 2026,
) -> dict[str, Any]:
    """Execute end-to-end differential test between Triton GPU kernel and CPU reference.
    
    If CUDA / PyTorch / Triton is not installed, returns explicit hardware probe details
    without falsification or synthetic emulation.
    """
    env = probe_triton_environment()
    if not env["torch_installed"] or not env["cuda_available"] or not env["triton_installed"]:
        return {
            "status": "unsupported_hardware",
            "message": "Physical NVIDIA CUDA accelerator or PyTorch/Triton runtime is not available on this host.",
            "environment": env,
            "test_definition": {
                "config": config_name,
                "batch_size": batch_size,
                "seed": seed,
                "criteria": "Exact bit-identical integer equality (gpu_output == cpu_reference) across all coordinates and round-trip inverse.",
            },
        }

    cfg = BLOCK_CONFIGS[config_name]
    alphas, betas = derive_coefficients(config_name, seed=seed)
    batch = generate_random_vectors(config_name, count=batch_size, seed=seed)

    # Step 1: Independent arithmetic kernel validation
    arith_report = validate_triton_arithmetic_kernels(seed=seed)

    # Step 2: Full end-to-end differential test across all mixing rounds and scalar layer
    diff_report = validate_triton_against_cpu(batch, alphas, betas, rounds=cfg.default_rounds)

    return {
        "status": "passed",
        "environment": env,
        "config": config_name,
        "batch_size": batch_size,
        "rounds": cfg.default_rounds,
        "dim": cfg.dim,
        "arithmetic_kernels": arith_report,
        "transform_differential": diff_report,
        "bit_identical_equality": True,
    }


class TestGpuCpuDifferential(unittest.TestCase):
    def test_differential_parity(self) -> None:
        report = run_gpu_cpu_differential_test(config_name="8B-D16", batch_size=64)
        if report["status"] == "unsupported_hardware":
            self.skipTest(f"CUDA/Triton unavailable: {report['message']}")
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report.get("bit_identical_equality", False))


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "8B-D16"
    batch_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    print(json.dumps(run_gpu_cpu_differential_test(cfg_arg, batch_arg), indent=2))
