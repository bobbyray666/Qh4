# SPDX-License-Identifier: Apache-2.0
"""Unified runner to test actual CUDA C++ and Triton kernels.

Behavior:
  - On a machine WITH an NVIDIA CUDA GPU (Google Colab, Modal, RunPod, local GPU):
      1. Captures real hardware identity (`nvidia-smi`, GPU device name, CUDA runtime)
      2. Compiles & executes native CUDA C++ kernels (`qh4_cuda_kernels.cu`) on GPU
      3. Compiles & executes Triton `@triton.jit` kernels (`triton_kernels.py`) on GPU
      4. Compares BOTH GPU backends bit-for-bit against `cpu_reference.py` across all 8 configs
  - On a CPU-only machine WITHOUT a physical GPU:
      - Explicitly reports `cuda_available: false` (never simulates GPU execution)
      - Verifies real compiler toolchains when installed:
          * NVIDIA NVRTC -> PTX -> ptxas CUBIN compilation of `qh4_cuda_kernels.cu`
          * Triton LLVM NVPTX -> PTX -> ptxas CUBIN compilation of all 8 `@triton.jit` kernels
          * Optional `TRITON_INTERPRET=1` interpreter differential check when `--triton-interpret` is passed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors
from qh4_experimental.cuda_runner import (
    compile_cuda_cpp_to_ptx,
    probe_cuda_toolchain,
    run_cuda_cpp_all_configs,
)
from qh4_experimental.triton_kernels import (
    compile_triton_kernels_aot,
    is_triton_interpret_enabled,
    probe_triton_environment,
    validate_triton_against_cpu,
    validate_triton_arithmetic_kernels,
)


def _capture_nvidia_smi() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    proc = subprocess.run(
        [smi, "--query-gpu=name,driver_version,memory.total,pci.bus_id", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def run_suite(
    *,
    batch_size: int = 128,
    seed: int = 2026,
    target_sm: int = 80,
    run_interpret: bool = False,
) -> dict[str, Any]:
    cuda_probe = probe_cuda_toolchain()
    triton_probe = probe_triton_environment()
    report: dict[str, Any] = {
        "hardware": {
            "cuda_available": cuda_probe["cuda_available"],
            "gpu_devices": cuda_probe["gpu_devices"],
            "nvidia_smi": _capture_nvidia_smi(),
            "nvcc": cuda_probe["nvcc"],
            "libnvrtc": cuda_probe["libnvrtc"],
            "ptxas": cuda_probe["ptxas"],
            "triton_version": triton_probe["triton_version"],
            "torch_version": triton_probe["torch_version"],
            "triton_interpret_active": is_triton_interpret_enabled(),
        },
        "compiler_verification": {},
        "execution_verification": {},
    }

    # 1. Real CUDA C++ NVRTC -> PTX -> CUBIN compilation check
    if not is_triton_interpret_enabled():
        if cuda_probe["libnvrtc"]:
            try:
                report["compiler_verification"]["cuda_cpp_nvrtc"] = compile_cuda_cpp_to_ptx(target_sm=target_sm)
            except Exception as exc:
                report["compiler_verification"]["cuda_cpp_nvrtc"] = {"status": "error", "error": str(exc)}
        else:
            report["compiler_verification"]["cuda_cpp_nvrtc"] = {
                "status": "unavailable",
                "reason": "libnvrtc.so not found",
            }

        if triton_probe["triton_installed"]:
            try:
                report["compiler_verification"]["triton_aot_ptx"] = compile_triton_kernels_aot(
                    target_sm=target_sm, dim=16, rounds=16
                )
            except Exception as exc:
                report["compiler_verification"]["triton_aot_ptx"] = {"status": "error", "error": str(exc)}
        else:
            report["compiler_verification"]["triton_aot_ptx"] = {
                "status": "unavailable",
                "reason": "triton package not installed",
            }
    else:
        report["compiler_verification"]["note"] = "Skipped AOT PTX/CUBIN compilation while TRITON_INTERPRET=1 is active."

    # 2. Physical NVIDIA GPU execution when available
    if cuda_probe["cuda_available"] and not is_triton_interpret_enabled():
        import torch  # type: ignore

        t0 = time.perf_counter()
        triton_arith = validate_triton_arithmetic_kernels(seed=seed, allow_interpret=False)
        triton_configs: list[dict[str, Any]] = []
        for name, cfg in BLOCK_CONFIGS.items():
            alphas, betas = derive_coefficients(name, seed=seed)
            batch = generate_random_vectors(name, count=batch_size, seed=seed)
            torch.cuda.reset_peak_memory_stats()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            res = validate_triton_against_cpu(batch, alphas, betas, rounds=cfg.default_rounds, allow_interpret=False)
            e1.record()
            torch.cuda.synchronize()
            triton_configs.append(
                {
                    "config": name,
                    "gpu_elapsed_ms": round(float(e0.elapsed_time(e1)), 4),
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    **res,
                }
            )

        report["execution_verification"]["triton_cuda_gpu"] = {
            "status": "passed",
            "arithmetic": triton_arith,
            "configs": triton_configs,
            "total_wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

        try:
            report["execution_verification"]["cuda_cpp_gpu"] = run_cuda_cpp_all_configs(
                batch_size=batch_size, seed=seed
            )
        except Exception as exc:
            report["execution_verification"]["cuda_cpp_gpu"] = {"status": "error", "error": str(exc)}
        report["overall_status"] = "physical_gpu_verified"
    elif run_interpret or is_triton_interpret_enabled():
        if not is_triton_interpret_enabled():
            report["execution_verification"]["triton_interpreter"] = {
                "status": "error",
                "error": "Re-run with TRITON_INTERPRET=1 environment variable before importing triton.",
            }
            report["overall_status"] = "interpret_env_required"
        else:
            arith = validate_triton_arithmetic_kernels(seed=seed, allow_interpret=True)
            interp_configs = []
            for name, cfg in BLOCK_CONFIGS.items():
                alphas, betas = derive_coefficients(name, seed=seed)
                batch = generate_random_vectors(name, count=min(batch_size, 16), seed=seed)
                res = validate_triton_against_cpu(batch, alphas, betas, rounds=cfg.default_rounds, allow_interpret=True)
                interp_configs.append({"config": name, **res})
            report["execution_verification"]["triton_interpreter"] = {
                "status": "passed",
                "note": "Executed real @triton.jit kernels on CPU via Triton's official TRITON_INTERPRET=1 interpreter. Does NOT establish physical GPU hardware execution.",
                "arithmetic": arith,
                "configs": interp_configs,
            }
            report["overall_status"] = "compiler_and_interpreter_verified_no_physical_gpu"
    else:
        report["execution_verification"] = {
            "status": "no_physical_gpu",
            "message": "No NVIDIA CUDA GPU device is attached to this host. Real hardware execution requires an NVIDIA GPU (Colab T4, Modal, RunPod, or local CUDA machine).",
        }
        report["overall_status"] = "compiler_verified_no_physical_gpu"

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test actual CUDA C++ & Triton kernels for QH4-FieldPermutation.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sm", type=int, default=80, help="Target SM architecture for PTX/CUBIN (e.g., 75 for T4, 80 for A100).")
    parser.add_argument("--triton-interpret", action="store_true", help="Run @triton.jit kernels via TRITON_INTERPRET=1.")
    parser.add_argument("--output-json", default=None, help="Optional file path to save the verification JSON report.")
    args = parser.parse_args(argv)

    if args.triton_interpret and not is_triton_interpret_enabled():
        # Re-exec cleanly so TRITON_INTERPRET=1 is set BEFORE triton is imported
        env = dict(os.environ)
        env["TRITON_INTERPRET"] = "1"
        cmd = [sys.executable, __file__, "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--sm", str(args.sm), "--triton-interpret"]
        if args.output_json:
            cmd.extend(["--output-json", args.output_json])
        return subprocess.call(cmd, env=env)

    report = run_suite(
        batch_size=args.batch_size,
        seed=args.seed,
        target_sm=args.sm,
        run_interpret=args.triton_interpret,
    )
    rendered = json.dumps(report, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
