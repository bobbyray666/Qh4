# SPDX-License-Identifier: Apache-2.0
"""CUDA C++ compiler probe, NVRTC->PTX->ptxas verification, and real GPU execution runner."""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors
from .cpu_reference import mod_inverse
from .mixing import transform_forward, transform_inverse
from .validation import (
    FIELD_MODULUS,
    INVERSE_EXPONENT,
    QH4InternalStateError,
    validate_coefficients,
    validate_rounds,
    validate_vector,
)

CUDA_SOURCE_PATH = Path(__file__).resolve().parent / "qh4_cuda_kernels.cu"


class CudaHardwareUnavailableError(RuntimeError):
    """Raised explicitly when physical NVIDIA CUDA GPU execution is requested without a GPU."""


class CudaCompilationError(RuntimeError):
    """Raised explicitly when NVRTC / nvcc / ptxas compilation fails."""


def _find_libnvrtc_path() -> str | None:
    env_path = os.environ.get("QH4_LIBNVRTC_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    candidates: list[str] = []
    for base in sys.path:
        if not base:
            continue
        candidates.extend(glob.glob(os.path.join(base, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")))
    candidates.extend(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*"))
    candidates.extend(glob.glob("/usr/lib/x86_64-linux-gnu/libnvrtc.so*"))

    for cand in sorted(candidates, reverse=True):
        if Path(cand).is_file():
            return cand
    return None


def _find_ptxas_path() -> str | None:
    which_ptxas = shutil.which("ptxas")
    if which_ptxas:
        return which_ptxas
    for base in sys.path:
        if not base:
            continue
        for rel in (
            os.path.join(base, "triton", "backends", "nvidia", "bin", "ptxas"),
            os.path.join(base, "nvidia", "cuda_nvcc", "bin", "ptxas"),
        ):
            if Path(rel).is_file():
                return rel
    return None


def probe_cuda_toolchain() -> dict[str, Any]:
    """Return factual availability of NVIDIA GPU driver, nvcc, NVRTC, and ptxas."""
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    nvrtc_path = _find_libnvrtc_path()
    ptxas_path = _find_ptxas_path()

    gpu_devices: list[str] = []
    cuda_available = False
    torch_version = None
    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    return {
        "cuda_available": cuda_available,
        "gpu_devices": gpu_devices,
        "nvidia_smi": nvidia_smi,
        "nvcc": nvcc,
        "libnvrtc": nvrtc_path,
        "ptxas": ptxas_path,
        "torch_version": torch_version,
        "cuda_source_exists": CUDA_SOURCE_PATH.is_file(),
    }


def compile_cuda_cpp_to_ptx(target_sm: int = 80) -> dict[str, Any]:
    """Compile qh4_cuda_kernels.cu using NVIDIA NVRTC -> PTX and verify with NVIDIA ptxas -> CUBIN."""
    if not CUDA_SOURCE_PATH.is_file():
        raise CudaCompilationError(f"Missing CUDA source file: {CUDA_SOURCE_PATH}")
    source = CUDA_SOURCE_PATH.read_text(encoding="utf-8")

    nvrtc_path = _find_libnvrtc_path()
    if not nvrtc_path:
        raise CudaHardwareUnavailableError(
            "NVIDIA NVRTC (libnvrtc.so) not found. Install nvidia-cuda-nvrtc-cu12 or CUDA Toolkit."
        )

    libnvrtc = ctypes.CDLL(nvrtc_path)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    libnvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))

    prog = ctypes.c_void_p()
    rc = libnvrtc.nvrtcCreateProgram(
        ctypes.byref(prog),
        source.encode("utf-8"),
        b"qh4_cuda_kernels.cu",
        0,
        None,
        None,
    )
    if rc != 0:
        raise CudaCompilationError(f"nvrtcCreateProgram failed with code {rc}")

    try:
        arch_flag = f"--gpu-architecture=compute_{int(target_sm)}".encode("utf-8")
        opts = (ctypes.c_char_p * 2)(arch_flag, b"--std=c++17")
        compile_rc = libnvrtc.nvrtcCompileProgram(prog, 2, opts)

        log_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_buf = ctypes.create_string_buffer(log_size.value)
        libnvrtc.nvrtcGetProgramLog(prog, log_buf)
        compile_log = log_buf.value.decode("utf-8", errors="replace").strip()

        if compile_rc != 0:
            raise CudaCompilationError(f"NVRTC compilation failed (code {compile_rc}):\n{compile_log}")

        ptx_size = ctypes.c_size_t()
        libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        libnvrtc.nvrtcGetPTX(prog, ptx_buf)
        ptx_bytes = ptx_buf.value
        ptx_text = ptx_bytes.decode("utf-8", errors="replace")
    finally:
        libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    ptxas_path = _find_ptxas_path()
    cubin_bytes: bytes | None = None
    cubin_sha256: str | None = None
    ptxas_stderr = ""
    if ptxas_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            ptx_file = Path(tmpdir) / "qh4_cuda_kernels.ptx"
            cubin_file = Path(tmpdir) / "qh4_cuda_kernels.cubin"
            ptx_file.write_text(ptx_text, encoding="utf-8")
            proc = subprocess.run(
                [ptxas_path, f"-arch=sm_{int(target_sm)}", str(ptx_file), "-o", str(cubin_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            ptxas_stderr = proc.stderr.strip()
            if proc.returncode != 0:
                raise CudaCompilationError(f"ptxas assembly failed (rc={proc.returncode}): {ptxas_stderr}")
            cubin_bytes = cubin_file.read_bytes()
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

    return {
        "status": "compiled",
        "nvrtc_version": f"{major.value}.{minor.value}",
        "target_arch": f"sm_{int(target_sm)}",
        "ptx_bytes": len(ptx_bytes),
        "ptx_sha256": hashlib.sha256(ptx_bytes).hexdigest(),
        "cubin_bytes": len(cubin_bytes) if cubin_bytes is not None else None,
        "cubin_sha256": cubin_sha256,
        "ptxas_path": ptxas_path,
        "compile_log": compile_log,
        "ptxas_stderr": ptxas_stderr,
        "exported_kernels": [
            "qh4_arith_add_kernel",
            "qh4_arith_mul_kernel",
            "qh4_arith_pow5_kernel",
            "qh4_arith_pow_e_kernel",
            "qh4_transform_forward_kernel",
            "qh4_transform_inverse_kernel",
        ],
    }


def _load_cuda_extension():
    """Compile and load PyTorch CUDA C++ extension on physical NVIDIA GPU hardware."""
    try:
        import torch  # type: ignore
        from torch.utils.cpp_extension import load_inline  # type: ignore
    except ImportError as exc:
        raise CudaHardwareUnavailableError(f"PyTorch CUDA extension support unavailable: {exc}") from exc

    if not torch.cuda.is_available():
        raise CudaHardwareUnavailableError("No NVIDIA CUDA GPU device detected (torch.cuda.is_available() == False).")

    cuda_src = CUDA_SOURCE_PATH.read_text(encoding="utf-8") + """
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

torch::Tensor qh4_cuda_forward_launch(
    torch::Tensor x_in,
    torch::Tensor alphas,
    torch::Tensor betas,
    int64_t rounds,
    int64_t q
) {
  TORCH_CHECK(x_in.is_cuda() && x_in.dtype() == torch::kInt64 && x_in.dim() == 2, "x_in must be 2D int64 CUDA tensor");
  int batch_size = (int)x_in.size(0);
  int dim = (int)x_in.size(1);
  auto out = torch::empty_like(x_in);
  const int threads = 128;
  const int blocks = (batch_size + threads - 1) / threads;
  qh4_transform_forward_kernel<<<blocks, threads>>>(
      (const long long*)x_in.data_ptr<int64_t>(),
      (const long long*)alphas.data_ptr<int64_t>(),
      (const long long*)betas.data_ptr<int64_t>(),
      (long long*)out.data_ptr<int64_t>(),
      batch_size,
      dim,
      (int)rounds,
      (long long)q
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor qh4_cuda_inverse_launch(
    torch::Tensor y_in,
    torch::Tensor alpha_invs,
    torch::Tensor betas,
    int64_t rounds,
    int64_t inv_exp,
    int64_t q
) {
  TORCH_CHECK(y_in.is_cuda() && y_in.dtype() == torch::kInt64 && y_in.dim() == 2, "y_in must be 2D int64 CUDA tensor");
  int batch_size = (int)y_in.size(0);
  int dim = (int)y_in.size(1);
  auto out = torch::empty_like(y_in);
  const int threads = 128;
  const int blocks = (batch_size + threads - 1) / threads;
  qh4_transform_inverse_kernel<<<blocks, threads>>>(
      (const long long*)y_in.data_ptr<int64_t>(),
      (const long long*)alpha_invs.data_ptr<int64_t>(),
      (const long long*)betas.data_ptr<int64_t>(),
      (long long*)out.data_ptr<int64_t>(),
      batch_size,
      dim,
      (int)rounds,
      (long long)inv_exp,
      (long long)q
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""
    cpp_src = """
#include <torch/extension.h>
torch::Tensor qh4_cuda_forward_launch(torch::Tensor x_in, torch::Tensor alphas, torch::Tensor betas, int64_t rounds, int64_t q);
torch::Tensor qh4_cuda_inverse_launch(torch::Tensor y_in, torch::Tensor alpha_invs, torch::Tensor betas, int64_t rounds, int64_t inv_exp, int64_t q);
"""
    return load_inline(
        name="qh4_cuda_cpp_ext",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=["qh4_cuda_forward_launch", "qh4_cuda_inverse_launch"],
        with_cuda=True,
        verbose=False,
    )


def run_cuda_cpp_batch(
    batch: Sequence[Sequence[int]],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
) -> dict[str, Any]:
    """Execute native CUDA C++ forward + inverse kernels on physical NVIDIA GPU hardware."""
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise CudaHardwareUnavailableError("No physical NVIDIA CUDA GPU is available.")
    if len(batch) == 0:
        raise ValueError("Batch cannot be empty.")

    dim = len(batch[0])
    rows = [validate_vector(r, expected_dim=dim, name=f"batch[{i}]") for i, r in enumerate(batch)]
    av, bv = validate_coefficients(alphas, betas, expected_dim=dim)
    n_rounds = validate_rounds(dim if rounds is None else rounds)
    a_invs = [mod_inverse(a) for a in av]

    ext = _load_cuda_extension()
    x_dev = torch.tensor(rows, dtype=torch.int64, device="cuda")
    a_dev = torch.tensor(av, dtype=torch.int64, device="cuda")
    a_inv_dev = torch.tensor(a_invs, dtype=torch.int64, device="cuda")
    b_dev = torch.tensor(bv, dtype=torch.int64, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    start_evt = torch.cuda.Event(enable_timing=True)
    mid_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    y_dev = ext.qh4_cuda_forward_launch(x_dev, a_dev, b_dev, n_rounds, FIELD_MODULUS)
    mid_evt.record()
    rec_dev = ext.qh4_cuda_inverse_launch(y_dev, a_inv_dev, b_dev, n_rounds, INVERSE_EXPONENT, FIELD_MODULUS)
    end_evt.record()
    torch.cuda.synchronize()

    fwd_out = y_dev.cpu().tolist()
    inv_out = rec_dev.cpu().tolist()
    cpu_fwd = [transform_forward(r, av, bv, rounds=n_rounds) for r in rows]
    cpu_inv = [transform_inverse(r, av, bv, rounds=n_rounds) for r in cpu_fwd]

    if fwd_out != cpu_fwd:
        raise QH4InternalStateError("CUDA_CPP_CPU_FWD_MISMATCH", "CUDA C++ forward kernel diverged from CPU reference.")
    if inv_out != cpu_inv or inv_out != rows:
        raise QH4InternalStateError("CUDA_CPP_CPU_INV_MISMATCH", "CUDA C++ inverse kernel diverged from CPU reference.")

    return {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "batch_size": len(rows),
        "dim": dim,
        "rounds": n_rounds,
        "forward_ms": round(float(start_evt.elapsed_time(mid_evt)), 4),
        "inverse_ms": round(float(mid_evt.elapsed_time(end_evt)), 4),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "bit_identical_equality": True,
    }


def run_cuda_cpp_all_configs(batch_size: int = 128, seed: int = 2026) -> dict[str, Any]:
    """Run native CUDA C++ differential verification across all 8 QH4 configurations on real GPU."""
    results = []
    for name, cfg in BLOCK_CONFIGS.items():
        alphas, betas = derive_coefficients(name, seed=seed)
        batch = generate_random_vectors(name, count=batch_size, seed=seed)
        res = run_cuda_cpp_batch(batch, alphas, betas, rounds=cfg.default_rounds)
        results.append({"config": name, **res})
    return {"status": "passed", "configs_tested": len(results), "results": results}
