# SPDX-License-Identifier: Apache-2.0
"""Independent Triton GPU kernels for QH4-FieldPermutation v0.2.0.

Directive 5 compliance:
- Separate arithmetic kernels (validated independently) and transform kernels.
- Explicit visibility for Triton JIT compilation and autotune warm-up errors.
- Uses 64-bit integer intermediates (tl.int64) to prevent 32-bit overflow since
  (q - 1)^2 = 65536^2 = 4,294,836,225 > INT32_MAX.
"""

from __future__ import annotations

import hashlib
import logging
import os
import traceback
from typing import Any, Sequence

from .cpu_reference import mod_inverse, scalar_forward_vector, scalar_inverse_vector
from .mixing import transform_forward, transform_inverse
from .validation import (
    FIELD_MODULUS,
    INVERSE_EXPONENT,
    QH4InternalStateError,
    validate_coefficients,
    validate_rounds,
    validate_vector,
)

logger = logging.getLogger("qh4.triton")


class TritonUnavailableError(RuntimeError):
    """Raised explicitly when PyTorch/CUDA/Triton runtime is unavailable."""


class TritonCompilationError(RuntimeError):
    """Raised explicitly (never swallowed) when Triton JIT or autotuning fails."""


def is_triton_interpret_enabled() -> bool:
    """Return True only when Triton's official CPU interpreter env var is explicitly active."""
    return os.environ.get("TRITON_INTERPRET", "").strip() in ("1", "true", "TRUE", "yes")


def probe_triton_environment() -> dict[str, Any]:
    """Return factual availability of torch, CUDA, and triton without masking errors."""
    info: dict[str, Any] = {
        "torch_installed": False,
        "cuda_available": False,
        "triton_installed": False,
        "triton_interpret_active": is_triton_interpret_enabled(),
        "triton_version": None,
        "torch_version": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            cnt = int(torch.cuda.device_count())
            info["device_count"] = cnt
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(cnt)]
    except Exception as exc:
        info["error"] = f"torch probe failed: {type(exc).__name__}: {exc}"
        return info

    try:
        import triton  # type: ignore

        info["triton_installed"] = True
        info["triton_version"] = getattr(triton, "__version__", None)
    except Exception as exc:
        info["error"] = f"triton probe failed: {type(exc).__name__}: {exc}"

    return info


def _require_triton_runtime(allow_interpret: bool = False):
    try:
        import torch  # type: ignore
        import triton  # type: ignore
        import triton.language as tl  # type: ignore
    except ImportError as exc:
        raise TritonUnavailableError(
            f"Triton GPU kernels require 'torch' and 'triton' packages: {exc}"
        ) from exc

    if torch.cuda.is_available():
        return torch, triton, tl, "cuda"

    if allow_interpret and is_triton_interpret_enabled():
        return torch, triton, tl, "cpu"

    raise TritonUnavailableError(
        "PyTorch is installed without CUDA GPU availability (torch.cuda.is_available() == False). "
        "Set TRITON_INTERPRET=1 only to test @triton.jit semantics via Triton's CPU interpreter, "
        "or execute on a machine with an NVIDIA CUDA GPU."
    )


try:
    import triton as _triton  # type: ignore
    import triton.language as _tl  # type: ignore

    HAS_TRITON = True
except ImportError:
    _triton = None  # type: ignore
    _tl = None  # type: ignore
    HAS_TRITON = False

if HAS_TRITON:
    tl = _tl  # noqa: N816 - module-level alias required by Triton AST resolver

    @_triton.jit
    def _mod_add_i64(a, b, q: tl.constexpr):
        res = (a + b) % q
        return tl.where(res < 0, res + q, res)

    @_triton.jit
    def _mod_sub_i64(a, b, q: tl.constexpr):
        res = (a - b) % q
        return tl.where(res < 0, res + q, res)

    @_triton.jit
    def _mod_mul_i64(a, b, q: tl.constexpr):
        return (a.to(tl.int64) * b.to(tl.int64)) % q

    @_triton.jit
    def _mod_pow5_i64(x, q: tl.constexpr):
        x64 = x.to(tl.int64) % q
        x2 = (x64 * x64) % q
        x4 = (x2 * x2) % q
        return (x4 * x64) % q

    @_triton.jit
    def _mod_pow_e_i64(x, exp_val: tl.constexpr, q: tl.constexpr):
        base = x.to(tl.int64) % q
        acc = tl.full(base.shape, 1, dtype=tl.int64)
        for bit in range(16):
            if (exp_val >> bit) & 1:
                acc = (acc * base) % q
            base = (base * base) % q
        return acc

    @_triton.jit
    def triton_arith_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, q: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, _mod_add_i64(a, b, q), mask=mask)

    @_triton.jit
    def triton_arith_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, q: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.int64)
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, _mod_mul_i64(a, b, q), mask=mask)

    @_triton.jit
    def triton_arith_pow5_kernel(x_ptr, out_ptr, n_elements, q: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, _mod_pow5_i64(x, q), mask=mask)

    @_triton.jit
    def triton_arith_pow_e_kernel(y_ptr, out_ptr, n_elements, exp_val: tl.constexpr, q: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        y = tl.load(y_ptr + offs, mask=mask, other=0).to(tl.int64)
        tl.store(out_ptr + offs, _mod_pow_e_i64(y, exp_val, q), mask=mask)

    @_triton.jit
    def triton_scalar_forward_kernel(
        x_ptr, alpha_ptr, beta_ptr, out_ptr, batch_size, dim: tl.constexpr, q: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < batch_size
        for col in range(dim):
            offs = rows * dim + col
            x = tl.load(x_ptr + offs, mask=row_mask, other=0).to(tl.int64)
            a = tl.load(alpha_ptr + col).to(tl.int64)
            b = tl.load(beta_ptr + col).to(tl.int64)
            x5 = _mod_pow5_i64(x, q)
            y = (a * x5 + b) % q
            tl.store(out_ptr + offs, y, mask=row_mask)

    @_triton.jit
    def triton_scalar_inverse_kernel(
        y_ptr, alpha_inv_ptr, beta_ptr, out_ptr, batch_size, dim: tl.constexpr, exp_val: tl.constexpr, q: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < batch_size
        for col in range(dim):
            offs = rows * dim + col
            y = tl.load(y_ptr + offs, mask=row_mask, other=0).to(tl.int64)
            a_inv = tl.load(alpha_inv_ptr + col).to(tl.int64)
            b = tl.load(beta_ptr + col).to(tl.int64)
            diff = _mod_sub_i64(y, b, q)
            inner = (a_inv * diff) % q
            x = _mod_pow_e_i64(inner, exp_val, q)
            tl.store(out_ptr + offs, x, mask=row_mask)

    @_triton.jit
    def triton_mixing_forward_kernel(
        buf_ptr, batch_size, dim: tl.constexpr, rounds: tl.constexpr, q: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        """In-place forward mixing L^rounds over cloned device buffer [batch_size, dim]."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < batch_size
        for _ in range(rounds):
            for i in range(dim - 1):
                cur_offs = rows * dim + i
                nxt_offs = rows * dim + (i + 1)
                cur = tl.load(buf_ptr + cur_offs, mask=row_mask, other=0).to(tl.int64)
                nxt = tl.load(buf_ptr + nxt_offs, mask=row_mask, other=0).to(tl.int64)
                tl.store(buf_ptr + cur_offs, _mod_add_i64(cur, nxt, q), mask=row_mask)
            for step in range(dim - 1):
                i = (dim - 1) - step
                cur_offs = rows * dim + i
                prv_offs = rows * dim + (i - 1)
                cur = tl.load(buf_ptr + cur_offs, mask=row_mask, other=0).to(tl.int64)
                prv = tl.load(buf_ptr + prv_offs, mask=row_mask, other=0).to(tl.int64)
                tl.store(buf_ptr + cur_offs, _mod_add_i64(cur, prv, q), mask=row_mask)

    @_triton.jit
    def triton_mixing_inverse_kernel(
        buf_ptr, batch_size, dim: tl.constexpr, rounds: tl.constexpr, q: tl.constexpr, BLOCK_M: tl.constexpr
    ):
        """In-place inverse mixing L^{-rounds} over cloned device buffer [batch_size, dim]."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < batch_size
        for _ in range(rounds):
            for i in range(1, dim):
                cur_offs = rows * dim + i
                prv_offs = rows * dim + (i - 1)
                cur = tl.load(buf_ptr + cur_offs, mask=row_mask, other=0).to(tl.int64)
                prv = tl.load(buf_ptr + prv_offs, mask=row_mask, other=0).to(tl.int64)
                tl.store(buf_ptr + cur_offs, _mod_sub_i64(cur, prv, q), mask=row_mask)
            for step in range(dim - 1):
                i = (dim - 2) - step
                cur_offs = rows * dim + i
                nxt_offs = rows * dim + (i + 1)
                cur = tl.load(buf_ptr + cur_offs, mask=row_mask, other=0).to(tl.int64)
                nxt = tl.load(buf_ptr + nxt_offs, mask=row_mask, other=0).to(tl.int64)
                tl.store(buf_ptr + cur_offs, _mod_sub_i64(cur, nxt, q), mask=row_mask)

    _TRITON_KERNELS = {
        "arith_add": triton_arith_add_kernel,
        "arith_mul": triton_arith_mul_kernel,
        "arith_pow5": triton_arith_pow5_kernel,
        "arith_pow_e": triton_arith_pow_e_kernel,
        "scalar_forward": triton_scalar_forward_kernel,
        "scalar_inverse": triton_scalar_inverse_kernel,
        "mixing_forward": triton_mixing_forward_kernel,
        "mixing_inverse": triton_mixing_inverse_kernel,
    }
else:
    _TRITON_KERNELS = {}


def build_triton_kernels(allow_no_cuda: bool = False):
    """Return compiled/registered module-level Triton JIT kernels; surfaces unavailability explicitly."""
    if not allow_no_cuda:
        _require_triton_runtime(allow_interpret=True)
    if not HAS_TRITON or not _TRITON_KERNELS:
        raise TritonUnavailableError("Triton kernels require 'triton' and 'torch' packages.")
    return _TRITON_KERNELS


def validate_triton_arithmetic_kernels(
    seed: int = 1234,
    allow_interpret: bool = False,
) -> dict[str, Any]:
    """Validate arithmetic kernels independently from transform kernels."""
    torch, triton, _, device = _require_triton_runtime(allow_interpret=allow_interpret)
    kernels = build_triton_kernels()

    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        n = 4096
        a_cpu = torch.randint(0, FIELD_MODULUS, (n,), generator=gen, dtype=torch.int64)
        b_cpu = torch.randint(0, FIELD_MODULUS, (n,), generator=gen, dtype=torch.int64)
        a_dev = a_cpu.to(device)
        b_dev = b_cpu.to(device)
        out_dev = torch.empty_like(a_dev)

        grid = (triton.cdiv(n, 256),)
        kernels["arith_add"][grid](a_dev, b_dev, out_dev, n, q=FIELD_MODULUS, BLOCK=256)
        if device == "cuda":
            torch.cuda.synchronize()
        add_ref = (a_cpu + b_cpu) % FIELD_MODULUS
        if not torch.equal(out_dev.cpu(), add_ref):
            raise QH4InternalStateError("TRITON_ARITH_ADD_MISMATCH", "Triton add kernel mismatched CPU reference.")

        kernels["arith_mul"][grid](a_dev, b_dev, out_dev, n, q=FIELD_MODULUS, BLOCK=256)
        if device == "cuda":
            torch.cuda.synchronize()
        mul_ref = (a_cpu * b_cpu) % FIELD_MODULUS
        if not torch.equal(out_dev.cpu(), mul_ref):
            raise QH4InternalStateError("TRITON_ARITH_MUL_MISMATCH", "Triton mul kernel mismatched CPU reference.")

        pow5_dev = torch.empty_like(a_dev)
        pow_e_dev = torch.empty_like(a_dev)
        kernels["arith_pow5"][grid](a_dev, pow5_dev, n, q=FIELD_MODULUS, BLOCK=256)
        kernels["arith_pow_e"][grid](pow5_dev, pow_e_dev, n, exp_val=INVERSE_EXPONENT, q=FIELD_MODULUS, BLOCK=256)
        if device == "cuda":
            torch.cuda.synchronize()
        if not torch.equal(pow_e_dev.cpu(), a_cpu):
            raise QH4InternalStateError("TRITON_ARITH_POW_INVERSE_MISMATCH", "Triton (x^5)^e != x.")

        return {
            "status": "passed",
            "execution_backend": "cuda_gpu" if device == "cuda" else "triton_interpreter_cpu",
            "elements_tested": n,
            "seed": seed,
        }
    except (TritonUnavailableError, QH4InternalStateError):
        raise
    except Exception as exc:
        logger.error("Triton arithmetic validation failure:\n%s", traceback.format_exc())
        raise TritonCompilationError(f"Triton arithmetic validation failed: {type(exc).__name__}: {exc}") from exc


def run_triton_transform_batch(
    batch: Sequence[Sequence[int]],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
    direction: str = "forward",
    allow_interpret: bool = False,
) -> list[list[int]]:
    """Run forward or inverse transform via Triton kernels without mutating input."""
    torch, triton, _, device = _require_triton_runtime(allow_interpret=allow_interpret)
    if len(batch) == 0:
        raise ValueError("Batch cannot be empty.")
    dim = len(batch[0])
    validated_rows = [validate_vector(r, expected_dim=dim, name=f"batch[{i}]") for i, r in enumerate(batch)]
    av, bv = validate_coefficients(alphas, betas, expected_dim=dim)
    n_rounds = validate_rounds(dim if rounds is None else rounds)

    kernels = build_triton_kernels()
    try:
        batch_size = len(validated_rows)
        x_dev = torch.tensor(validated_rows, dtype=torch.int64, device=device)
        out_dev = torch.empty_like(x_dev)
        b_dev = torch.tensor(bv, dtype=torch.int64, device=device)
        grid = (triton.cdiv(batch_size, 64),)

        if direction == "forward":
            a_dev = torch.tensor(av, dtype=torch.int64, device=device)
            kernels["scalar_forward"][grid](
                x_dev, a_dev, b_dev, out_dev, batch_size, dim=dim, q=FIELD_MODULUS, BLOCK_M=64
            )
            kernels["mixing_forward"][grid](
                out_dev, batch_size, dim=dim, rounds=n_rounds, q=FIELD_MODULUS, BLOCK_M=64
            )
        elif direction == "inverse":
            a_inv = [mod_inverse(a) for a in av]
            a_inv_dev = torch.tensor(a_inv, dtype=torch.int64, device=device)
            buf_dev = x_dev.clone()
            kernels["mixing_inverse"][grid](
                buf_dev, batch_size, dim=dim, rounds=n_rounds, q=FIELD_MODULUS, BLOCK_M=64
            )
            kernels["scalar_inverse"][grid](
                buf_dev, a_inv_dev, b_dev, out_dev, batch_size, dim=dim, exp_val=INVERSE_EXPONENT, q=FIELD_MODULUS, BLOCK_M=64
            )
        else:
            raise ValueError(f"Unsupported direction: {direction!r}")

        if device == "cuda":
            torch.cuda.synchronize()
        return out_dev.cpu().tolist()
    except Exception as exc:
        logger.error("Triton transform kernel execution failure:\n%s", traceback.format_exc())
        raise TritonCompilationError(f"Triton transform execution failed: {type(exc).__name__}: {exc}") from exc


def validate_triton_against_cpu(
    batch: Sequence[Sequence[int]],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
    allow_interpret: bool = False,
) -> dict[str, Any]:
    """Cross-validate Triton transform kernels against trusted CPU baseline."""
    _, _, _, device = _require_triton_runtime(allow_interpret=allow_interpret)
    gpu_fwd = run_triton_transform_batch(
        batch, alphas, betas, rounds=rounds, direction="forward", allow_interpret=allow_interpret
    )
    cpu_fwd = [transform_forward(row, alphas, betas, rounds=rounds) for row in batch]
    if gpu_fwd != cpu_fwd:
        raise QH4InternalStateError("TRITON_CPU_FWD_MISMATCH", "Triton forward output diverged from CPU reference.")

    gpu_inv = run_triton_transform_batch(
        gpu_fwd, alphas, betas, rounds=rounds, direction="inverse", allow_interpret=allow_interpret
    )
    cpu_inv = [transform_inverse(row, alphas, betas, rounds=rounds) for row in cpu_fwd]
    if gpu_inv != cpu_inv or gpu_inv != [list(r) for r in batch]:
        raise QH4InternalStateError("TRITON_CPU_INV_MISMATCH", "Triton inverse output diverged from CPU reference.")

    _ = scalar_forward_vector(batch[0], alphas, betas)
    _ = scalar_inverse_vector(cpu_fwd[0], alphas, betas)
    return {
        "status": "passed",
        "execution_backend": "cuda_gpu" if device == "cuda" else "triton_interpreter_cpu",
        "batch_size": len(batch),
        "dim": len(batch[0]),
    }


def compile_triton_kernels_aot(
    target_sm: int = 80,
    dim: int = 16,
    rounds: int = 16,
    block_m: int = 64,
) -> dict[str, Any]:
    """Compile all 8 @triton.jit kernels ahead-of-time to real NVIDIA PTX & CUBIN via LLVM + ptxas.

    Note: This validates real Triton IR lowering and NVIDIA CUDA PTX/CUBIN codegen
    without requiring a physical GPU device driver, because Triton bundles LLVM NVPTX
    and NVIDIA's ptxas assembler. When TRITON_INTERPRET=1 is set in the current process,
    @triton.jit returns InterpretedFunction objects; therefore AOT compilation requires
    TRITON_INTERPRET to be disabled.
    """
    if is_triton_interpret_enabled():
        raise TritonCompilationError(
            "AOT PTX/CUBIN compilation requires TRITON_INTERPRET=0 (currently running under InterpretedFunction)."
        )

    try:
        import triton  # type: ignore
        from triton.backends.compiler import GPUTarget  # type: ignore
        from triton.compiler import ASTSource  # type: ignore
    except ImportError as exc:
        raise TritonUnavailableError(f"Triton compiler unavailable: {exc}") from exc

    kernels = build_triton_kernels(allow_no_cuda=True)
    target = GPUTarget("cuda", int(target_sm), 32)

    specs: list[tuple[str, dict[str, str], dict[str, Any]]] = [
        (
            "arith_add",
            {"a_ptr": "*i64", "b_ptr": "*i64", "out_ptr": "*i64", "n_elements": "i32", "q": "constexpr", "BLOCK": "constexpr"},
            {"q": FIELD_MODULUS, "BLOCK": 256},
        ),
        (
            "arith_mul",
            {"a_ptr": "*i64", "b_ptr": "*i64", "out_ptr": "*i64", "n_elements": "i32", "q": "constexpr", "BLOCK": "constexpr"},
            {"q": FIELD_MODULUS, "BLOCK": 256},
        ),
        (
            "arith_pow5",
            {"x_ptr": "*i64", "out_ptr": "*i64", "n_elements": "i32", "q": "constexpr", "BLOCK": "constexpr"},
            {"q": FIELD_MODULUS, "BLOCK": 256},
        ),
        (
            "arith_pow_e",
            {"y_ptr": "*i64", "out_ptr": "*i64", "n_elements": "i32", "exp_val": "constexpr", "q": "constexpr", "BLOCK": "constexpr"},
            {"exp_val": INVERSE_EXPONENT, "q": FIELD_MODULUS, "BLOCK": 256},
        ),
        (
            "scalar_forward",
            {"x_ptr": "*i64", "alpha_ptr": "*i64", "beta_ptr": "*i64", "out_ptr": "*i64", "batch_size": "i32", "dim": "constexpr", "q": "constexpr", "BLOCK_M": "constexpr"},
            {"dim": dim, "q": FIELD_MODULUS, "BLOCK_M": block_m},
        ),
        (
            "scalar_inverse",
            {"y_ptr": "*i64", "alpha_inv_ptr": "*i64", "beta_ptr": "*i64", "out_ptr": "*i64", "batch_size": "i32", "dim": "constexpr", "exp_val": "constexpr", "q": "constexpr", "BLOCK_M": "constexpr"},
            {"dim": dim, "exp_val": INVERSE_EXPONENT, "q": FIELD_MODULUS, "BLOCK_M": block_m},
        ),
        (
            "mixing_forward",
            {"buf_ptr": "*i64", "batch_size": "i32", "dim": "constexpr", "rounds": "constexpr", "q": "constexpr", "BLOCK_M": "constexpr"},
            {"dim": dim, "rounds": rounds, "q": FIELD_MODULUS, "BLOCK_M": block_m},
        ),
        (
            "mixing_inverse",
            {"buf_ptr": "*i64", "batch_size": "i32", "dim": "constexpr", "rounds": "constexpr", "q": "constexpr", "BLOCK_M": "constexpr"},
            {"dim": dim, "rounds": rounds, "q": FIELD_MODULUS, "BLOCK_M": block_m},
        ),
    ]

    compiled_reports: list[dict[str, Any]] = []
    for name, signature, constexprs in specs:
        src = ASTSource(fn=kernels[name], signature=signature, constexprs=constexprs)
        compiled = triton.compile(src, target=target)
        ptx_text = compiled.asm.get("ptx", "")
        cubin_bytes = compiled.asm.get("cubin", b"")
        compiled_reports.append(
            {
                "kernel": name,
                "ptx_bytes": len(ptx_text.encode("utf-8")) if isinstance(ptx_text, str) else len(ptx_text),
                "cubin_bytes": len(cubin_bytes),
                "cubin_sha256": hashlib.sha256(cubin_bytes).hexdigest(),
                "n_regs": getattr(compiled.metadata, "n_regs", None),
                "n_spills": getattr(compiled.metadata, "n_spills", None),
                "shared_bytes": getattr(compiled.metadata, "shared", None),
            }
        )

    return {
        "status": "compiled",
        "triton_version": getattr(triton, "__version__", None),
        "target": f"cuda:sm_{target_sm}",
        "dim": dim,
        "rounds": rounds,
        "block_m": block_m,
        "kernel_count": len(compiled_reports),
        "kernels": compiled_reports,
    }
