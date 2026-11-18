# SPDX-License-Identifier: Apache-2.0
"""Structured Benchmarking & Telemetry Harness for QH4-FieldPermutation v0.2.0.

Directive 7 compliance:
Records Cold-start, Warm kernel/compute, Warm batch, and End-to-end latencies
with hardware profiles, software/driver versions, dimensions, batch sizes,
worker counts, input seeds, and SHA-256 checksum verifications.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from typing import Any

from .configs import derive_coefficients, generate_random_vectors, get_config
from .cpu_reference import cpu_backend_name
from .mixing import transform_forward, transform_inverse
from .pelagos_orchestrator import PelagosProcessOrchestrator
from .cuda_runner import probe_cuda_toolchain
from .triton_kernels import probe_triton_environment


def batch_sha256(rows: list[list[int]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update((",".join(str(v) for v in r) + "\n").encode("utf-8"))
    return h.hexdigest()


def collect_environment_metadata() -> dict[str, Any]:
    triton_env = probe_triton_environment()
    cuda_env = probe_cuda_toolchain()
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_reference_backend": cpu_backend_name(),
        "triton_environment": triton_env,
        "cuda_toolchain": cuda_env,
    }


def run_structured_benchmark(
    config_name: str = "8B-D16",
    batch_size: int = 64,
    worker_count: int = 2,
    seed: int = 2026,
    backend_mode: str = "cpu_reference",
) -> dict[str, Any]:
    """Run cold-start, warm compute, warm batch, and end-to-end structured benchmark."""
    cfg = get_config(config_name)
    alphas, betas = derive_coefficients(cfg.name, seed=seed)

    # 1. Input Preparation (counted inside End-to-End)
    e2e_t0 = time.perf_counter()
    inputs = generate_random_vectors(cfg.name, count=batch_size, seed=seed)
    input_checksum = batch_sha256(inputs)

    # 2. Cold-start: Worker startup + first execution
    device_ids = [str(i) for i in range(worker_count)]
    cold_t0 = time.perf_counter()
    with PelagosProcessOrchestrator(device_ids=device_ids, backend_mode=backend_mode) as pool:
        boot_stats = pool.boot_stats
        cold_res = pool.execute_batch(inputs, alphas, betas, rounds=cfg.default_rounds, direction="forward")
        cold_start_ms = (time.perf_counter() - cold_t0) * 1000.0

        # 3. Warm batch: Persistent workers already initialized (transfer + compute + retrieval)
        warm_batch_t0 = time.perf_counter()
        warm_fwd = pool.execute_batch(inputs, alphas, betas, rounds=cfg.default_rounds, direction="forward")
        warm_batch_ms = (time.perf_counter() - warm_batch_t0) * 1000.0

        # 4. Warm inverse round-trip check
        warm_inv = pool.execute_batch(
            warm_fwd["outputs"], alphas, betas, rounds=cfg.default_rounds, direction="inverse"
        )

    e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0

    # 5. Pure local compute reference timing (no IPC/serialization)
    warm_compute_t0 = time.perf_counter()
    cpu_fwd = [transform_forward(r, alphas, betas, rounds=cfg.default_rounds) for r in inputs]
    warm_compute_ms = (time.perf_counter() - warm_compute_t0) * 1000.0

    cpu_inv = [transform_inverse(r, alphas, betas, rounds=cfg.default_rounds) for r in cpu_fwd]
    roundtrip_match = (warm_inv["outputs"] == inputs) and (cpu_inv == inputs) and (warm_fwd["outputs"] == cpu_fwd)

    return {
        "config": cfg.name,
        "blocks": cfg.blocks,
        "dim": cfg.dim,
        "rounds": cfg.default_rounds,
        "batch_size": batch_size,
        "worker_count": worker_count,
        "backend_mode": backend_mode,
        "seed": seed,
        "timings_ms": {
            "cold_start": round(cold_start_ms, 3),
            "warm_compute": round(warm_compute_ms, 3),
            "warm_batch": round(warm_batch_ms, 3),
            "end_to_end": round(e2e_ms, 3),
        },
        "checksums": {
            "input_sha256": input_checksum,
            "forward_sha256": batch_sha256(warm_fwd["outputs"]),
            "inverse_sha256": batch_sha256(warm_inv["outputs"]),
            "roundtrip_exact_match": roundtrip_match,
        },
        "shards": cold_res["shards"],
        "worker_boot_ms": [round(b["boot_ms"], 3) for b in boot_stats],
        "environment": collect_environment_metadata(),
    }


if __name__ == "__main__":
    report = run_structured_benchmark()
    print(json.dumps(report, indent=2))
