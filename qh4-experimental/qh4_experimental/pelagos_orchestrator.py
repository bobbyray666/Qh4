# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU Process-Backend Orchestrator (Pelagos-inspired adapter).

Directive 6 compliance:
1. Device Isolation: Sets CUDA_VISIBLE_DEVICES in each worker subprocess BEFORE
   importing torch/triton.
2. Persistent Workers: Spawns workers once and reuses them across batches.
3. Balanced Static Sharding: Quotient + remainder distribution across workers.
4. Stable Contiguous Ordering: Recombines processed shards in original sequence.
5. Ordinary transfer/serialization first (no premature shared memory).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .mixing import transform_forward, transform_inverse
from .validation import QH4ValidationError, validate_batch, validate_coefficients, validate_rounds


@dataclass(frozen=True)
class ShardPlan:
    worker_id: int
    device_id: str
    start_idx: int
    end_idx: int
    count: int


def compute_static_shards(batch_size: int, device_ids: Sequence[str]) -> list[ShardPlan]:
    """Balanced static sharding using quotient and remainder logic."""
    if batch_size < 1:
        raise QH4ValidationError("EMPTY_BATCH", f"Batch size must be >= 1, received {batch_size}.")
    if len(device_ids) < 1:
        raise QH4ValidationError("NO_WORKERS", "At least one worker/device ID is required.")

    num_workers = len(device_ids)
    q, r = divmod(batch_size, num_workers)
    plans: list[ShardPlan] = []
    cursor = 0
    for wid, dev_id in enumerate(device_ids):
        count = q + (1 if wid < r else 0)
        plans.append(
            ShardPlan(
                worker_id=wid,
                device_id=dev_id,
                start_idx=cursor,
                end_idx=cursor + count,
                count=count,
            )
        )
        cursor += count
    return plans


def _worker_loop(
    worker_id: int,
    cuda_visible_device: str,
    backend_mode: str,
    req_queue: mp.Queue,
    resp_queue: mp.Queue,
) -> None:
    """Persistent worker entrypoint. Isolates GPU visibility BEFORE importing torch/triton."""
    startup_t0 = time.perf_counter()
    if backend_mode == "triton_gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)

    try:
        if backend_mode == "triton_gpu":
            from .triton_kernels import build_triton_kernels, run_triton_transform_batch

            # Eagerly verify Triton/CUDA readiness on worker boot so failures are visible
            _ = build_triton_kernels()
            executor = run_triton_transform_batch
        elif backend_mode == "cpu_reference":
            executor = None
        else:
            raise ValueError(f"Unsupported backend_mode={backend_mode!r}")

        boot_ms = (time.perf_counter() - startup_t0) * 1000.0
        resp_queue.put({"type": "ready", "worker_id": worker_id, "pid": os.getpid(), "boot_ms": boot_ms})
    except Exception as exc:
        resp_queue.put(
            {
                "type": "boot_error",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return

    while True:
        task = req_queue.get()
        if task is None or task.get("cmd") == "shutdown":
            resp_queue.put({"type": "shutdown_ack", "worker_id": worker_id})
            break

        req_id = task["req_id"]
        shard = task["shard"]
        alphas = task["alphas"]
        betas = task["betas"]
        rounds = task["rounds"]
        direction = task["direction"]

        t0 = time.perf_counter()
        try:
            if len(shard) == 0:
                out: list[list[int]] = []
            elif backend_mode == "triton_gpu":
                out = executor(shard, alphas, betas, rounds=rounds, direction=direction)  # type: ignore[misc]
            else:
                fn = transform_forward if direction == "forward" else transform_inverse
                out = [fn(row, alphas, betas, rounds=rounds) for row in shard]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp_queue.put(
                {
                    "type": "result",
                    "req_id": req_id,
                    "worker_id": worker_id,
                    "output": out,
                    "worker_ms": elapsed_ms,
                }
            )
        except Exception as exc:
            resp_queue.put(
                {
                    "type": "task_error",
                    "req_id": req_id,
                    "worker_id": worker_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class PelagosProcessOrchestrator:
    """Persistent multi-worker process orchestrator for QH4 batches."""

    def __init__(self, device_ids: Sequence[str], backend_mode: str = "triton_gpu") -> None:
        if len(device_ids) == 0:
            raise QH4ValidationError("NO_WORKERS", "Provide at least one device ID for PelagosProcessOrchestrator.")
        if backend_mode not in ("triton_gpu", "cpu_reference"):
            raise QH4ValidationError("INVALID_BACKEND_MODE", f"Invalid backend_mode={backend_mode!r}.")

        self.device_ids = list(device_ids)
        self.backend_mode = backend_mode
        self._ctx = mp.get_context("spawn")
        self._req_queues: list[mp.Queue] = []
        self._resp_queue: mp.Queue = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._req_counter = 0
        self.boot_stats: list[dict[str, Any]] = []
        self._started = False

    def start(self) -> list[dict[str, Any]]:
        if self._started:
            return self.boot_stats

        for wid, dev_id in enumerate(self.device_ids):
            rq: mp.Queue = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(wid, dev_id, self.backend_mode, rq, self._resp_queue),
                daemon=True,
            )
            proc.start()
            self._req_queues.append(rq)
            self._workers.append(proc)

        ready_map: dict[int, dict[str, Any]] = {}
        errors: list[str] = []
        for _ in range(len(self.device_ids)):
            msg = self._resp_queue.get(timeout=60.0)
            if msg["type"] == "ready":
                ready_map[msg["worker_id"]] = msg
            else:
                errors.append(f"worker {msg.get('worker_id')}: {msg.get('error')}")

        if errors:
            self.close()
            raise RuntimeError("Pelagos worker boot failed (errors surfaced explicitly): " + "; ".join(errors))

        self.boot_stats = [ready_map[i] for i in range(len(self.device_ids))]
        self._started = True
        return self.boot_stats

    def execute_batch(
        self,
        batch: Sequence[Sequence[int]],
        alphas: Sequence[int],
        betas: Sequence[int],
        rounds: int | None = None,
        direction: str = "forward",
    ) -> dict[str, Any]:
        """Distribute batch statically across persistent workers and recombine contiguously."""
        if not self._started:
            self.start()

        validated = validate_batch(batch, expected_dim=len(batch[0]) if len(batch) > 0 else 2)
        dim = len(validated[0])
        av, bv = validate_coefficients(alphas, betas, expected_dim=dim)
        n_rounds = validate_rounds(dim if rounds is None else rounds)
        if direction not in ("forward", "inverse"):
            raise QH4ValidationError("INVALID_DIRECTION", f"Direction must be 'forward' or 'inverse', got {direction!r}.")

        self._req_counter += 1
        req_id = self._req_counter
        plans = compute_static_shards(len(validated), self.device_ids)

        t_dispatch = time.perf_counter()
        for plan in plans:
            shard = validated[plan.start_idx : plan.end_idx]
            self._req_queues[plan.worker_id].put(
                {
                    "cmd": "run",
                    "req_id": req_id,
                    "shard": shard,
                    "alphas": av,
                    "betas": bv,
                    "rounds": n_rounds,
                    "direction": direction,
                }
            )

        results: dict[int, list[list[int]]] = {}
        worker_timings: dict[int, float] = {}
        errors: list[str] = []
        for _ in range(len(plans)):
            msg = self._resp_queue.get(timeout=120.0)
            if msg.get("req_id") != req_id:
                continue
            if msg["type"] == "result":
                results[msg["worker_id"]] = msg["output"]
                worker_timings[msg["worker_id"]] = msg["worker_ms"]
            else:
                errors.append(f"worker {msg.get('worker_id')}: {msg.get('error')}")

        if errors:
            raise RuntimeError("Worker batch execution error: " + "; ".join(errors))

        # Stable contiguous reassembly in exact original shard order
        reassembled: list[list[int]] = []
        for plan in plans:
            reassembled.extend(results[plan.worker_id])

        total_ms = (time.perf_counter() - t_dispatch) * 1000.0
        return {
            "outputs": reassembled,
            "shards": [
                {"worker_id": p.worker_id, "device_id": p.device_id, "start": p.start_idx, "end": p.end_idx, "count": p.count}
                for p in plans
            ],
            "worker_timings_ms": worker_timings,
            "roundtrip_ms": total_ms,
        }

    def close(self) -> None:
        for rq in self._req_queues:
            try:
                rq.put({"cmd": "shutdown"})
            except Exception:
                pass
        for proc in self._workers:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        self._req_queues.clear()
        self._workers.clear()
        self._started = False

    def __enter__(self) -> "PelagosProcessOrchestrator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
