# SPDX-License-Identifier: Apache-2.0
"""Reversible two-sweep mixing layer L and full transformation T_n = L^n o S."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .cpu_reference import scalar_forward_vector, scalar_inverse_vector
from .validation import (
    FIELD_MODULUS,
    QH4InternalStateError,
    validate_coefficients,
    validate_rounds,
    validate_vector,
)


def mixing_forward_round(x: Sequence[int], q: int = FIELD_MODULUS) -> list[int]:
    """Execute one complete forward mixing round L without mutating input x.

    Sweep 1: x[i] <- (x[i] + x[i+1]) mod q for i = 0, ..., n-2
    Sweep 2: x[i] <- (x[i] + x[i-1]) mod q for i = n-1, ..., 1
    """
    buf = validate_vector(x, name="x", q=q)
    n = len(buf)
    for i in range(0, n - 1):
        buf[i] = (buf[i] + buf[i + 1]) % q
    for i in range(n - 1, 0, -1):
        buf[i] = (buf[i] + buf[i - 1]) % q
    return buf


def mixing_inverse_round(y: Sequence[int], q: int = FIELD_MODULUS) -> list[int]:
    """Execute exact inverse mixing round L^{-1} without mutating input y.

    Reverses Sweep 2 then Sweep 1 in exact reverse iteration order:
      Reverse Sweep 2: y[i] <- (y[i] - y[i-1]) mod q for i = 1, ..., n-1
      Reverse Sweep 1: y[i] <- (y[i] - y[i+1]) mod q for i = n-2, ..., 0
    """
    buf = validate_vector(y, name="y", q=q)
    n = len(buf)
    for i in range(1, n):
        buf[i] = (buf[i] - buf[i - 1]) % q
    for i in range(n - 2, -1, -1):
        buf[i] = (buf[i] - buf[i + 1]) % q
    return buf


def transform_forward(
    x: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
    q: int = FIELD_MODULUS,
) -> list[int]:
    """Execute T_n(x) = (L^n o S)(x) without mutating input arrays."""
    xv = validate_vector(x, name="x", q=q)
    n_dim = len(xv)
    n_rounds = validate_rounds(n_dim if rounds is None else rounds)
    av, bv = validate_coefficients(alphas, betas, expected_dim=n_dim, q=q)

    state = scalar_forward_vector(xv, av, bv, q=q)
    for _ in range(n_rounds):
        state = mixing_forward_round(state, q=q)
    return state


def transform_inverse(
    y: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
    q: int = FIELD_MODULUS,
) -> list[int]:
    """Execute T_n^{-1}(y) = (S^{-1} o L^{-n})(y) without mutating input arrays."""
    yv = validate_vector(y, name="y", q=q)
    n_dim = len(yv)
    n_rounds = validate_rounds(n_dim if rounds is None else rounds)
    av, bv = validate_coefficients(alphas, betas, expected_dim=n_dim, q=q)

    state = list(yv)
    for _ in range(n_rounds):
        state = mixing_inverse_round(state, q=q)
    return scalar_inverse_vector(state, av, bv, q=q)


@dataclass(frozen=True)
class TraceStage:
    index: int
    direction: str  # "forward" | "inverse"
    kind: str  # "input" | "scalar" | "sweep1" | "sweep2" | "output"
    round_num: int
    label: str
    values: list[int]


def transform_with_trace(
    x: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int | None = None,
    q: int = FIELD_MODULUS,
) -> dict[str, object]:
    """Run forward and inverse transforms while capturing every sweep stage."""
    xv = validate_vector(x, name="x", q=q)
    n_dim = len(xv)
    n_rounds = validate_rounds(n_dim if rounds is None else rounds)
    av, bv = validate_coefficients(alphas, betas, expected_dim=n_dim, q=q)

    stages: list[TraceStage] = []
    idx = 0
    stages.append(TraceStage(idx, "forward", "input", 0, "Input x", list(xv)))
    idx += 1

    # Scalar layer S
    cur = scalar_forward_vector(xv, av, bv, q=q)
    stages.append(TraceStage(idx, "forward", "scalar", 0, "Forward Scalar S", list(cur)))
    idx += 1

    # Mixing rounds L^n (capturing Sweep 1 and Sweep 2 separately)
    for r in range(1, n_rounds + 1):
        buf = list(cur)
        for i in range(0, n_dim - 1):
            buf[i] = (buf[i] + buf[i + 1]) % q
        stages.append(TraceStage(idx, "forward", "sweep1", r, f"L{r} · Sweep 1 (i -> i+1)", list(buf)))
        idx += 1

        for i in range(n_dim - 1, 0, -1):
            buf[i] = (buf[i] + buf[i - 1]) % q
        stages.append(TraceStage(idx, "forward", "sweep2", r, f"L{r} · Sweep 2 (i -> i-1)", list(buf)))
        idx += 1
        cur = buf

    y_out = list(cur)
    stages.append(TraceStage(idx, "forward", "output", n_rounds, "Forward Output y = T_n(x)", list(y_out)))
    idx += 1

    # Inverse L^{-n}
    inv_cur = list(y_out)
    for r in range(n_rounds, 0, -1):
        buf = list(inv_cur)
        for i in range(1, n_dim):
            buf[i] = (buf[i] - buf[i - 1]) % q
        stages.append(TraceStage(idx, "inverse", "sweep2", r, f"Inv L{r} · Undo Sweep 2", list(buf)))
        idx += 1

        for i in range(n_dim - 2, -1, -1):
            buf[i] = (buf[i] - buf[i + 1]) % q
        stages.append(TraceStage(idx, "inverse", "sweep1", r, f"Inv L{r} · Undo Sweep 1", list(buf)))
        idx += 1
        inv_cur = buf

    reconstructed = scalar_inverse_vector(inv_cur, av, bv, q=q)
    stages.append(TraceStage(idx, "inverse", "scalar", 0, "Inverse Scalar S^{-1} (Recovered x)", list(reconstructed)))

    if reconstructed != xv:
        raise QH4InternalStateError(
            "ROUND_TRIP_MISMATCH",
            "Inverse transformation failed to reconstruct input vector exactly.",
            {"input": xv, "reconstructed": reconstructed},
        )

    return {
        "input": list(xv),
        "output": y_out,
        "reconstructed": reconstructed,
        "exact_match": reconstructed == xv,
        "stages": stages,
    }
