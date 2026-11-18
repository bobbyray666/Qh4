# SPDX-License-Identifier: Apache-2.0
"""Trusted CPU Reference Baseline for QH4-FieldPermutation v0.2.0 over F_{65,537}.

Uses NumPy vectorized uint32/int64 paths when available, with an exact pure-Python
integer backend fallback so verification executes identically in minimal runtimes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .validation import (
    FIELD_MODULUS,
    FORWARD_EXPONENT,
    INVERSE_EXPONENT,
    QH4InternalStateError,
    QH4ValidationError,
    validate_alpha,
    validate_beta,
    validate_coefficients,
    validate_field_element,
    validate_strict_int,
    validate_vector,
)

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def cpu_backend_name() -> str:
    return "numpy" if HAS_NUMPY else "pure_python"


def is_prime(n: int) -> bool:
    """Deterministic primality check for field modulus verification."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = math.isqrt(n)
    d = 3
    while d <= limit:
        if n % d == 0:
            return False
        d += 2
    return True


def field_add(a: int, b: int, q: int = FIELD_MODULUS) -> int:
    """Compute (a + b) mod q in F_q with strict input validation."""
    av = validate_field_element(a, name="a", q=q)
    bv = validate_field_element(b, name="b", q=q)
    return (av + bv) % q


def field_sub(a: int, b: int, q: int = FIELD_MODULUS) -> int:
    """Compute (a - b) mod q in F_q with strict input validation."""
    av = validate_field_element(a, name="a", q=q)
    bv = validate_field_element(b, name="b", q=q)
    return (av - bv) % q


def field_mul(a: int, b: int, q: int = FIELD_MODULUS) -> int:
    """Compute (a * b) mod q in F_q with strict input validation."""
    av = validate_field_element(a, name="a", q=q)
    bv = validate_field_element(b, name="b", q=q)
    return (av * bv) % q


def mod_inverse(a: int, q: int = FIELD_MODULUS) -> int:
    """Compute multiplicative inverse a^{-1} mod q using Python's extended-GCD pow."""
    a_val = validate_alpha(a, name="a", q=q)
    inv = pow(a_val, -1, q)
    if (a_val * inv) % q != 1:
        raise QH4InternalStateError(
            "MOD_INV_INVARIANT_FAILED",
            f"Computed inverse {inv} for {a_val} mod {q} failed a*inv == 1.",
        )
    return inv


def field_pow(base: int, exp: int, q: int = FIELD_MODULUS) -> int:
    """Compute (base ** exp) mod q in F_q, supporting negative exponents via mod_inverse."""
    bv = validate_field_element(base, name="base", q=q)
    ev = validate_strict_int(exp, name="exp")
    if ev < 0:
        if bv == 0:
            raise QH4ValidationError(
                "ZERO_NEGATIVE_EXPONENT",
                f"Cannot raise 0 to negative exponent {ev} in F_{{{q}}}.",
                {"base": bv, "exp": ev},
            )
        inv = mod_inverse(bv, q=q)
        return pow(inv, -ev, q)
    return pow(bv, ev, q)


def scalar_forward_element(x: int, alpha: int, beta: int, q: int = FIELD_MODULUS) -> int:
    """Forward frozen scalar equation: S(x) = (alpha * x^5 + beta) mod q."""
    xv = validate_field_element(x, name="x", q=q)
    av = validate_alpha(alpha, name="alpha", q=q)
    bv = validate_beta(beta, name="beta", q=q)
    x5 = pow(xv, FORWARD_EXPONENT, q)
    out = (av * x5 + bv) % q
    if out < 0 or out >= q:
        raise QH4InternalStateError("SCALAR_FWD_OUT_OF_RANGE", f"Output {out} out of F_{{{q}}}.")
    return out


def scalar_inverse_element(y: int, alpha: int, beta: int, q: int = FIELD_MODULUS) -> int:
    """Inverse frozen scalar equation: S^{-1}(y) = (alpha^{-1} * (y - beta))^e mod q, e = 52429."""
    yv = validate_field_element(y, name="y", q=q)
    av = validate_alpha(alpha, name="alpha", q=q)
    bv = validate_beta(beta, name="beta", q=q)
    a_inv = mod_inverse(av, q=q)
    inner = (a_inv * ((yv - bv) % q)) % q
    out = pow(inner, INVERSE_EXPONENT, q)
    if out < 0 or out >= q:
        raise QH4InternalStateError("SCALAR_INV_OUT_OF_RANGE", f"Output {out} out of F_{{{q}}}.")
    return out


def scalar_forward_vector(
    x: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    q: int = FIELD_MODULUS,
) -> list[int]:
    """Apply scalar layer S element-wise across vector x without mutating x."""
    xv = validate_vector(x, name="x", q=q)
    av, bv = validate_coefficients(alphas, betas, expected_dim=len(xv), q=q)

    if HAS_NUMPY and np is not None:
        arr = np.asarray(xv, dtype=np.int64)
        a_arr = np.asarray(av, dtype=np.int64)
        b_arr = np.asarray(bv, dtype=np.int64)
        x2 = (arr * arr) % q
        x4 = (x2 * x2) % q
        x5 = (x4 * arr) % q
        out = (a_arr * x5 + b_arr) % q
        return [int(v) for v in out.tolist()]

    return [(av[i] * pow(xv[i], FORWARD_EXPONENT, q) + bv[i]) % q for i in range(len(xv))]


def scalar_inverse_vector(
    y: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    q: int = FIELD_MODULUS,
) -> list[int]:
    """Apply inverse scalar layer S^{-1} element-wise across vector y without mutating y."""
    yv = validate_vector(y, name="y", q=q)
    av, bv = validate_coefficients(alphas, betas, expected_dim=len(yv), q=q)

    if HAS_NUMPY and np is not None:
        arr = np.asarray(yv, dtype=np.int64)
        a_inv_arr = np.asarray([mod_inverse(a, q=q) for a in av], dtype=np.int64)
        b_arr = np.asarray(bv, dtype=np.int64)
        base = (a_inv_arr * ((arr - b_arr) % q)) % q
        # Square-and-multiply for e = 52429 = 0b1100110011001101
        res = np.ones_like(base)
        cur = base.copy()
        exp = INVERSE_EXPONENT
        while exp > 0:
            if exp & 1:
                res = (res * cur) % q
            cur = (cur * cur) % q
            exp >>= 1
        return [int(v) for v in res.tolist()]

    out: list[int] = []
    for i in range(len(yv)):
        a_inv = mod_inverse(av[i], q=q)
        inner = (a_inv * ((yv[i] - bv[i]) % q)) % q
        out.append(pow(inner, INVERSE_EXPONENT, q))
    return out


def vector_sha256(vec: Sequence[int]) -> str:
    """Compute canonical SHA-256 digest of a validated field vector."""
    xv = validate_vector(vec, name="vec")
    payload = ",".join(str(v) for v in xv).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
