# SPDX-License-Identifier: Apache-2.0
"""Pytest unit tests for finite field operations over F_{65,537}."""

from __future__ import annotations

import random

import pytest

from qh4_experimental.cpu_reference import (
    field_add,
    field_mul,
    field_pow,
    field_sub,
    is_prime,
    mod_inverse,
)
from qh4_experimental.validation import (
    FIELD_MODULUS,
    FORWARD_EXPONENT,
    INVERSE_EXPONENT,
    QH4ValidationError,
)

Q = FIELD_MODULUS  # 65537
Q_MAX = Q - 1      # 65536


@pytest.fixture(scope="module")
def sample_elements() -> list[int]:
    """Deterministic boundary, midpoint, prime-factor, and random field elements."""
    rng = random.Random(65537)
    boundary = [0, 1, 2, 3, 4, 5, 255, 256, 257, 32768, 32769, Q - 2, Q_MAX]
    random_vals = [rng.randrange(0, Q) for _ in range(32)]
    return boundary + random_vals


@pytest.fixture(scope="module")
def nonzero_elements(sample_elements: list[int]) -> list[int]:
    return [x for x in sample_elements if x != 0]


def test_field_modulus_is_fermat_prime() -> None:
    assert Q == 65537
    assert Q == (1 << 16) + 1
    assert is_prime(Q) is True


# ---------------------------------------------------------------------------
# Addition Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, 0),
        (0, Q_MAX, Q_MAX),
        (Q_MAX, 0, Q_MAX),
        (Q_MAX, 1, 0),
        (Q_MAX, 2, 1),
        (Q_MAX, Q_MAX, Q - 2),
        (32768, 32769, 0),
        (32769, 32769, 1),
        (12345, 54321, (12345 + 54321) % Q),
    ],
)
def test_field_add_edge_cases(a: int, b: int, expected: int) -> None:
    res = field_add(a, b)
    assert res == expected
    assert isinstance(res, int)
    assert 0 <= res < Q


def test_field_add_axioms(sample_elements: list[int]) -> None:
    for a in sample_elements:
        # Additive identity
        assert field_add(a, 0) == a
        assert field_add(0, a) == a
        # Additive inverse
        neg_a = (Q - a) % Q
        assert field_add(a, neg_a) == 0

    for a in sample_elements[:15]:
        for b in sample_elements[:15]:
            # Commutativity
            assert field_add(a, b) == field_add(b, a)
            for c in sample_elements[:10]:
                # Associativity
                assert field_add(field_add(a, b), c) == field_add(a, field_add(b, c))


# ---------------------------------------------------------------------------
# Subtraction Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, 0),
        (0, 1, Q_MAX),
        (0, Q_MAX, 1),
        (1, 2, Q_MAX),
        (Q_MAX, Q_MAX, 0),
        (Q_MAX, 0, Q_MAX),
        (100, 200, Q - 100),
        (32768, 32769, Q_MAX),
    ],
)
def test_field_sub_edge_cases(a: int, b: int, expected: int) -> None:
    res = field_sub(a, b)
    assert res == expected
    assert 0 <= res < Q


def test_field_sub_axioms_and_inverse_relation(sample_elements: list[int]) -> None:
    for a in sample_elements:
        assert field_sub(a, 0) == a
        assert field_sub(a, a) == 0

    for a in sample_elements[:20]:
        for b in sample_elements[:20]:
            neg_b = field_sub(0, b)
            assert field_sub(a, b) == field_add(a, neg_b)
            # Round-trip cancellation with addition
            assert field_add(field_sub(a, b), b) == a
            assert field_sub(field_add(a, b), b) == a


# ---------------------------------------------------------------------------
# Multiplication Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (0, 0, 0),
        (0, Q_MAX, 0),
        (Q_MAX, 0, 0),
        (1, Q_MAX, Q_MAX),
        (Q_MAX, 1, Q_MAX),
        # (-1) * (-1) == 1 (mod 65537); tests values > INT32_MAX (65536^2 = 4,294,836,225)
        (Q_MAX, Q_MAX, 1),
        # 256 * 256 == 65536 == -1 (mod 65537)
        (256, 256, Q_MAX),
        # 257 * 255 == 256^2 - 1 == -2 (mod 65537)
        (257, 255, Q - 2),
        # 2 * 32769 == 65538 == 1 (mod 65537)
        (2, 32769, 1),
    ],
)
def test_field_mul_edge_cases(a: int, b: int, expected: int) -> None:
    res = field_mul(a, b)
    assert res == expected
    assert 0 <= res < Q


def test_field_mul_axioms_and_distributivity(sample_elements: list[int]) -> None:
    for a in sample_elements:
        # Multiplicative identity & zero absorption
        assert field_mul(a, 1) == a
        assert field_mul(1, a) == a
        assert field_mul(a, 0) == 0
        assert field_mul(0, a) == 0

    for a in sample_elements[:15]:
        for b in sample_elements[:15]:
            # Commutativity
            assert field_mul(a, b) == field_mul(b, a)
            # Integral domain: no zero divisors
            if a != 0 and b != 0:
                assert field_mul(a, b) != 0
            for c in sample_elements[:10]:
                # Associativity
                assert field_mul(field_mul(a, b), c) == field_mul(a, field_mul(b, c))
                # Left and right distributivity over addition and subtraction
                assert field_mul(a, field_add(b, c)) == field_add(field_mul(a, b), field_mul(a, c))
                assert field_mul(field_add(a, b), c) == field_add(field_mul(a, c), field_mul(b, c))
                assert field_mul(a, field_sub(b, c)) == field_sub(field_mul(a, b), field_mul(a, c))


# ---------------------------------------------------------------------------
# Exponentiation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "exp", "expected"),
    [
        (0, 0, 1),
        (0, 1, 0),
        (0, 5, 0),
        (0, INVERSE_EXPONENT, 0),
        (1, 0, 1),
        (1, 65536, 1),
        (Q_MAX, 0, 1),
        (Q_MAX, 1, Q_MAX),
        (Q_MAX, 2, 1),
        (Q_MAX, 3, Q_MAX),
        # 2^16 == 65536 == -1 (mod 65537), 2^32 == 1 (mod 65537)
        (2, 16, Q_MAX),
        (2, 32, 1),
        # 3 is a primitive root mod 65537 => 3^((q-1)/2) == -1 (mod 65537)
        (3, 32768, Q_MAX),
        (3, 65536, 1),
    ],
)
def test_field_pow_edge_cases(base: int, exp: int, expected: int) -> None:
    res = field_pow(base, exp)
    assert res == expected
    assert 0 <= res < Q


def test_field_pow_fermat_and_exponent_laws(nonzero_elements: list[int]) -> None:
    # Fermat's Little Theorem: a^(q-1) == 1 and a^q == a
    assert field_pow(0, Q) == 0
    for a in nonzero_elements:
        assert field_pow(a, Q - 1) == 1
        assert field_pow(a, Q) == a

    # QH4 frozen permutation exponents: 5 * 52429 == 1 (mod 65536)
    assert (FORWARD_EXPONENT * INVERSE_EXPONENT) % (Q - 1) == 1
    for u in [0] + nonzero_elements:
        assert field_pow(field_pow(u, FORWARD_EXPONENT), INVERSE_EXPONENT) == u
        assert field_pow(field_pow(u, INVERSE_EXPONENT), FORWARD_EXPONENT) == u

    # Exponent algebra laws
    for a in nonzero_elements[:12]:
        for b in nonzero_elements[:12]:
            m, n = 7, 13
            assert field_pow(a, m + n) == field_mul(field_pow(a, m), field_pow(a, n))
            assert field_pow(field_pow(a, m), n) == field_pow(a, m * n)
            assert field_pow(field_mul(a, b), m) == field_mul(field_pow(a, m), field_pow(b, m))


def test_field_pow_negative_exponents(nonzero_elements: list[int]) -> None:
    for a in nonzero_elements:
        assert field_pow(a, -1) == mod_inverse(a)
        assert field_pow(a, -5) == mod_inverse(field_pow(a, 5))
        assert field_mul(field_pow(a, 3), field_pow(a, -3)) == 1

    with pytest.raises(QH4ValidationError) as exc_info:
        field_pow(0, -1)
    assert exc_info.value.code == "ZERO_NEGATIVE_EXPONENT"


# ---------------------------------------------------------------------------
# Modular Inverse Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "expected_inv"),
    [
        (1, 1),
        (Q_MAX, Q_MAX),
        (2, 32769),
        (32769, 2),
        (256, 65281),
        (65281, 256),
        (3, 21846),
        (21846, 3),
    ],
)
def test_mod_inverse_edge_cases(a: int, expected_inv: int) -> None:
    inv = mod_inverse(a)
    assert inv == expected_inv
    assert field_mul(a, inv) == 1
    assert mod_inverse(inv) == a


def test_mod_inverse_fermat_homomorphism_and_self_inverses(nonzero_elements: list[int]) -> None:
    for a in nonzero_elements:
        inv = mod_inverse(a)
        # Fermat's Little Theorem equivalence: a^{-1} == a^{q-2} mod q
        assert inv == field_pow(a, Q - 2)
        assert field_mul(a, inv) == 1
        assert mod_inverse(inv) == a
        # Only 1 and q-1 are self-inverse in a prime field
        if inv == a:
            assert a in (1, Q_MAX)

    for a in nonzero_elements[:15]:
        for b in nonzero_elements[:15]:
            assert mod_inverse(field_mul(a, b)) == field_mul(mod_inverse(a), mod_inverse(b))


def test_mod_inverse_exhaustive_all_nonzero_elements() -> None:
    seen_inverses = bytearray(Q)
    for a in range(1, Q):
        inv = mod_inverse(a)
        assert 1 <= inv < Q
        assert (a * inv) % Q == 1
        assert seen_inverses[inv] == 0
        seen_inverses[inv] = 1
    assert sum(seen_inverses) == Q - 1


# ---------------------------------------------------------------------------
# Validation & Invalid Input Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_val", [-1, -100, Q, Q + 1000, True, False, 1.5, float("nan"), float("inf"), "123", None])
def test_field_ops_reject_invalid_inputs(bad_val: object) -> None:
    with pytest.raises(QH4ValidationError):
        field_add(bad_val, 1)  # type: ignore[arg-type]
    with pytest.raises(QH4ValidationError):
        field_sub(1, bad_val)  # type: ignore[arg-type]
    with pytest.raises(QH4ValidationError):
        field_mul(bad_val, 2)  # type: ignore[arg-type]
    with pytest.raises(QH4ValidationError):
        field_pow(bad_val, 5)  # type: ignore[arg-type]
    with pytest.raises(QH4ValidationError):
        mod_inverse(bad_val)  # type: ignore[arg-type]


def test_mod_inverse_rejects_zero() -> None:
    with pytest.raises(QH4ValidationError) as exc_info:
        mod_inverse(0)
    assert exc_info.value.code == "ALPHA_ZERO_NON_INVERTIBLE"


@pytest.mark.parametrize("bad_exp", [True, False, 2.5, float("nan"), "5", None])
def test_field_pow_rejects_invalid_exponent_types(bad_exp: object) -> None:
    with pytest.raises(QH4ValidationError):
        field_pow(3, bad_exp)  # type: ignore[arg-type]
