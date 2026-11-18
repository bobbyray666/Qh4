# SPDX-License-Identifier: Apache-2.0
"""Mathematical Test Suite M01-M10 for QH4-FieldPermutation v0.2.0."""

from __future__ import annotations

import copy
import math
import random
import time
import unittest

from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients
from qh4_experimental.cpu_reference import (
    cpu_backend_name,
    is_prime,
    mod_inverse,
    scalar_forward_element,
    scalar_inverse_element,
    vector_sha256,
)
from qh4_experimental.mixing import (
    mixing_forward_round,
    mixing_inverse_round,
    transform_forward,
    transform_inverse,
)
from qh4_experimental.validation import (
    FIELD_MODULUS,
    FORWARD_EXPONENT,
    INVERSE_EXPONENT,
    QH4ValidationError,
)


class TestMathematicalSuiteM01ToM10(unittest.TestCase):
    def test_m01_modulus_primality(self) -> None:
        self.assertEqual(FIELD_MODULUS, 65537)
        self.assertEqual(FIELD_MODULUS, (1 << 16) + 1)
        self.assertTrue(is_prime(FIELD_MODULUS))

    def test_m02_coprimality_gcd(self) -> None:
        self.assertEqual(math.gcd(FORWARD_EXPONENT, FIELD_MODULUS - 1), 1)
        for alpha in (1, 2, 3, 257, 32768, FIELD_MODULUS - 1):
            self.assertEqual(math.gcd(alpha, FIELD_MODULUS), 1)

    def test_m03_inverse_exponent_identity(self) -> None:
        self.assertEqual(INVERSE_EXPONENT, 52429)
        self.assertEqual((FORWARD_EXPONENT * INVERSE_EXPONENT) % (FIELD_MODULUS - 1), 1)
        # Exhaustive verification across all 65,537 elements of F_{65,537}
        for u in range(FIELD_MODULUS):
            p5 = pow(u, FORWARD_EXPONENT, FIELD_MODULUS)
            rec = pow(p5, INVERSE_EXPONENT, FIELD_MODULUS)
            self.assertEqual(rec, u)

    def test_m04_unique_scalar_permutation_exhaustive(self) -> None:
        # Verify S(x) = (alpha * x^5 + beta) mod q is an exact bijection over all 65,537 elements
        alpha, beta = 31337, 42424
        seen = bytearray(FIELD_MODULUS)
        for x in range(FIELD_MODULUS):
            y = scalar_forward_element(x, alpha, beta)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, FIELD_MODULUS)
            self.assertEqual(seen[y], 0, f"Collision detected at y={y} for x={x}")
            seen[y] = 1
            self.assertEqual(scalar_inverse_element(y, alpha, beta), x)
        self.assertEqual(sum(seen), FIELD_MODULUS)

    def test_m05_field_range_and_validation_errors(self) -> None:
        for bad in (-1, FIELD_MODULUS, 100000, True, False, 3.14, float("nan"), "123", None):
            with self.assertRaises(QH4ValidationError):
                scalar_forward_element(bad, 1, 0)  # type: ignore[arg-type]
        with self.assertRaises(QH4ValidationError):
            scalar_forward_element(5, 0, 1)  # alpha=0 is non-invertible

    def test_m06_multiplicative_inverse_accuracy(self) -> None:
        for a in (1, 2, 3, 5, 17, 257, 65535, 65536):
            inv = mod_inverse(a)
            self.assertEqual((a * inv) % FIELD_MODULUS, 1)

    def test_m07_mixing_layer_invertibility(self) -> None:
        rng = random.Random(707)
        for dim in (8, 9, 16, 18, 32, 36):
            vec = [rng.randrange(0, FIELD_MODULUS) for _ in range(dim)]
            mixed = mixing_forward_round(vec)
            unmixed = mixing_inverse_round(mixed)
            self.assertEqual(unmixed, vec)

    def test_m08_full_transform_round_trip(self) -> None:
        rng = random.Random(808)
        for cfg in BLOCK_CONFIGS.values():
            alphas, betas = derive_coefficients(cfg.name, seed=808)
            vec = [rng.randrange(0, FIELD_MODULUS) for _ in range(cfg.dim)]
            y = transform_forward(vec, alphas, betas, rounds=cfg.default_rounds)
            rec = transform_inverse(y, alphas, betas, rounds=cfg.default_rounds)
            self.assertEqual(rec, vec)

    def test_m09_avalanche_diffusion_and_uniqueness(self) -> None:
        cfg = BLOCK_CONFIGS["8B-D16"]
        alphas, betas = derive_coefficients(cfg.name, seed=909)
        x1 = [0] * cfg.dim
        x2 = [0] * cfg.dim
        x2[0] = 1
        y1 = transform_forward(x1, alphas, betas, rounds=cfg.default_rounds)
        y2 = transform_forward(x2, alphas, betas, rounds=cfg.default_rounds)
        diffs = sum(1 for a, b in zip(y1, y2) if a != b)
        self.assertGreaterEqual(diffs, cfg.dim - 1)

    def test_m10_non_mutation_and_determinism(self) -> None:
        cfg = BLOCK_CONFIGS["9B-D18"]
        alphas, betas = derive_coefficients(cfg.name, seed=1010)
        original = [i * 137 % FIELD_MODULUS for i in range(cfg.dim)]
        snapshot = copy.deepcopy(original)
        y1 = transform_forward(original, alphas, betas, rounds=cfg.default_rounds)
        y2 = transform_forward(original, alphas, betas, rounds=cfg.default_rounds)
        self.assertEqual(original, snapshot)
        self.assertEqual(vector_sha256(y1), vector_sha256(y2))


def run_suite_report() -> dict[str, object]:
    t0 = time.perf_counter()
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMathematicalSuiteM01ToM10)
    results = []
    passed = 0
    for test_case in suite:
        res = unittest.TestResult()
        case_t0 = time.perf_counter()
        test_case.run(res)
        dt_ms = round((time.perf_counter() - case_t0) * 1000.0, 3)
        ok = res.wasSuccessful()
        if ok:
            passed += 1
        err_msg = ""
        if res.failures:
            err_msg = res.failures[0][1]
        elif res.errors:
            err_msg = res.errors[0][1]
        results.append(
            {
                "id": test_case._testMethodName,
                "passed": ok,
                "duration_ms": dt_ms,
                "error": err_msg,
            }
        )
    return {
        "suite": "M01-M10",
        "backend": cpu_backend_name(),
        "total": len(results),
        "passed": passed,
        "all_passed": passed == len(results),
        "total_duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "checks": results,
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
