# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for input validation, failure logging, and visualization module."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from qh4_experimental.validation import (
    FIELD_MODULUS,
    QH4ValidationError,
    clear_validation_failures,
    configure_validation_log_file,
    get_validation_failures,
    parse_vector_string,
    validate_coefficients,
    validate_inputs_report,
    validate_rounds,
    validate_seed,
    validate_vector,
)
from qh4_experimental.visualization import (
    render_ascii_summary,
    visualize_transformation,
)


class TestValidationAndLogging(unittest.TestCase):
    def setUp(self) -> None:
        clear_validation_failures()
        configure_validation_log_file(None)

    def tearDown(self) -> None:
        configure_validation_log_file(None)

    def test_invalid_inputs_raise_specific_errors_and_log_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "validation_failures.jsonl"
            configure_validation_log_file(log_path)

            cases = [
                (lambda: validate_vector([True, 1, 2, 3, 4, 5, 6, 7], expected_dim=8), "INVALID_TYPE_BOOL"),
                (lambda: validate_vector([1.5, 1, 2, 3, 4, 5, 6, 7], expected_dim=8), "INVALID_TYPE_FLOAT"),
                (lambda: validate_vector([math.nan, 1, 2, 3, 4, 5, 6, 7], expected_dim=8), "NON_FINITE_NUMBER"),
                (lambda: validate_vector([-1, 1, 2, 3, 4, 5, 6, 7], expected_dim=8), "FIELD_RANGE_ERROR"),
                (lambda: validate_vector([FIELD_MODULUS, 1, 2, 3, 4, 5, 6, 7], expected_dim=8), "FIELD_RANGE_ERROR"),
                (lambda: validate_vector([1, 2, 3], expected_dim=8), "DIMENSION_MISMATCH"),
                (lambda: validate_vector("not a vector", expected_dim=8), "INVALID_VECTOR_CONTAINER"),
                (lambda: validate_coefficients([0] * 8, [1] * 8, expected_dim=8), "ALPHA_ZERO_NON_INVERTIBLE"),
                (lambda: validate_rounds(0), "INVALID_ROUND_COUNT"),
                (lambda: validate_seed(-4), "INVALID_SEED"),
                (lambda: parse_vector_string("1, 2, abc, 4, 5, 6, 7, 8", expected_dim=8), "INVALID_TOKEN"),
                (lambda: visualize_transformation(config_name="8B-D8", transform_type="bogus"), "INVALID_TRANSFORM_TYPE"),
            ]

            for fn, expected_code in cases:
                with self.assertRaises(QH4ValidationError) as ctx:
                    fn()
                self.assertEqual(ctx.exception.code, expected_code)

            logged = get_validation_failures()
            self.assertEqual(len(logged), len(cases))
            self.assertEqual([entry["code"] for entry in logged], [code for _, code in cases])

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(cases))
            parsed_last = json.loads(lines[-1])
            self.assertEqual(parsed_last["code"], "INVALID_TRANSFORM_TYPE")

    def test_non_throwing_report_collects_multiple_errors(self) -> None:
        report = validate_inputs_report(
            vector=[1, 2, 70000],
            expected_dim=8,
            alphas=[0] * 8,
            betas=[1] * 8,
            rounds=0,
            seed=-1,
        )
        self.assertFalse(report["valid"])
        codes = {err["code"] for err in report["errors"]}
        self.assertEqual(codes, {"DIMENSION_MISMATCH", "ALPHA_ZERO_NON_INVERTIBLE", "INVALID_ROUND_COUNT", "INVALID_SEED"})
        self.assertGreaterEqual(len(get_validation_failures()), 4)


class TestVisualizationModule(unittest.TestCase):
    def test_all_transform_types_and_artifacts(self) -> None:
        sample = [0, 1, 255, 1024, 32768, 50000, 65535, 65536]
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in ("full", "scalar_only", "mixing_only"):
                res = visualize_transformation(
                    vector=sample,
                    config_name="8B-D8",
                    transform_type=mode,
                    rounds=4,
                    seed=77,
                    engine="auto",
                    output_dir=tmpdir,
                )
                self.assertTrue(res.exact_roundtrip_match)
                self.assertTrue(res.input_unmutated)
                self.assertEqual(res.input_vector, sample)
                self.assertEqual(res.recovered_vector, sample)
                self.assertEqual(len(res.element_permutations), 8)
                self.assertIn("<svg", res.svg_source)
                self.assertIn("digraph", res.dot_source)
                self.assertTrue(any(f.endswith(".svg") for f in res.generated_files))
                self.assertTrue(any(f.endswith(".dot") for f in res.generated_files))
                self.assertTrue(any(f.endswith(".txt") for f in res.generated_files))

            # Confirm scalar_only skips mixing and mixing_only skips scalar
            res_scalar = visualize_transformation(vector=sample, config_name="8B-D8", transform_type="scalar_only", seed=77)
            self.assertEqual(res_scalar.forward_output_vector, res_scalar.after_scalar_vector)

            res_mix = visualize_transformation(vector=sample, config_name="8B-D8", transform_type="mixing_only", seed=77)
            self.assertEqual(res_mix.after_scalar_vector, sample)
            self.assertNotEqual(res_mix.forward_output_vector, sample)

    def test_custom_block_size_layout(self) -> None:
        vec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        res = visualize_transformation(vector=vec, block_size=4, transform_type="full", rounds=6, seed=9)
        self.assertEqual(res.blocks, 4)
        self.assertEqual(res.dim, 12)
        self.assertEqual(res.block_width, 3)
        self.assertTrue(res.exact_roundtrip_match)


if __name__ == "__main__":
    unittest.main()
