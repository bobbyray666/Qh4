# SPDX-License-Identifier: Apache-2.0
"""Directive 4 Validation: 10,000 reproducible randomized vectors per configuration."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import unittest

from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors
from qh4_experimental.mixing import transform_forward, transform_inverse
from qh4_experimental.validation import FIELD_MODULUS

IMPLEMENTATION_VERSION = "0.2.0"
HASH_SPECIFICATION = (
    "SHA-256 stream of comma-separated decimal integers ending with newline for each vector in order of generation"
)


def validate_all_eight_configs(vectors_per_config: int = 10000, base_seed: int = 202602) -> dict[str, object]:
    """Run randomized round-trip & zero-mutation checks across all 8 block configurations."""
    t_all = time.perf_counter()
    config_reports = []

    for idx, (cfg_name, cfg) in enumerate(BLOCK_CONFIGS.items()):
        cfg_seed = base_seed + idx
        alphas, betas = derive_coefficients(cfg_name, seed=cfg_seed)
        vectors = generate_random_vectors(cfg_name, count=vectors_per_config, seed=cfg_seed)

        t0 = time.perf_counter()
        in_hasher = hashlib.sha256()
        out_hasher = hashlib.sha256()
        rec_hasher = hashlib.sha256()

        mismatches = 0
        mutations = 0
        range_violations = 0

        for vec in vectors:
            snapshot = list(vec)
            in_hasher.update((",".join(str(v) for v in snapshot) + "\n").encode("utf-8"))

            y = transform_forward(vec, alphas, betas, rounds=cfg.default_rounds)
            if vec != snapshot:
                mutations += 1

            for val in y:
                if val < 0 or val >= FIELD_MODULUS:
                    range_violations += 1

            out_hasher.update((",".join(str(v) for v in y) + "\n").encode("utf-8"))

            rec = transform_inverse(y, alphas, betas, rounds=cfg.default_rounds)
            rec_hasher.update((",".join(str(v) for v in rec) + "\n").encode("utf-8"))

            if rec != snapshot:
                mismatches += 1

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        full_in_sha = in_hasher.hexdigest()
        full_out_sha = out_hasher.hexdigest()
        full_rec_sha = rec_hasher.hexdigest()

        config_reports.append(
            {
                "config": cfg.name,
                "blocks": cfg.blocks,
                "dim": cfg.dim,
                "block_width": cfg.block_width,
                "rounds": cfg.default_rounds,
                "seed": cfg_seed,
                "vectors_tested": vectors_per_config,
                "alphas_preview": alphas[:4],
                "betas_preview": betas[:4],
                "mismatches": mismatches,
                "input_mutations": mutations,
                "range_violations": range_violations,
                "input_sha256": full_in_sha,
                "output_sha256": full_out_sha,
                "reconstructed_sha256": full_rec_sha,
                "roundtrip_sha256_match": (full_in_sha == full_rec_sha),
                "duration_ms": elapsed_ms,
                "vectors_per_sec": round((vectors_per_config / (elapsed_ms / 1000.0)), 1) if elapsed_ms > 0 else 0,
                "passed": (mismatches == 0 and mutations == 0 and range_violations == 0 and full_in_sha == full_rec_sha),
            }
        )

    all_passed = all(r["passed"] for r in config_reports)
    return {
        "version": IMPLEMENTATION_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hash_specification": HASH_SPECIFICATION,
        "base_seed": base_seed,
        "vectors_per_config": vectors_per_config,
        "total_vectors": vectors_per_config * len(BLOCK_CONFIGS),
        "all_passed": all_passed,
        "total_duration_ms": round((time.perf_counter() - t_all) * 1000.0, 2),
        "configs": config_reports,
    }


class TestEightConfigurations10K(unittest.TestCase):
    def test_all_eight_configs_10000_vectors(self) -> None:
        report = validate_all_eight_configs(vectors_per_config=10000, base_seed=202602)
        self.assertTrue(report["all_passed"], json.dumps(report, indent=2))


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 202602
    print(json.dumps(validate_all_eight_configs(vectors_per_config=count, base_seed=seed), indent=2))
