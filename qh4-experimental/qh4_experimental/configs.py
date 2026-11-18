# SPDX-License-Identifier: Apache-2.0
"""Eight block configurations and deterministic coefficient derivation for QH4 v0.2.0."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .validation import (
    FIELD_MODULUS,
    QH4ValidationError,
    _record_validation_failure,
    validate_seed,
)


@dataclass(frozen=True)
class BlockConfig:
    name: str
    blocks: int
    dim: int
    block_width: int
    default_rounds: int


BLOCK_CONFIGS: dict[str, BlockConfig] = {
    "8B-D8": BlockConfig(name="8B-D8", blocks=8, dim=8, block_width=1, default_rounds=8),
    "8B-D16": BlockConfig(name="8B-D16", blocks=8, dim=16, block_width=2, default_rounds=16),
    "9B-D9": BlockConfig(name="9B-D9", blocks=9, dim=9, block_width=1, default_rounds=9),
    "9B-D18": BlockConfig(name="9B-D18", blocks=9, dim=18, block_width=2, default_rounds=18),
    "16B-D16": BlockConfig(name="16B-D16", blocks=16, dim=16, block_width=1, default_rounds=16),
    "16B-D32": BlockConfig(name="16B-D32", blocks=16, dim=32, block_width=2, default_rounds=32),
    "18B-D18": BlockConfig(name="18B-D18", blocks=18, dim=18, block_width=1, default_rounds=18),
    "18B-D36": BlockConfig(name="18B-D36", blocks=18, dim=36, block_width=2, default_rounds=36),
}


def get_config(name: str) -> BlockConfig:
    """Look up a validated QH4 block configuration."""
    if not isinstance(name, str) or name not in BLOCK_CONFIGS:
        valid = ", ".join(BLOCK_CONFIGS.keys())
        ctx = {"field": "config", "received": str(name), "valid": list(BLOCK_CONFIGS.keys())}
        msg = f"Unknown block configuration {name!r}. Valid configurations: {valid}."
        _record_validation_failure("UNKNOWN_CONFIGURATION", msg, ctx)
        raise QH4ValidationError("UNKNOWN_CONFIGURATION", msg, ctx)
    return BLOCK_CONFIGS[name]


def derive_coefficients(
    config_name: str,
    seed: int = 42,
    q: int = FIELD_MODULUS,
) -> tuple[list[int], list[int]]:
    """Deterministically derive (alphas, betas) for a configuration and seed.

    Ensures alpha_i in [1, q-1] (invertible) and beta_i in [0, q-1].
    """
    cfg = get_config(config_name)
    s = validate_seed(seed)

    alphas: list[int] = []
    betas: list[int] = []
    for idx in range(cfg.dim):
        block_idx = idx // cfg.block_width
        lane_idx = idx % cfg.block_width
        tag = f"QH4-v0.2.0|{cfg.name}|seed={s}|block={block_idx}|lane={lane_idx}".encode("utf-8")
        digest = hashlib.sha256(tag).digest()
        raw_a = int.from_bytes(digest[0:8], "big")
        raw_b = int.from_bytes(digest[8:16], "big")
        alpha_i = 1 + (raw_a % (q - 1))
        beta_i = raw_b % q
        alphas.append(alpha_i)
        betas.append(beta_i)
    return alphas, betas


def generate_random_vectors(
    config_name: str,
    count: int,
    seed: int = 0,
    q: int = FIELD_MODULUS,
) -> list[list[int]]:
    """Generate reproducible randomized vectors in F_q^D using Mersenne Twister seeded deterministically."""
    cfg = get_config(config_name)
    s = validate_seed(seed)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        ctx = {"field": "count", "count": str(count)}
        msg = f"Vector count must be a positive integer, received {count!r}."
        _record_validation_failure("INVALID_VECTOR_COUNT", msg, ctx)
        raise QH4ValidationError("INVALID_VECTOR_COUNT", msg, ctx)
    rng = random.Random(f"QH4-VEC|{cfg.name}|{s}")
    return [[rng.randrange(0, q) for _ in range(cfg.dim)] for _ in range(count)]
