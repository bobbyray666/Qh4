# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint delegating to qh4_experimental.visualization with validation failure reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly as `python3 scripts/visualize_transform.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qh4_experimental.configs import BLOCK_CONFIGS
from qh4_experimental.validation import (
    QH4ValidationError,
    configure_validation_log_file,
    get_validation_failures,
)
from qh4_experimental.visualization import (
    VALID_ENGINES,
    VALID_TRANSFORM_TYPES,
    visualize_transformation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize QH4-FieldPermutation block layout, before/after stages, and element permutations."
    )
    parser.add_argument(
        "--config",
        default=None,
        choices=list(BLOCK_CONFIGS.keys()),
        help="Named block configuration (defaults to 16B-D32 if neither --config nor --block-size is given).",
    )
    parser.add_argument("--block-size", type=int, default=None, help="Custom block count (e.g., 8, 9, 16, 18).")
    parser.add_argument(
        "--transform-type",
        default="full",
        choices=list(VALID_TRANSFORM_TYPES),
        help="Transformation mode: full (L^n o S), scalar_only (S), or mixing_only (L^n).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic coefficient & vector seed.")
    parser.add_argument("--rounds", type=int, default=None, help="Optional custom mixing rounds (defaults to D).")
    parser.add_argument("--preset", choices=["random", "impulse", "ramp"], default="impulse", help="Sample vector preset.")
    parser.add_argument("--vector", default=None, help="Optional explicit vector integers (comma/space/JSON formatted).")
    parser.add_argument("--engine", choices=list(VALID_ENGINES), default="auto", help="Rendering backend.")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory for generated diagram files.")
    parser.add_argument("--log-file", default=None, help="Optional JSONL log path for validation failure reporting.")
    parser.add_argument("--print-ascii", action="store_true", help="Print the ASCII permutation summary table.")
    args = parser.parse_args(argv)

    if args.log_file:
        configure_validation_log_file(args.log_file)

    config_name = args.config if (args.config is not None or args.block_size is not None) else "16B-D32"

    try:
        result = visualize_transformation(
            vector=args.vector,
            config_name=config_name,
            block_size=args.block_size,
            transform_type=args.transform_type,
            rounds=args.rounds,
            seed=args.seed,
            preset=args.preset,
            engine=args.engine,
            output_dir=args.output_dir,
        )
    except QH4ValidationError as exc:
        err_payload = {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message, "context": exc.context},
            "logged_failures": get_validation_failures(limit=10),
        }
        print(json.dumps(err_payload, indent=2), file=sys.stderr)
        return 2

    if args.print_ascii:
        print(result.ascii_summary)
        print()

    summary = {
        "ok": True,
        "config": result.config_name,
        "blocks": result.blocks,
        "dim": result.dim,
        "block_width": result.block_width,
        "transform_type": result.transform_type,
        "rounds": result.rounds,
        "stages_recorded": len(result.stages),
        "exact_roundtrip_match": result.exact_roundtrip_match,
        "input_unmutated": result.input_unmutated,
        "input_sha256": result.input_sha256,
        "output_sha256": result.output_sha256,
        "reconstructed_sha256": result.reconstructed_sha256,
        "optional_backends": result.optional_backends,
        "generated_files": result.generated_files,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
