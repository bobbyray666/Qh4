# SPDX-License-Identifier: Apache-2.0
"""Visualization module for QH4-FieldPermutation (v0.2.0).

Accepts sample input vectors and configuration parameters (block configuration or
custom block size, transformation type: full | scalar_only | mixing_only, rounds,
and seed), performs strict F_{65,537} validation with failure logging, and generates:
  - Structured before/after forward & inverse trace data
  - Field element scalar bijection & rank permutation analysis
  - Standalone multi-panel SVG diagrams (zero external dependencies)
  - Graphviz DOT data-flow specifications
  - Optional Matplotlib PNG charts when matplotlib is installed
  - Terminal-friendly ASCII diagnostic tables
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .configs import BLOCK_CONFIGS, BlockConfig, derive_coefficients, generate_random_vectors, get_config
from .cpu_reference import scalar_forward_vector, scalar_inverse_vector
from .mixing import TraceStage, mixing_forward_round, mixing_inverse_round
from .validation import (
    FIELD_MODULUS,
    QH4InternalStateError,
    QH4ValidationError,
    _record_validation_failure,
    get_validation_failures,
    parse_vector_string,
    validate_coefficients,
    validate_rounds,
    validate_seed,
    validate_strict_int,
    validate_vector,
)

VALID_TRANSFORM_TYPES = ("full", "scalar_only", "mixing_only")
VALID_ENGINES = ("auto", "svg", "graphviz", "matplotlib", "ascii")


@dataclass(frozen=True)
class FieldElementPermutation:
    """Tracks scalar bijection, mixing perturbation, and rank permutation for one coordinate."""

    index: int
    block_index: int
    lane_index: int
    alpha: int
    beta: int
    input_value: int
    after_scalar: int
    forward_output: int
    after_inverse_mixing: int
    recovered_value: int
    scalar_delta_mod_q: int
    mixing_delta_mod_q: int
    total_delta_mod_q: int
    input_rank: int
    scalar_rank: int
    output_rank: int
    recovered_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualizationResult:
    """Complete visualization bundle and diagnostic report."""

    config_name: str
    blocks: int
    dim: int
    block_width: int
    transform_type: str
    rounds: int
    seed: int
    input_vector: list[int]
    after_scalar_vector: list[int]
    forward_output_vector: list[int]
    after_inverse_mixing_vector: list[int]
    recovered_vector: list[int]
    exact_roundtrip_match: bool
    input_unmutated: bool
    input_sha256: str
    output_sha256: str
    reconstructed_sha256: str
    alphas: list[int]
    betas: list[int]
    stages: list[dict[str, Any]]
    element_permutations: list[dict[str, Any]]
    ascii_summary: str
    dot_source: str
    svg_source: str
    generated_files: list[str]
    optional_backends: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transform_type(transform_type: Any) -> str:
    """Validate transformation mode: 'full', 'scalar_only', or 'mixing_only'."""
    if not isinstance(transform_type, str) or transform_type not in VALID_TRANSFORM_TYPES:
        ctx = {
            "field": "transform_type",
            "received": repr(transform_type),
            "valid": list(VALID_TRANSFORM_TYPES),
        }
        msg = (
            f"Invalid transform_type {transform_type!r}. "
            f"Expected one of: {', '.join(VALID_TRANSFORM_TYPES)}."
        )
        _record_validation_failure("INVALID_TRANSFORM_TYPE", msg, ctx)
        raise QH4ValidationError("INVALID_TRANSFORM_TYPE", msg, ctx)
    return transform_type


def resolve_block_layout(
    *,
    config_name: str | None = None,
    block_size: int | None = None,
    vector_len: int | None = None,
) -> BlockConfig:
    """Resolve and validate a QH4 block configuration from config_name and/or block_size."""
    if config_name is not None:
        cfg = get_config(config_name)
        if block_size is not None:
            bs = validate_strict_int(block_size, name="block_size")
            if bs != cfg.blocks and bs != cfg.block_width and bs != cfg.dim:
                ctx = {
                    "field": "block_size",
                    "value": bs,
                    "config": cfg.name,
                    "expected_blocks": cfg.blocks,
                    "expected_dim": cfg.dim,
                }
                msg = (
                    f"block_size={bs} conflicts with config '{cfg.name}' "
                    f"(blocks={cfg.blocks}, block_width={cfg.block_width}, dim={cfg.dim})."
                )
                _record_validation_failure("BLOCK_SIZE_CONFIG_MISMATCH", msg, ctx)
                raise QH4ValidationError("BLOCK_SIZE_CONFIG_MISMATCH", msg, ctx)
        return cfg

    if block_size is None:
        return BLOCK_CONFIGS["8B-D16"]

    bs = validate_strict_int(block_size, name="block_size")
    if bs < 2 or bs > 256:
        ctx = {"field": "block_size", "value": bs, "min": 2, "max": 256}
        msg = f"block_size={bs} is out of range [2, 256]."
        _record_validation_failure("INVALID_BLOCK_SIZE", msg, ctx)
        raise QH4ValidationError("INVALID_BLOCK_SIZE", msg, ctx)

    # Match an existing named configuration when possible
    if vector_len is not None:
        for candidate in BLOCK_CONFIGS.values():
            if candidate.blocks == bs and candidate.dim == vector_len:
                return candidate
        if vector_len < bs or vector_len % bs != 0:
            ctx = {"field": "block_size", "block_size": bs, "vector_len": vector_len}
            msg = f"Vector dimension {vector_len} must be divisible by block_size={bs} (and >= block_size)."
            _record_validation_failure("INVALID_BLOCK_PARTITION", msg, ctx)
            raise QH4ValidationError("INVALID_BLOCK_PARTITION", msg, ctx)
        width = vector_len // bs
        return BlockConfig(
            name=f"{bs}B-D{vector_len}",
            blocks=bs,
            dim=vector_len,
            block_width=width,
            default_rounds=vector_len,
        )

    for candidate in BLOCK_CONFIGS.values():
        if candidate.blocks == bs and candidate.block_width == 1:
            return candidate
    return BlockConfig(name=f"{bs}B-D{bs}", blocks=bs, dim=bs, block_width=1, default_rounds=bs)


def _derive_custom_layout_coefficients(cfg: BlockConfig, seed: int, q: int = FIELD_MODULUS) -> tuple[list[int], list[int]]:
    if cfg.name in BLOCK_CONFIGS:
        return derive_coefficients(cfg.name, seed=seed, q=q)
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
        alphas.append(1 + (raw_a % (q - 1)))
        betas.append(raw_b % q)
    return alphas, betas


def _value_ranks(values: Sequence[int]) -> list[int]:
    """Compute dense coordinate rank (0 = smallest value in vector, D-1 = largest)."""
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for rank, idx in enumerate(ordered):
        ranks[idx] = rank
    return ranks


def _sha256_vec(values: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(v) for v in values).encode("utf-8")).hexdigest()


def _trace_with_mode(
    x: Sequence[int],
    alphas: Sequence[int],
    betas: Sequence[int],
    rounds: int,
    transform_type: str,
    q: int = FIELD_MODULUS,
) -> tuple[list[TraceStage], list[int], list[int], list[int], list[int]]:
    xv = validate_vector(x, name="vector", q=q)
    n_dim = len(xv)
    av, bv = validate_coefficients(alphas, betas, expected_dim=n_dim, q=q)
    n_rounds = validate_rounds(rounds)
    mode = validate_transform_type(transform_type)

    stages: list[TraceStage] = []
    idx = 0
    stages.append(TraceStage(idx, "forward", "input", 0, "Input x", list(xv)))
    idx += 1

    # Forward path
    if mode in ("full", "scalar_only"):
        after_scalar = scalar_forward_vector(xv, av, bv, q=q)
        stages.append(TraceStage(idx, "forward", "scalar", 0, "Forward Scalar S", list(after_scalar)))
        idx += 1
    else:
        after_scalar = list(xv)

    cur = list(after_scalar)
    if mode in ("full", "mixing_only"):
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
    stages.append(
        TraceStage(
            idx,
            "forward",
            "output",
            0 if mode == "scalar_only" else n_rounds,
            f"Forward Output y ({mode})",
            list(y_out),
        )
    )
    idx += 1

    # Inverse path
    inv_cur = list(y_out)
    if mode in ("full", "mixing_only"):
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

    after_inv_mix = list(inv_cur)

    if mode in ("full", "scalar_only"):
        recovered = scalar_inverse_vector(after_inv_mix, av, bv, q=q)
        stages.append(
            TraceStage(idx, "inverse", "scalar", 0, "Inverse Scalar S^{-1} (Recovered x)", list(recovered))
        )
    else:
        recovered = list(after_inv_mix)
        stages.append(
            TraceStage(idx, "inverse", "output", 0, "Recovered x (After L^{-n})", list(recovered))
        )

    if recovered != xv:
        raise QH4InternalStateError(
            "ROUND_TRIP_MISMATCH",
            "Inverse transformation failed to reconstruct input vector exactly.",
            {"mode": mode, "input": xv, "reconstructed": recovered},
        )

    # Sanity cross-check against single-round helpers when mixing is active
    if mode in ("full", "mixing_only") and n_rounds == 1:
        if mixing_forward_round(after_scalar, q=q) != y_out:
            raise QH4InternalStateError("MIXING_FORWARD_INCONSISTENCY", "Trace diverged from mixing_forward_round.")
        if mixing_inverse_round(y_out, q=q) != after_inv_mix:
            raise QH4InternalStateError("MIXING_INVERSE_INCONSISTENCY", "Trace diverged from mixing_inverse_round.")

    return stages, after_scalar, y_out, after_inv_mix, recovered


def generate_graphviz_dot(
    cfg: BlockConfig,
    rounds: int,
    transform_type: str,
    alphas: Sequence[int],
    betas: Sequence[int],
    stages: Sequence[TraceStage],
) -> str:
    """Produce a Graphviz DOT graph showing block layout, scalar/mixing stages, and permutations."""
    x_in = stages[0].values
    y_out = next(s for s in stages if s.kind == "output" and s.direction == "forward").values
    lines: list[str] = [
        f'digraph "QH4_{cfg.name}_{transform_type}" {{',
        '  rankdir=TB;',
        '  node [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled", color="#CBD5E1"];',
        '  edge [color="#475569", arrowsize=0.7];',
        '',
        '  subgraph cluster_layout {',
        f'    label="Data Arrangement: {cfg.name} ({cfg.blocks} Blocks x {cfg.block_width} Lane(s) = {cfg.dim}D | mode={transform_type})";',
        '    style="rounded"; color="#94A3B8"; bgcolor="#F8FAFC";',
    ]

    for b in range(cfg.blocks):
        lines.append(f'    subgraph cluster_block_{b} {{')
        lines.append(f'      label="B{b}"; style="filled"; fillcolor="#FFFFFF"; color="#CBD5E1";')
        for lane in range(cfg.block_width):
            idx = b * cfg.block_width + lane
            lines.append(
                f'      lane_{idx} [label="x[{idx}]={x_in[idx]}\\na={alphas[idx]}, b={betas[idx]}\\ny[{idx}]={y_out[idx]}", fillcolor="#EFF6FF"];'
            )
        lines.append("    }")
    lines.append("  }")
    lines.append("")

    selected = [
        st
        for st in stages
        if st.kind in ("input", "scalar", "output") or st.round_num in (1, rounds)
    ]

    prev_id = None
    for st in selected:
        node_id = f"stage_{st.index}"
        preview = ", ".join(str(v) for v in st.values[: min(4, len(st.values))])
        suffix = ", ..." if len(st.values) > 4 else ""
        if st.kind == "scalar":
            fill = "#FEF3C7"
            font = ' fontcolor="#92400E"'
        elif st.kind == "output" and st.direction == "forward":
            fill = "#0F4C81"
            font = ' fontcolor="#FFFFFF"'
        else:
            fill = "#DBEAFE" if st.direction == "forward" else "#E0F2FE"
            font = ' fontcolor="#0F172A"'
        lines.append(f'  {node_id} [label="{st.label}\\n[{preview}{suffix}]", fillcolor="{fill}"{font}];')
        if prev_id is not None:
            lines.append(f"  {prev_id} -> {node_id};")
        prev_id = node_id

    lines.append(f'  lane_0 -> stage_{selected[0].index} [style=dashed, label="init"];')
    lines.append("}")
    return "\n".join(lines)


def render_visualization_svg(
    cfg: BlockConfig,
    rounds: int,
    transform_type: str,
    alphas: Sequence[int],
    betas: Sequence[int],
    stages: Sequence[TraceStage],
    permutations: Sequence[FieldElementPermutation],
) -> str:
    """Render a zero-dependency 4-panel SVG highlighting layout, stages, I/O before/after, and permutations."""
    x_in = stages[0].values
    after_scalar = [p.after_scalar for p in permutations]
    y_out = [p.forward_output for p in permutations]
    x_rec = stages[-1].values

    width = 1140
    pad_l = 180
    pad_r = 32
    plot_w = width - pad_l - pad_r
    col_w = plot_w / cfg.dim

    if len(stages) <= 24:
        shown_stages = list(stages)
    else:
        key_idxs = {0, 1, len(stages) - 1}
        out_idx = next(i for i, s in enumerate(stages) if s.kind == "output" and s.direction == "forward")
        key_idxs.add(out_idx)
        step = max(1, len(stages) // 20)
        for i in range(0, len(stages), step):
            key_idxs.add(i)
        shown_stages = [stages[i] for i in sorted(key_idxs)]

    row_h = 18
    arr_y = 66
    heat_y = 152
    heat_h = len(shown_stages) * row_h
    bar_y = heat_y + heat_h + 56
    bar_h = 135
    perm_y = bar_y + bar_h + 74
    perm_h = 148
    total_h = perm_y + perm_h + 44

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" viewBox="0 0 {width} {total_h}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="24" y="30" font-size="16" font-weight="bold" fill="#0F172A">QH4-FieldPermutation {html.escape(cfg.name)} · Mode={html.escape(transform_type)} · Before/After &amp; Permutation Map</text>',
        f'<text x="24" y="48" font-size="11" fill="#475569">Blocks={cfg.blocks} · Dim={cfg.dim} · Width={cfg.block_width} · Rounds={rounds} · Modulus=65,537 · Exact Round-Trip={x_in == x_rec}</text>',
    ]

    # Panel 1: Block & Lane Arrangement
    parts.append(f'<text x="24" y="{arr_y + 22}" font-size="11" font-weight="bold" fill="#0F172A">1. Block Arrangement</text>')
    parts.append(f'<text x="24" y="{arr_y + 38}" font-size="9.5" fill="#64748B">Lane index &amp; x[i]</text>')
    for b in range(cfg.blocks):
        bx = pad_l + b * cfg.block_width * col_w
        bw = cfg.block_width * col_w
        fill = "#F8FAFC" if b % 2 == 0 else "#F1F5F9"
        parts.append(f'<rect x="{bx:.1f}" y="{arr_y}" width="{bw:.1f}" height="56" fill="{fill}" stroke="#CBD5E1"/>')
        parts.append(
            f'<text x="{bx + bw / 2:.1f}" y="{arr_y + 15}" font-size="10" font-weight="bold" text-anchor="middle" fill="#0F4C81">B{b}</text>'
        )
        for lane in range(cfg.block_width):
            idx = b * cfg.block_width + lane
            cx = pad_l + (idx + 0.5) * col_w
            parts.append(f'<text x="{cx:.1f}" y="{arr_y + 32}" font-size="8.5" text-anchor="middle" fill="#0F172A">{idx}</text>')
            parts.append(f'<text x="{cx:.1f}" y="{arr_y + 46}" font-size="7.5" text-anchor="middle" fill="#64748B">{x_in[idx]}</text>')

    # Panel 2: Stage Heatmap (Scalar highlighted in amber, Mixing in cobalt/teal)
    parts.append(
        f'<text x="24" y="{heat_y - 12}" font-size="11" font-weight="bold" fill="#0F172A">2. Stage Heatmap ({len(shown_stages)} of {len(stages)} stages · Amber=Scalar S, Blue=Mixing L)</text>'
    )
    for r_idx, st in enumerate(shown_stages):
        y = heat_y + r_idx * row_h
        label = st.label[:25]
        label_color = "#B45309" if st.kind == "scalar" else ("#0F4C81" if st.direction == "forward" else "#0369A1")
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 12}" font-size="9" text-anchor="end" fill="{label_color}">{html.escape(label)}</text>'
        )
        for c_idx, val in enumerate(st.values):
            x = pad_l + c_idx * col_w
            norm = val / (FIELD_MODULUS - 1)
            if st.kind == "scalar":
                r = round(254 - norm * (254 - 180))
                g = round(243 - norm * (243 - 83))
                b_col = round(199 - norm * (199 - 9))
            elif st.direction == "inverse":
                r = round(224 - norm * (224 - 3))
                g = round(242 - norm * (242 - 105))
                b_col = round(254 - norm * (254 - 161))
            else:
                r = round(239 - norm * (239 - 15))
                g = round(246 - norm * (246 - 76))
                b_col = round(255 - norm * (255 - 129))
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(1.0, col_w - 1):.1f}" height="{row_h - 1}" fill="rgb({r},{g},{b_col})"/>'
            )

    # Panel 3: Before / After Forward & Inverse Grouped Bars
    parts.append(
        f'<text x="24" y="{bar_y - 14}" font-size="11" font-weight="bold" fill="#0F172A">3. Before / After Vector Profile (Slate=x, Amber=S(x), Cobalt=y, Emerald=Recovered x_hat)</text>'
    )
    parts.append(f'<line x1="{pad_l}" y1="{bar_y + bar_h}" x2="{width - pad_r}" y2="{bar_y + bar_h}" stroke="#94A3B8"/>')
    sub_w = max(1.8, (col_w - 4) / 4.0)
    for idx in range(cfg.dim):
        x0 = pad_l + idx * col_w + 2
        series = [
            (x_in[idx], "#64748B"),
            (after_scalar[idx], "#D97706"),
            (y_out[idx], "#0F4C81"),
            (x_rec[idx], "#15803D"),
        ]
        for k, (v, col) in enumerate(series):
            bh = max(2.0, (v / (FIELD_MODULUS - 1)) * bar_h)
            parts.append(
                f'<rect x="{x0 + k * sub_w:.1f}" y="{bar_y + bar_h - bh:.1f}" width="{max(1.2, sub_w - 0.4):.1f}" height="{bh:.1f}" fill="{col}"/>'
            )
        parts.append(f'<text x="{x0 + 2 * sub_w:.1f}" y="{bar_y + bar_h + 13}" font-size="8" text-anchor="middle" fill="#475569">{idx}</text>')

    # Panel 4: Field Element Value & Rank Permutation Flow (x -> S(x) -> y)
    parts.append(
        f'<text x="24" y="{perm_y - 16}" font-size="11" font-weight="bold" fill="#0F172A">4. Field Element Permutation Flow (Top: Input Rank · Middle: S(x) Bijection · Bottom: Output Rank)</text>'
    )
    row_in_y = perm_y + 18
    row_sc_y = perm_y + 72
    row_out_y = perm_y + 126
    parts.append(f'<text x="{pad_l - 10}" y="{row_in_y + 4}" font-size="9" text-anchor="end" fill="#475569">Input Rank(x)</text>')
    parts.append(f'<text x="{pad_l - 10}" y="{row_sc_y + 4}" font-size="9" text-anchor="end" fill="#B45309">After Scalar S</text>')
    parts.append(f'<text x="{pad_l - 10}" y="{row_out_y + 4}" font-size="9" text-anchor="end" fill="#0F4C81">Output Rank(y)</text>')

    for perm in permutations:
        cx_in = pad_l + (perm.input_rank + 0.5) * col_w
        cx_sc = pad_l + (perm.scalar_rank + 0.5) * col_w
        cx_out = pad_l + (perm.output_rank + 0.5) * col_w
        parts.append(
            f'<path d="M {cx_in:.1f} {row_in_y + 7} C {cx_in:.1f} {row_in_y + 28}, {cx_sc:.1f} {row_sc_y - 28}, {cx_sc:.1f} {row_sc_y - 7}" fill="none" stroke="#F59E0B" stroke-opacity="0.6" stroke-width="1.3"/>'
        )
        parts.append(
            f'<path d="M {cx_sc:.1f} {row_sc_y + 7} C {cx_sc:.1f} {row_sc_y + 28}, {cx_out:.1f} {row_out_y - 28}, {cx_out:.1f} {row_out_y - 7}" fill="none" stroke="#0F4C81" stroke-opacity="0.55" stroke-width="1.3"/>'
        )
        parts.append(f'<circle cx="{cx_in:.1f}" cy="{row_in_y}" r="4.5" fill="#64748B"/>')
        parts.append(f'<circle cx="{cx_sc:.1f}" cy="{row_sc_y}" r="4.5" fill="#D97706"/>')
        parts.append(f'<circle cx="{cx_out:.1f}" cy="{row_out_y}" r="4.5" fill="#0F4C81"/>')
        parts.append(f'<text x="{cx_in:.1f}" y="{row_in_y - 9}" font-size="7.5" text-anchor="middle" fill="#334155">#{perm.index}</text>')
        parts.append(f'<text x="{cx_out:.1f}" y="{row_out_y + 15}" font-size="7.5" text-anchor="middle" fill="#0F4C81">#{perm.index}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def render_ascii_summary(
    cfg: BlockConfig,
    transform_type: str,
    rounds: int,
    permutations: Sequence[FieldElementPermutation],
) -> str:
    """Render an informative ASCII table summarizing scalar, mixing, and permutation metrics."""
    lines = [
        f"QH4-FieldPermutation {cfg.name} (blocks={cfg.blocks}, width={cfg.block_width}, dim={cfg.dim}, mode={transform_type}, rounds={rounds}, q={FIELD_MODULUS})",
        "idx | blk | alpha  | beta   | input x | S(x)   | out y  | rec x^ | dScalar | dMix   | rank(x->S->y)",
        "----+-----+--------+--------+---------+--------+--------+--------+---------+--------+--------------",
    ]
    for p in permutations:
        lines.append(
            f"{p.index:>3} | B{p.block_index:<2} | {p.alpha:>6} | {p.beta:>6} | "
            f"{p.input_value:>7} | {p.after_scalar:>6} | {p.forward_output:>6} | {p.recovered_value:>6} | "
            f"{p.scalar_delta_mod_q:>7} | {p.mixing_delta_mod_q:>6} | "
            f"{p.input_rank:>2} -> {p.scalar_rank:>2} -> {p.output_rank:>2}"
        )
    return "\n".join(lines)


def render_with_matplotlib(
    out_path: Path,
    cfg: BlockConfig,
    rounds: int,
    transform_type: str,
    alphas: Sequence[int],
    stages: Sequence[TraceStage],
    permutations: Sequence[FieldElementPermutation],
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import gridspec  # type: ignore

    matrix = [st.values for st in stages]
    x_in = stages[0].values
    after_scalar = [p.after_scalar for p in permutations]
    y_out = [p.forward_output for p in permutations]
    x_rec = stages[-1].values

    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.85, 2.1, 1.4], hspace=0.48)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_title(
        f"QH4 {cfg.name}: Data Arrangement ({cfg.blocks}B x {cfg.block_width}L = {cfg.dim}D, mode={transform_type}, rounds={rounds})",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    for b in range(cfg.blocks):
        x_start = b * cfg.block_width
        ax0.axvspan(
            x_start - 0.45,
            x_start + cfg.block_width - 0.55,
            color="#F1F5F9" if b % 2 == 0 else "#E2E8F0",
            alpha=0.7,
        )
        ax0.text(x_start + (cfg.block_width - 1) / 2.0, 0.78, f"B{b}", ha="center", va="center", fontsize=8, fontweight="bold")
    for idx in range(cfg.dim):
        ax0.text(idx, 0.35, f"x[{idx}]={x_in[idx]}\na={alphas[idx]}", ha="center", va="center", fontsize=6.5)
    ax0.set_xlim(-0.6, cfg.dim - 0.4)
    ax0.set_ylim(0, 1)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    im = ax1.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=FIELD_MODULUS - 1, interpolation="nearest")
    ax1.set_title("Vector Coordinate Evolution Across Forward & Inverse Stages", fontsize=11, fontweight="bold", loc="left")
    ax1.set_xlabel("Coordinate Index (Lane)")
    ax1.set_ylabel("Stage Index")
    ax1.set_xticks(range(cfg.dim))
    fig.colorbar(im, ax=ax1, label="Field Value in F_65537")

    ax2 = fig.add_subplot(gs[2])
    indices = list(range(cfg.dim))
    width = 0.2
    ax2.bar([i - 1.5 * width for i in indices], x_in, width=width, label="Input x", color="#64748B")
    ax2.bar([i - 0.5 * width for i in indices], after_scalar, width=width, label="Scalar S(x)", color="#D97706")
    ax2.bar([i + 0.5 * width for i in indices], y_out, width=width, label="Forward Output y", color="#0F4C81")
    ax2.bar([i + 1.5 * width for i in indices], x_rec, width=width, label="Recovered x_hat", color="#15803D")
    ax2.set_title("Before / After Forward & Inverse Comparison", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xticks(indices)
    ax2.set_ylim(0, FIELD_MODULUS)
    ax2.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def visualize_transformation(
    *,
    vector: Sequence[int] | str | None = None,
    config_name: str | None = None,
    block_size: int | None = None,
    transform_type: str = "full",
    rounds: int | None = None,
    seed: int = 42,
    preset: str = "impulse",
    alphas: Sequence[int] | None = None,
    betas: Sequence[int] | None = None,
    engine: str = "auto",
    output_dir: str | Path | None = None,
) -> VisualizationResult:
    """Validate inputs, run QH4 transform trace, analyze element permutations, and render visuals."""
    mode = validate_transform_type(transform_type)
    s = validate_seed(seed)

    if not isinstance(engine, str) or engine not in VALID_ENGINES:
        ctx = {"field": "engine", "received": repr(engine), "valid": list(VALID_ENGINES)}
        msg = f"Invalid rendering engine {engine!r}. Valid options: {', '.join(VALID_ENGINES)}."
        _record_validation_failure("INVALID_VISUALIZATION_ENGINE", msg, ctx)
        raise QH4ValidationError("INVALID_VISUALIZATION_ENGINE", msg, ctx)

    # Parse/inspect vector first if explicit vector supplied without config_name
    parsed_vector: list[int] | None = None
    if isinstance(vector, str):
        parsed_vector = parse_vector_string(vector, expected_dim=None, name="vector")
    elif vector is not None:
        parsed_vector = validate_vector(vector, expected_dim=None, name="vector")

    cfg = resolve_block_layout(
        config_name=config_name,
        block_size=block_size,
        vector_len=len(parsed_vector) if parsed_vector is not None else None,
    )

    if parsed_vector is not None:
        xv = validate_vector(parsed_vector, expected_dim=cfg.dim, name="vector")
    else:
        if preset == "impulse":
            xv = [1 if i == 0 else 0 for i in range(cfg.dim)]
        elif preset == "ramp":
            xv = [((i + 1) * 1973) % FIELD_MODULUS for i in range(cfg.dim)]
        elif preset == "random":
            if cfg.name in BLOCK_CONFIGS:
                xv = generate_random_vectors(cfg.name, count=1, seed=s)[0]
            else:
                # Deterministic fallback for custom block layouts
                xv = [
                    int.from_bytes(hashlib.sha256(f"QH4-CUST|{cfg.name}|{s}|{i}".encode("utf-8")).digest()[:4], "big")
                    % FIELD_MODULUS
                    for i in range(cfg.dim)
                ]
        else:
            ctx = {"field": "preset", "received": repr(preset), "valid": ["impulse", "ramp", "random"]}
            msg = f"Invalid vector preset {preset!r}. Expected 'impulse', 'ramp', or 'random'."
            _record_validation_failure("INVALID_PRESET", msg, ctx)
            raise QH4ValidationError("INVALID_PRESET", msg, ctx)

    input_copy = list(xv)
    n_rounds = validate_rounds(cfg.default_rounds if rounds is None else rounds)

    if alphas is not None or betas is not None:
        if alphas is None or betas is None:
            ctx = {"field": "coefficients"}
            msg = "Both 'alphas' and 'betas' must be supplied together when overriding coefficients."
            _record_validation_failure("INCOMPLETE_COEFFICIENTS", msg, ctx)
            raise QH4ValidationError("INCOMPLETE_COEFFICIENTS", msg, ctx)
        av, bv = validate_coefficients(alphas, betas, expected_dim=cfg.dim)
    else:
        av, bv = _derive_custom_layout_coefficients(cfg, seed=s)

    stages, after_scalar, y_out, after_inv_mix, recovered = _trace_with_mode(
        xv, av, bv, rounds=n_rounds, transform_type=mode
    )

    ranks_in = _value_ranks(xv)
    ranks_scalar = _value_ranks(after_scalar)
    ranks_out = _value_ranks(y_out)

    permutations: list[FieldElementPermutation] = []
    for idx in range(cfg.dim):
        permutations.append(
            FieldElementPermutation(
                index=idx,
                block_index=idx // cfg.block_width,
                lane_index=idx % cfg.block_width,
                alpha=av[idx],
                beta=bv[idx],
                input_value=xv[idx],
                after_scalar=after_scalar[idx],
                forward_output=y_out[idx],
                after_inverse_mixing=after_inv_mix[idx],
                recovered_value=recovered[idx],
                scalar_delta_mod_q=(after_scalar[idx] - xv[idx]) % FIELD_MODULUS,
                mixing_delta_mod_q=(y_out[idx] - after_scalar[idx]) % FIELD_MODULUS,
                total_delta_mod_q=(y_out[idx] - xv[idx]) % FIELD_MODULUS,
                input_rank=ranks_in[idx],
                scalar_rank=ranks_scalar[idx],
                output_rank=ranks_out[idx],
                recovered_exact=recovered[idx] == xv[idx],
            )
        )

    dot_source = generate_graphviz_dot(cfg, n_rounds, mode, av, bv, stages)
    svg_source = render_visualization_svg(cfg, n_rounds, mode, av, bv, stages, permutations)
    ascii_summary = render_ascii_summary(cfg, mode, n_rounds, permutations)

    optional_backends = {"matplotlib": False, "graphviz": False}
    try:
        import matplotlib  # type: ignore  # noqa: F401

        optional_backends["matplotlib"] = True
    except ImportError:
        pass
    try:
        import graphviz  # type: ignore  # noqa: F401

        optional_backends["graphviz"] = True
    except ImportError:
        pass

    generated_files: list[str] = []
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stem = f"qh4_{cfg.name}_{mode}_seed{s}"

        if engine == "matplotlib":
            png_file = out_path / f"{stem}.png"
            render_with_matplotlib(png_file, cfg, n_rounds, mode, av, stages, permutations)
            generated_files.append(str(png_file))
        elif engine == "graphviz":
            import graphviz  # type: ignore

            rendered = graphviz.Source(dot_source).render(
                filename=str(out_path / f"{stem}_graph"), format="svg", cleanup=True
            )
            generated_files.append(str(rendered))
        elif engine == "svg":
            svg_file = out_path / f"{stem}.svg"
            svg_file.write_text(svg_source, encoding="utf-8")
            generated_files.append(str(svg_file))
        elif engine == "ascii":
            txt_file = out_path / f"{stem}.txt"
            txt_file.write_text(ascii_summary + "\n", encoding="utf-8")
            generated_files.append(str(txt_file))
        else:
            # auto: always write portable .svg, .dot, and .txt; also write .png if matplotlib is available
            if optional_backends["matplotlib"]:
                png_file = out_path / f"{stem}.png"
                render_with_matplotlib(png_file, cfg, n_rounds, mode, av, stages, permutations)
                generated_files.append(str(png_file))
            svg_file = out_path / f"{stem}.svg"
            svg_file.write_text(svg_source, encoding="utf-8")
            generated_files.append(str(svg_file))

            dot_file = out_path / f"{stem}.dot"
            dot_file.write_text(dot_source, encoding="utf-8")
            generated_files.append(str(dot_file))

            txt_file = out_path / f"{stem}.txt"
            txt_file.write_text(ascii_summary + "\n", encoding="utf-8")
            generated_files.append(str(txt_file))

    return VisualizationResult(
        config_name=cfg.name,
        blocks=cfg.blocks,
        dim=cfg.dim,
        block_width=cfg.block_width,
        transform_type=mode,
        rounds=n_rounds,
        seed=s,
        input_vector=input_copy,
        after_scalar_vector=after_scalar,
        forward_output_vector=y_out,
        after_inverse_mixing_vector=after_inv_mix,
        recovered_vector=recovered,
        exact_roundtrip_match=recovered == input_copy,
        input_unmutated=xv == input_copy,
        input_sha256=_sha256_vec(input_copy),
        output_sha256=_sha256_vec(y_out),
        reconstructed_sha256=_sha256_vec(recovered),
        alphas=av,
        betas=bv,
        stages=[asdict(st) for st in stages],
        element_permutations=[p.to_dict() for p in permutations],
        ascii_summary=ascii_summary,
        dot_source=dot_source,
        svg_source=svg_source,
        generated_files=generated_files,
        optional_backends=optional_backends,
    )


def format_validation_failure_report(limit: int = 20) -> str:
    """Format recorded validation failures as structured JSON for diagnostics."""
    failures = get_validation_failures(limit=limit)
    return json.dumps({"failure_count": len(failures), "failures": failures}, indent=2)
