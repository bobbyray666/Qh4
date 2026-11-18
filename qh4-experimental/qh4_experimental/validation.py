# SPDX-License-Identifier: Apache-2.0
"""Comprehensive input and internal state validation with structured failure logging."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

FIELD_MODULUS: int = 65537  # Fermat prime F_4 = 2^16 + 1
FORWARD_EXPONENT: int = 5
INVERSE_EXPONENT: int = 52429  # 5 * 52429 = 262145 = 4 * 65536 + 1 == 1 (mod 65536)

logger = logging.getLogger("qh4.validation")
if not logger.handlers:
    _handler = logging.NullHandler()
    logger.addHandler(_handler)


@dataclass(frozen=True)
class ValidationFailureRecord:
    """Structured diagnostic record emitted whenever input validation fails."""

    timestamp_utc: str
    code: str
    field: str
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_LOG_LOCK = threading.Lock()
_VALIDATION_FAILURE_LOG: deque[ValidationFailureRecord] = deque(maxlen=256)
_LOG_FILE_PATH: Path | None = None


def configure_validation_log_file(path: str | Path | None) -> Path | None:
    """Configure an optional JSONL file sink for validation failure reporting."""
    global _LOG_FILE_PATH
    with _LOG_LOCK:
        if path is None:
            _LOG_FILE_PATH = None
            return None
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE_PATH = resolved
        return resolved


def get_validation_failures(limit: int | None = None) -> list[dict[str, Any]]:
    """Return logged validation failure records (newest last)."""
    with _LOG_LOCK:
        records = [rec.to_dict() for rec in _VALIDATION_FAILURE_LOG]
    if limit is not None and limit >= 0:
        return records[-limit:] if limit > 0 else []
    return records


def clear_validation_failures() -> None:
    """Clear the in-memory validation failure audit buffer."""
    with _LOG_LOCK:
        _VALIDATION_FAILURE_LOG.clear()


def _record_validation_failure(code: str, message: str, context: dict[str, Any] | None = None) -> None:
    ctx = dict(context or {})
    field = str(ctx.get("field", "unknown"))
    record = ValidationFailureRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        code=code,
        field=field,
        message=message,
        context=ctx,
    )
    with _LOG_LOCK:
        _VALIDATION_FAILURE_LOG.append(record)
        log_path = _LOG_FILE_PATH
    logger.warning("QH4 validation failure [%s] field=%s: %s", code, field, message)
    if log_path is not None:
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict()) + "\n")
        except OSError as exc:
            logger.error("Failed to write validation failure log file %s: %s", log_path, exc)


def _fail(code: str, message: str, context: dict[str, Any] | None = None) -> QH4ValidationError:
    _record_validation_failure(code, message, context)
    return QH4ValidationError(code, message, context)


class QH4Error(ValueError):
    """Base exception for QH4 validation and runtime invariant violations."""

    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.context = context or {}
        super().__init__(f"[{code}] {message}")


class QH4ValidationError(QH4Error):
    """Raised when user-supplied parameters or vectors fail validation."""


class QH4InternalStateError(RuntimeError):
    """Raised when an internal mathematical invariant or buffer contract fails."""

    def __init__(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.context = context or {}
        super().__init__(f"[{code}] {message}")


def validate_strict_int(value: Any, name: str = "value") -> int:
    """Validate that value is a genuine integer (rejecting bool, float, NaN, str, None)."""
    if isinstance(value, bool):
        raise _fail(
            "INVALID_TYPE_BOOL",
            f"Parameter '{name}' must be an integer, received boolean ({value!r}).",
            {"field": name, "received_type": "bool"},
        )
    if hasattr(value, "dtype") and hasattr(value, "item"):
        # Support NumPy integer scalars while rejecting NumPy float/bool scalars
        kind = getattr(value.dtype, "kind", "")
        if kind in ("i", "u"):
            return int(value.item())
        raise _fail(
            "INVALID_NUMPY_SCALAR",
            f"Parameter '{name}' must have integer dtype, received dtype={value.dtype!r}.",
            {"field": name, "dtype": str(value.dtype)},
        )
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _fail(
                "NON_FINITE_NUMBER",
                f"Parameter '{name}' must be a finite integer, received {value!r}.",
                {"field": name, "received": str(value)},
            )
        raise _fail(
            "INVALID_TYPE_FLOAT",
            f"Parameter '{name}' must be an integer, received float ({value!r}).",
            {"field": name, "received_type": "float"},
        )
    if not isinstance(value, int):
        raise _fail(
            "INVALID_TYPE",
            f"Parameter '{name}' must be an integer, received {type(value).__name__}.",
            {"field": name, "received_type": type(value).__name__},
        )
    return int(value)


def validate_field_element(value: Any, name: str = "x", q: int = FIELD_MODULUS) -> int:
    """Validate an element of F_q: integer in [0, q - 1]."""
    ival = validate_strict_int(value, name=name)
    if ival < 0 or ival >= q:
        raise _fail(
            "FIELD_RANGE_ERROR",
            f"Field element '{name}'={ival} is out of range [0, {q - 1}] for F_{{{q}}}.",
            {"field": name, "value": ival, "min": 0, "max": q - 1},
        )
    return ival


def validate_alpha(value: Any, name: str = "alpha", q: int = FIELD_MODULUS) -> int:
    """Validate multiplicative coefficient alpha in F_q^*: integer in [1, q - 1]."""
    ival = validate_field_element(value, name=name, q=q)
    if ival == 0:
        raise _fail(
            "ALPHA_ZERO_NON_INVERTIBLE",
            f"Coefficient '{name}' cannot be 0 because 0 has no multiplicative inverse in F_{{{q}}}.",
            {"field": name, "value": 0},
        )
    return ival


def validate_beta(value: Any, name: str = "beta", q: int = FIELD_MODULUS) -> int:
    """Validate additive coefficient beta in F_q: integer in [0, q - 1]."""
    return validate_field_element(value, name=name, q=q)


def validate_vector(
    vec: Any,
    expected_dim: int | None = None,
    name: str = "vector",
    q: int = FIELD_MODULUS,
) -> list[int]:
    """Validate a 1D vector of field elements in F_q without mutating the input."""
    if vec is None or isinstance(vec, (str, bytes, bytearray, dict)):
        raise _fail(
            "INVALID_VECTOR_CONTAINER",
            f"'{name}' must be a sequence or 1D array of integers, received {type(vec).__name__}.",
            {"field": name, "received_type": type(vec).__name__},
        )

    # Reject multi-dimensional NumPy arrays explicitly
    if hasattr(vec, "ndim"):
        ndim = getattr(vec, "ndim")
        if ndim != 1:
            raise _fail(
                "INVALID_VECTOR_RANK",
                f"'{name}' must be 1-dimensional, received rank {ndim} with shape {getattr(vec, 'shape', None)}.",
                {"field": name, "ndim": int(ndim)},
            )

    if not isinstance(vec, Iterable):
        raise _fail(
            "NON_ITERABLE_VECTOR",
            f"'{name}' is not iterable (received {type(vec).__name__}).",
            {"field": name, "received_type": type(vec).__name__},
        )

    items = list(vec)
    if len(items) < 2:
        raise _fail(
            "VECTOR_TOO_SHORT",
            f"'{name}' length must be at least 2 for two-sweep mixing, received length {len(items)}.",
            {"field": name, "length": len(items)},
        )

    if expected_dim is not None and len(items) != expected_dim:
        raise _fail(
            "DIMENSION_MISMATCH",
            f"'{name}' expected dimension {expected_dim}, received {len(items)}.",
            {"field": name, "expected": expected_dim, "actual": len(items)},
        )

    validated: list[int] = []
    for idx, elem in enumerate(items):
        validated.append(validate_field_element(elem, name=f"{name}[{idx}]", q=q))
    return validated


def validate_coefficients(
    alphas: Any,
    betas: Any,
    expected_dim: int,
    q: int = FIELD_MODULUS,
) -> tuple[list[int], list[int]]:
    """Validate parallel alpha and beta coefficient sequences."""
    if alphas is None or isinstance(alphas, (str, bytes, bytearray, dict)) or not isinstance(alphas, Iterable):
        raise _fail(
            "INVALID_ALPHA_CONTAINER",
            f"'alphas' must be a 1D sequence of {expected_dim} integers, received {type(alphas).__name__}.",
            {"field": "alphas", "received_type": type(alphas).__name__},
        )
    if betas is None or isinstance(betas, (str, bytes, bytearray, dict)) or not isinstance(betas, Iterable):
        raise _fail(
            "INVALID_BETA_CONTAINER",
            f"'betas' must be a 1D sequence of {expected_dim} integers, received {type(betas).__name__}.",
            {"field": "betas", "received_type": type(betas).__name__},
        )
    alpha_list = list(alphas)
    beta_list = list(betas)
    if len(alpha_list) != expected_dim:
        raise _fail(
            "ALPHA_LENGTH_MISMATCH",
            f"Expected {expected_dim} alpha coefficients, received {len(alpha_list)}.",
            {"field": "alphas", "expected": expected_dim, "actual": len(alpha_list)},
        )
    if len(beta_list) != expected_dim:
        raise _fail(
            "BETA_LENGTH_MISMATCH",
            f"Expected {expected_dim} beta coefficients, received {len(beta_list)}.",
            {"field": "betas", "expected": expected_dim, "actual": len(beta_list)},
        )
    val_alphas = [validate_alpha(a, name=f"alpha[{i}]", q=q) for i, a in enumerate(alpha_list)]
    val_betas = [validate_beta(b, name=f"beta[{i}]", q=q) for i, b in enumerate(beta_list)]
    return val_alphas, val_betas


def validate_rounds(rounds: Any, max_rounds: int = 1024) -> int:
    """Validate mixing round count n >= 1."""
    r = validate_strict_int(rounds, name="rounds")
    if r < 1 or r > max_rounds:
        raise _fail(
            "INVALID_ROUND_COUNT",
            f"Round count 'rounds'={r} must be within [1, {max_rounds}].",
            {"field": "rounds", "value": r, "min": 1, "max": max_rounds},
        )
    return r


def validate_seed(seed: Any) -> int:
    """Validate deterministic non-negative 64-bit seed integer."""
    s = validate_strict_int(seed, name="seed")
    if s < 0 or s >= (1 << 64):
        raise _fail(
            "INVALID_SEED",
            f"Seed must be in [0, 2^64 - 1], received {s}.",
            {"field": "seed", "value": s},
        )
    return s


def validate_batch(
    batch: Any,
    expected_dim: int,
    name: str = "batch",
    q: int = FIELD_MODULUS,
) -> list[list[int]]:
    """Validate a 2D batch of vectors [batch_size, expected_dim]."""
    if batch is None or isinstance(batch, (str, bytes, bytearray, dict)) or not isinstance(batch, Iterable):
        raise _fail(
            "INVALID_BATCH_CONTAINER",
            f"'{name}' must be a non-empty sequence of vectors, received {type(batch).__name__}.",
            {"field": name, "received_type": type(batch).__name__},
        )
    rows = list(batch)
    if len(rows) == 0:
        raise _fail(
            "EMPTY_BATCH",
            f"'{name}' cannot be empty; provide at least 1 vector.",
            {"field": name},
        )
    return [validate_vector(row, expected_dim=expected_dim, name=f"{name}[{i}]", q=q) for i, row in enumerate(rows)]


def parse_vector_string(raw: Any, expected_dim: int | None = None, name: str = "vector", q: int = FIELD_MODULUS) -> list[int]:
    """Parse and validate a comma-, whitespace-, or JSON-formatted vector string."""
    if not isinstance(raw, str):
        raise _fail(
            "INVALID_VECTOR_TEXT",
            f"'{name}' text input must be a string, received {type(raw).__name__}.",
            {"field": name, "received_type": type(raw).__name__},
        )
    trimmed = raw.strip()
    if not trimmed:
        raise _fail(
            "EMPTY_VECTOR_INPUT",
            f"'{name}' string is empty. Provide integers in [0, {q - 1}].",
            {"field": name},
        )
    if trimmed.startswith("["):
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError as exc:
            raise _fail(
                "MALFORMED_JSON_VECTOR",
                f"JSON syntax error in '{name}': {exc.msg}.",
                {"field": name},
            ) from exc
        return validate_vector(parsed, expected_dim=expected_dim, name=name, q=q)

    tokens = [tok for tok in trimmed.replace(";", ",").replace(",", " ").split() if tok]
    parsed_vals: list[int] = []
    for idx, tok in enumerate(tokens):
        try:
            val = int(tok, 10)
        except ValueError as exc:
            raise _fail(
                "INVALID_TOKEN",
                f"Token {tok!r} at {name}[{idx}] is not a valid base-10 integer in [0, {q - 1}].",
                {"field": f"{name}[{idx}]", "token": tok, "index": idx},
            ) from exc
        parsed_vals.append(val)
    return validate_vector(parsed_vals, expected_dim=expected_dim, name=name, q=q)


def validate_inputs_report(
    *,
    vector: Any,
    expected_dim: int | None = None,
    alphas: Any = None,
    betas: Any = None,
    rounds: Any = None,
    seed: Any = None,
    q: int = FIELD_MODULUS,
) -> dict[str, Any]:
    """Validate a complete QH4 call bundle non-destructively and return a structured report."""
    errors: list[dict[str, Any]] = []
    validated_vec: list[int] | None = None
    validated_alphas: list[int] | None = None
    validated_betas: list[int] | None = None
    validated_rounds: int | None = None
    validated_seed: int | None = None

    try:
        validated_vec = validate_vector(vector, expected_dim=expected_dim, name="vector", q=q)
    except QH4ValidationError as exc:
        errors.append({"code": exc.code, "message": exc.message, "context": exc.context})

    dim_for_checks = expected_dim or (len(validated_vec) if validated_vec is not None else None)
    if (alphas is not None or betas is not None) and dim_for_checks is not None:
        try:
            validated_alphas, validated_betas = validate_coefficients(alphas, betas, dim_for_checks, q=q)
        except QH4ValidationError as exc:
            errors.append({"code": exc.code, "message": exc.message, "context": exc.context})

    if rounds is not None:
        try:
            validated_rounds = validate_rounds(rounds)
        except QH4ValidationError as exc:
            errors.append({"code": exc.code, "message": exc.message, "context": exc.context})

    if seed is not None:
        try:
            validated_seed = validate_seed(seed)
        except QH4ValidationError as exc:
            errors.append({"code": exc.code, "message": exc.message, "context": exc.context})

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "validated": {
            "vector": validated_vec,
            "alphas": validated_alphas,
            "betas": validated_betas,
            "rounds": validated_rounds,
            "seed": validated_seed,
        },
    }
