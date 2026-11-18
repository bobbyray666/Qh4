# SPDX-License-Identifier: Apache-2.0
"""QH4-Pelagos-Adapter / qh4-experimental v0.2.0."""

from .validation import (
    FIELD_MODULUS,
    FORWARD_EXPONENT,
    INVERSE_EXPONENT,
    QH4ValidationError,
    QH4InternalStateError,
    ValidationFailureRecord,
    clear_validation_failures,
    configure_validation_log_file,
    get_validation_failures,
    parse_vector_string,
    validate_inputs_report,
)
from .configs import BLOCK_CONFIGS, BlockConfig, get_config, derive_coefficients
from .cpu_reference import (
    field_add,
    field_sub,
    field_mul,
    field_pow,
    scalar_forward_element,
    scalar_inverse_element,
    scalar_forward_vector,
    scalar_inverse_vector,
    mod_inverse,
    vector_sha256,
)
from .mixing import (
    mixing_forward_round,
    mixing_inverse_round,
    transform_forward,
    transform_inverse,
    transform_with_trace,
)
from .visualization import (
    FieldElementPermutation,
    VisualizationResult,
    format_validation_failure_report,
    generate_graphviz_dot,
    render_ascii_summary,
    render_visualization_svg,
    visualize_transformation,
)

__version__ = "0.2.0"

__all__ = [
    "FIELD_MODULUS",
    "FORWARD_EXPONENT",
    "INVERSE_EXPONENT",
    "QH4ValidationError",
    "QH4InternalStateError",
    "ValidationFailureRecord",
    "clear_validation_failures",
    "configure_validation_log_file",
    "get_validation_failures",
    "parse_vector_string",
    "validate_inputs_report",
    "BLOCK_CONFIGS",
    "BlockConfig",
    "get_config",
    "derive_coefficients",
    "field_add",
    "field_sub",
    "field_mul",
    "field_pow",
    "scalar_forward_element",
    "scalar_inverse_element",
    "scalar_forward_vector",
    "scalar_inverse_vector",
    "mod_inverse",
    "vector_sha256",
    "mixing_forward_round",
    "mixing_inverse_round",
    "transform_forward",
    "transform_inverse",
    "transform_with_trace",
    "FieldElementPermutation",
    "VisualizationResult",
    "format_validation_failure_report",
    "generate_graphviz_dot",
    "render_ascii_summary",
    "render_visualization_svg",
    "visualize_transformation",
]
