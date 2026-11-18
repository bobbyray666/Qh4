# QH4-Pelagos-Adapter (`qh4-experimental` v0.2.0)

Independent experimental implementation of **QH4-FieldPermutation v0.2.0** over $\mathbb{F}_{65,537}$.

## Mathematical Specification

- **Prime Field**: $q = 65,537 = 2^{16} + 1$ ($\mathbb{F}_{65,537}$)
- **Forward Exponent**: $d = 5$, satisfying $\gcd(5, 65,536) = 1$
- **Inverse Exponent**: $e = 52,429$, satisfying $5 \cdot e \equiv 1 \pmod{65,536}$
- **Scalar Layer $S$**: For $\alpha_i \in \{1, \dots, q-1\}$ and $\beta_i \in \{0, \dots, q-1\}$:
  - Forward: $S_i(x_i) = (\alpha_i \cdot x_i^5 + \beta_i) \bmod q$
  - Inverse: $S_i^{-1}(y_i) = \big(\alpha_i^{-1} \cdot (y_i - \beta_i \bmod q)\big)^e \bmod q$
- **Reversible Mixing Layer $L$** on $\mathbf{x} = (x_0, \dots, x_{n-1})$:
  - Forward Sweep 1: $x_i \leftarrow (x_i + x_{i+1}) \bmod q$ for $i = 0, \dots, n-2$
  - Forward Sweep 2: $x_i \leftarrow (x_i + x_{i-1}) \bmod q$ for $i = n-1, \dots, 1$
  - Inverse Sweep 2 (reverse order): $x_i \leftarrow (x_i - x_{i-1}) \bmod q$ for $i = 1, \dots, n-1$
  - Inverse Sweep 1 (reverse order): $x_i \leftarrow (x_i - x_{i+1}) \bmod q$ for $i = n-2, \dots, 0$
- **Complete Transformation**: $T_n = L^n \circ S$, $T_n^{-1} = S^{-1} \circ L^{-n}$

## Supported Block Configurations

`8B-D8`, `8B-D16`, `9B-D9`, `9B-D18`, `16B-D16`, `16B-D32`, `18B-D18`, `18B-D36`

## Modules

- `qh4_experimental/validation.py`: Strict boundary, type, range, and invariant validation.
- `qh4_experimental/cpu_reference.py`: Pure Python/NumPy CPU ground-truth reference baseline.
- `qh4_experimental/mixing.py`: Two-sweep mixing layer $L$, $L^{-1}$, and stage-by-stage trace capture.
- `qh4_experimental/configs.py`: All 8 block configurations and deterministic coefficient assignment.
- `qh4_experimental/triton_kernels.py`: Separate Triton arithmetic and transform GPU kernels with explicit compile/autotune failure surfacing.
- `qh4_experimental/pelagos_orchestrator.py`: Multi-GPU worker pool with pre-import `CUDA_VISIBLE_DEVICES` isolation, persistent workers, static quotient+remainder sharding, and contiguous reassembly.
- `qh4_experimental/benchmark.py`: Structured telemetry harness (cold-start, warm kernel, warm batch, end-to-end, metadata & checksums).
