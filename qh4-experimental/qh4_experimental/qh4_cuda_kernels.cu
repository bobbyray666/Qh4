// SPDX-License-Identifier: Apache-2.0
// Native NVIDIA CUDA C++ kernels for QH4-FieldPermutation v0.2.0 over F_{65,537}.
// Uses signed 64-bit integers (long long) because (65536)^2 = 4,294,836,225 > INT32_MAX.
// Compatible with both nvcc / PyTorch C++ CUDA extensions and NVRTC runtime compilation.

#ifndef QH4_CUDA_KERNELS_INCLUDED
#define QH4_CUDA_KERNELS_INCLUDED

extern "C" {

__device__ __forceinline__ long long qh4_mod_pos(long long v, long long q) {
  long long r = v % q;
  return r < 0 ? r + q : r;
}

__device__ __forceinline__ long long qh4_mod_pow(long long base, long long exp, long long q) {
  long long res = 1;
  long long cur = qh4_mod_pos(base, q);
  while (exp > 0) {
    if (exp & 1LL) {
      res = (res * cur) % q;
    }
    cur = (cur * cur) % q;
    exp >>= 1;
  }
  return res;
}

__device__ __forceinline__ long long qh4_mod_pow5(long long x, long long q) {
  long long x1 = qh4_mod_pos(x, q);
  long long x2 = (x1 * x1) % q;
  long long x4 = (x2 * x2) % q;
  return (x4 * x1) % q;
}

__global__ void qh4_arith_add_kernel(
    const long long* __restrict__ a,
    const long long* __restrict__ b,
    long long* __restrict__ out,
    int n_elements,
    long long q
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    out[idx] = qh4_mod_pos(a[idx] + b[idx], q);
  }
}

__global__ void qh4_arith_mul_kernel(
    const long long* __restrict__ a,
    const long long* __restrict__ b,
    long long* __restrict__ out,
    int n_elements,
    long long q
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    out[idx] = (qh4_mod_pos(a[idx], q) * qh4_mod_pos(b[idx], q)) % q;
  }
}

__global__ void qh4_arith_pow5_kernel(
    const long long* __restrict__ x,
    long long* __restrict__ out,
    int n_elements,
    long long q
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    out[idx] = qh4_mod_pow5(x[idx], q);
  }
}

__global__ void qh4_arith_pow_e_kernel(
    const long long* __restrict__ y,
    long long* __restrict__ out,
    int n_elements,
    long long exp_val,
    long long q
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_elements) {
    out[idx] = qh4_mod_pow(y[idx], exp_val, q);
  }
}

__global__ void qh4_transform_forward_kernel(
    const long long* __restrict__ x_in,
    const long long* __restrict__ alphas,
    const long long* __restrict__ betas,
    long long* __restrict__ out,
    int batch_size,
    int dim,
    int rounds,
    long long q
) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch_size) return;

  long long* row_out = out + ((long long)row * dim);
  const long long* row_in = x_in + ((long long)row * dim);

  // Layer 1: Scalar permutation S(x_i) = (alpha_i * x_i^5 + beta_i) mod q
  for (int i = 0; i < dim; ++i) {
    long long x5 = qh4_mod_pow5(row_in[i], q);
    row_out[i] = (alphas[i] * x5 + betas[i]) % q;
  }

  // Layer 2: Reversible two-sweep mixing L^rounds
  for (int r = 0; r < rounds; ++r) {
    for (int i = 0; i < dim - 1; ++i) {
      row_out[i] = (row_out[i] + row_out[i + 1]) % q;
    }
    for (int i = dim - 1; i >= 1; --i) {
      row_out[i] = (row_out[i] + row_out[i - 1]) % q;
    }
  }
}

__global__ void qh4_transform_inverse_kernel(
    const long long* __restrict__ y_in,
    const long long* __restrict__ alpha_invs,
    const long long* __restrict__ betas,
    long long* __restrict__ out,
    int batch_size,
    int dim,
    int rounds,
    long long inv_exp,
    long long q
) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch_size) return;

  long long* row_out = out + ((long long)row * dim);
  const long long* row_in = y_in + ((long long)row * dim);

  for (int i = 0; i < dim; ++i) {
    row_out[i] = row_in[i];
  }

  // Inverse Layer 2: L^{-rounds}
  for (int r = 0; r < rounds; ++r) {
    for (int i = 1; i < dim; ++i) {
      row_out[i] = qh4_mod_pos(row_out[i] - row_out[i - 1], q);
    }
    for (int i = dim - 2; i >= 0; --i) {
      row_out[i] = qh4_mod_pos(row_out[i] - row_out[i + 1], q);
    }
  }

  // Inverse Layer 1: S^{-1}(y_i) = (alpha_i^{-1} * (y_i - beta_i))^e mod q
  for (int i = 0; i < dim; ++i) {
    long long diff = qh4_mod_pos(row_out[i] - betas[i], q);
    long long inner = (alpha_invs[i] * diff) % q;
    row_out[i] = qh4_mod_pow(inner, inv_exp, q);
  }
}

}  // extern "C"

#endif
