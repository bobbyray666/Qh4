# SPDX-License-Identifier: Apache-2.0
"""Compiler and execution tests for NVIDIA CUDA C++ and Triton kernels."""

from __future__ import annotations

import unittest

from qh4_experimental.configs import BLOCK_CONFIGS, derive_coefficients, generate_random_vectors
from qh4_experimental.cuda_runner import (
    compile_cuda_cpp_to_ptx,
    probe_cuda_toolchain,
    run_cuda_cpp_all_configs,
)
from qh4_experimental.triton_kernels import (
    compile_triton_kernels_aot,
    is_triton_interpret_enabled,
    probe_triton_environment,
    validate_triton_against_cpu,
    validate_triton_arithmetic_kernels,
)


class TestCudaAndTritonToolchains(unittest.TestCase):
    def test_cuda_cpp_nvrtc_ptx_compilation(self) -> None:
        probe = probe_cuda_toolchain()
        if not probe["libnvrtc"]:
            self.skipTest("libnvrtc.so not installed in current Python environment.")
        res = compile_cuda_cpp_to_ptx(target_sm=80)
        self.assertEqual(res["status"], "compiled")
        self.assertGreater(res["ptx_bytes"], 1000)
        if probe["ptxas"]:
            self.assertGreater(res["cubin_bytes"], 1000)
            self.assertEqual(len(res["cubin_sha256"]), 64)

    def test_triton_aot_ptx_cubin_compilation(self) -> None:
        env = probe_triton_environment()
        if not env["triton_installed"]:
            self.skipTest("triton package not installed in current Python environment.")
        if is_triton_interpret_enabled():
            self.skipTest("AOT compilation test requires TRITON_INTERPRET=0.")
        res = compile_triton_kernels_aot(target_sm=80, dim=16, rounds=16)
        self.assertEqual(res["status"], "compiled")
        self.assertEqual(res["kernel_count"], 8)
        for k in res["kernels"]:
            self.assertGreater(k["ptx_bytes"], 500)
            self.assertGreater(k["cubin_bytes"], 500)

    def test_triton_interpreter_differential_parity(self) -> None:
        env = probe_triton_environment()
        if not env["triton_installed"] or not env["torch_installed"]:
            self.skipTest("torch/triton not installed.")
        if not is_triton_interpret_enabled():
            self.skipTest("Run with TRITON_INTERPRET=1 to execute Triton interpreter parity tests.")
        arith = validate_triton_arithmetic_kernels(seed=1234, allow_interpret=True)
        self.assertEqual(arith["status"], "passed")
        for name, cfg in BLOCK_CONFIGS.items():
            a, b = derive_coefficients(name, seed=2026)
            batch = generate_random_vectors(name, count=4, seed=2026)
            res = validate_triton_against_cpu(batch, a, b, rounds=cfg.default_rounds, allow_interpret=True)
            self.assertEqual(res["status"], "passed")

    def test_physical_nvidia_gpu_differential_parity(self) -> None:
        probe = probe_cuda_toolchain()
        env = probe_triton_environment()
        if not probe["cuda_available"]:
            self.skipTest("No physical NVIDIA CUDA GPU attached on this host.")
        if env["triton_installed"] and not is_triton_interpret_enabled():
            arith = validate_triton_arithmetic_kernels(seed=1234, allow_interpret=False)
            self.assertEqual(arith["status"], "passed")
            for name, cfg in BLOCK_CONFIGS.items():
                a, b = derive_coefficients(name, seed=2026)
                batch = generate_random_vectors(name, count=32, seed=2026)
                res = validate_triton_against_cpu(batch, a, b, rounds=cfg.default_rounds, allow_interpret=False)
                self.assertEqual(res["status"], "passed")
        cuda_res = run_cuda_cpp_all_configs(batch_size=32, seed=2026)
        self.assertEqual(cuda_res["status"], "passed")


if __name__ == "__main__":
    unittest.main()
