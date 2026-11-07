"""Unit tests for sglang.srt.utils.triton_utils — no server, no GPU.

SGLang overrides ``triton.next_power_of_2`` process-wide. The override must keep
SGLang's host-side results and stay callable from ``@triton.jit`` code, or other
libraries' kernels in the same process stop compiling (#48460).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import triton
import triton.language as tl

from sglang.srt.utils.triton_utils import next_power_of_2
from sglang.test.test_utils import CustomTestCase


@triton.jit
def _copy_with_padded_block(x_ptr, out_ptr, N: tl.constexpr):
    # Same pattern as a kernel outside SGLang that relies on upstream Triton.
    BLOCK: tl.constexpr = triton.next_power_of_2(N)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


class TestNextPowerOf2(CustomTestCase):
    def test_helper_values(self):
        cases = {-3: 1, 0: 1, 1: 1, 2: 2, 3: 4, 1024: 1024, 1025: 2048}
        for n, expected in cases.items():
            self.assertEqual(next_power_of_2(n), expected, n)

    def test_triton_override_matches_helper_on_host(self):
        for n in range(-4, 4097):
            self.assertEqual(triton.next_power_of_2(n), next_power_of_2(n), n)

    def test_triton_override_unwraps_constexpr_on_host(self):
        if not hasattr(triton, "constexpr_function"):
            self.skipTest("Triton < 3.5 has no constexpr_function")
        self.assertEqual(triton.next_power_of_2(tl.constexpr(100)), 128)
        self.assertEqual(triton.next_power_of_2(n=5), 8)

    def test_triton_override_is_callable_from_jit_kernels(self):
        if not hasattr(triton, "constexpr_function"):
            self.skipTest("Triton < 3.5: next_power_of_2 is host-only upstream too")
        from triton.backends import backends
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        if "nvidia" not in backends:
            self.skipTest("Triton build has no NVIDIA backend to compile against")

        src = ASTSource(
            fn=_copy_with_padded_block,
            signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "N": "constexpr"},
            constexprs={"N": 100},
        )
        # An explicit target compiles without a GPU.
        kernel = triton.compile(src, target=GPUTarget("cuda", 90, 32))
        self.assertIn("tt.make_range {end = 128 : i32", kernel.asm["ttir"])


if __name__ == "__main__":
    unittest.main()
