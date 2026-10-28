# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import sys
import unittest

_MISSING = [
    name
    for name in ("jax", "torch", "torchax")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    print(f"{', '.join(_MISSING)} not found, skipping torchax_attention_test")
    sys.exit(77)

import jax
import numpy as np
import torch
import torchax

from iree.compiler import tools
from iree_torchax import torchax_attention

try:
    from iree import runtime as iree_runtime
except ImportError:
    iree_runtime = None

torchax_attention.install()

_B, _H, _L, _S, _E, _EV = 2, 3, 5, 7, 4, 6
_RNG = np.random.default_rng(3)
_Q = _RNG.standard_normal((_B, _H, _L, _E)).astype(np.float32)
_K = _RNG.standard_normal((_B, _H, _S, _E)).astype(np.float32)
_V = _RNG.standard_normal((_B, _H, _S, _EV)).astype(np.float32)
_BOOL_MASK = _RNG.random((_L, _S)) > 0.3
_BOOL_MASK[:, 0] = True
_FLOAT_MASK = _RNG.standard_normal((_B, _H, _L, _S)).astype(np.float32)
# A fully masked row attends to nothing; torch returns zeros there.
_EMPTY_ROW_BOOL_MASK = _BOOL_MASK.copy()
_EMPTY_ROW_BOOL_MASK[2, :] = False
_EMPTY_ROW_FLOAT_MASK = _FLOAT_MASK.copy()
_EMPTY_ROW_FLOAT_MASK[1, 2, 3, :] = -np.inf

_sdpa = torch.nn.functional.scaled_dot_product_attention
_CASES = {
    "plain": (lambda q, k, v: _sdpa(q, k, v), (_Q, _K, _V)),
    "scale": (lambda q, k, v: _sdpa(q, k, v, scale=0.3), (_Q, _K, _V)),
    "causal": (lambda q, k, v: _sdpa(q, k, v, is_causal=True), (_Q, _K, _V)),
    "bool_mask": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, _BOOL_MASK),
    ),
    "float_mask": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, _FLOAT_MASK),
    ),
    "bool_mask_empty_row": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, _EMPTY_ROW_BOOL_MASK),
    ),
    "float_mask_empty_row": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, _EMPTY_ROW_FLOAT_MASK),
    ),
    # Masks broadcast over heads, or over batches and heads, as Hugging Face
    # builds them.
    "float_mask_head_broadcast": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, np.ascontiguousarray(_FLOAT_MASK[:, :1])),
    ),
    "bool_mask_batch_head_broadcast": (
        lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m),
        (_Q, _K, _V, _BOOL_MASK.reshape(1, 1, _L, _S)),
    ),
}


class _Wrap(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _export(fn, args):
    weights, jax_fn = torchax.extract_jax(_Wrap(fn))
    jax_args = tuple(jax.numpy.asarray(a) for a in args)
    return jax.jit(jax_fn).lower(weights, jax_args).as_text(), weights, jax_fn


def _execute(module, args):
    vmfb = tools.compile_str(
        module,
        input_type="stablehlo",
        target_backends=["llvm-cpu"],
        extra_args=["--iree-llvmcpu-target-cpu=generic"],
    )
    config = iree_runtime.Config("local-task")
    context = iree_runtime.SystemContext(config=config)
    vm_module = iree_runtime.VmModule.copy_buffer(config.vm_instance, vmfb)
    context.add_vm_module(vm_module)
    return context.modules[vm_module.name].main(*args).to_host()


class TorchaxAttentionTest(unittest.TestCase):
    def _check(self, fn, args):
        with torchax.disable_temporarily():
            expected = fn(*(torch.from_numpy(a) for a in args)).numpy()
        module, weights, jax_fn = _export(fn, args)
        self.assertEqual(
            module.count('stablehlo.composite "torch.scaled_dot_product_attention"'),
            1,
        )
        with jax.disable_jit():
            decomposed = np.asarray(
                jax_fn(weights, tuple(map(jax.numpy.asarray, args)))
            )
        np.testing.assert_allclose(decomposed, expected, rtol=1e-5, atol=1e-6)

        flow = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic", "--compile-to=flow"],
        ).decode()
        self.assertNotIn("stablehlo.composite", flow)
        self.assertEqual(flow.count("iree_linalg_ext.attention"), 1)

        if iree_runtime is None:
            return
        result = _execute(module, args)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_grouped_query_attention_matches_torch(self):
        # Six query heads share two key and value heads; the attention op reads
        # each key and value head for its three query heads.
        q = _RNG.standard_normal((_B, 6, _L, _E)).astype(np.float32)
        k = _RNG.standard_normal((_B, 2, _S, _E)).astype(np.float32)
        v = _RNG.standard_normal((_B, 2, _S, _EV)).astype(np.float32)
        fn = lambda q, k, v: _sdpa(q, k, v, enable_gqa=True)
        with torchax.disable_temporarily():
            expected = fn(*(torch.from_numpy(a) for a in (q, k, v))).numpy()
        module, _, _ = _export(fn, (q, k, v))
        self.assertIn("enable_gqa = true", module)
        flow = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic", "--compile-to=flow"],
        ).decode()
        self.assertIn("iree_linalg_ext.attention", flow)
        # The key and value heads are not repeated to six.
        self.assertNotRegex(flow, r"tensor<2x6x7x[46]xf32>")
        if iree_runtime is None:
            return
        result = _execute(module, (q, k, v))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_causal_grouped_query_attention_spans_tiles(self):
        # Eight query heads share two key and value heads. 70 queries and 90
        # keys span several CPU tiles, whose score indices must give the query
        # and key positions, not the group.
        q = _RNG.standard_normal((1, 8, 70, _E)).astype(np.float32)
        k = _RNG.standard_normal((1, 2, 90, _E)).astype(np.float32)
        v = _RNG.standard_normal((1, 2, 90, _EV)).astype(np.float32)
        self._check(
            lambda q, k, v: _sdpa(q, k, v, is_causal=True, enable_gqa=True),
            (q, k, v),
        )

    def test_grouped_mask_hides_one_head_row(self):
        # A per-head mask hides one row of one query head; only that row is
        # zero.
        q = _RNG.standard_normal((_B, 6, _L, _E)).astype(np.float32)
        k = _RNG.standard_normal((_B, 2, _S, _E)).astype(np.float32)
        v = _RNG.standard_normal((_B, 2, _S, _EV)).astype(np.float32)
        mask = _RNG.random((_B, 6, _L, _S)) > 0.3
        mask[..., 0] = True
        mask[1, 4, 2, :] = False
        self._check(
            lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m, enable_gqa=True),
            (q, k, v, mask),
        )

    def test_symbolic_composite_lowers_to_attention(self):
        (n,) = jax.export.symbolic_shape("n")

        def fn(q, k, v):
            return torchax_attention._sdpa(q, k, v, is_causal=True, enable_gqa=False)

        spec = jax.ShapeDtypeStruct((1, 3, n, 64), np.float32)
        module = jax.export.export(jax.jit(fn))(spec, spec, spec).mlir_module()
        flow = tools.compile_str(
            module,
            input_type="stablehlo",
            target_backends=["llvm-cpu"],
            extra_args=["--iree-llvmcpu-target-cpu=generic", "--compile-to=flow"],
        ).decode()
        self.assertNotIn("stablehlo.composite", flow)
        self.assertEqual(flow.count("iree_linalg_ext.attention"), 1)

    def test_symbolic_composite_shares_the_module_symbols(self):
        # Operands of length 2n and a later division of n export together.
        (n,) = jax.export.symbolic_shape("n")

        def fn(q):
            doubled = jax.numpy.concatenate([q, q], axis=2)
            result = torchax_attention._sdpa(
                doubled, doubled, doubled, is_causal=False, enable_gqa=False
            )
            return result, jax.numpy.floor_divide(result.shape[2], 2)

        exported = jax.export.export(jax.jit(fn))(
            jax.ShapeDtypeStruct((1, 2, n, 4), np.float32)
        )
        self.assertIn(
            'stablehlo.composite "torch.scaled_dot_product_attention"',
            exported.mlir_module(),
        )

    def test_composite_exports_for_several_platforms(self):
        # JAX passes the platform index to the decomposition as a leading
        # operand, which IREE's composite handlers do not take.
        def fn(q):
            return torchax_attention._sdpa(q, q, q, is_causal=False, enable_gqa=False)

        module = jax.export.export(jax.jit(fn), platforms=("cpu", "cuda"))(
            jax.ShapeDtypeStruct((1, 2, 3, 4), np.float32)
        ).mlir_module()
        self.assertIn(
            'stablehlo.composite "torch.scaled_dot_product_attention"', module
        )
        self.assertRegex(
            module,
            r"func\.func private @torch\.scaled_dot_product_attention\("
            r"%arg0: tensor<i32> \{jax\.global_constant = \"_platform_index\"\}",
        )

    def test_dropout_keeps_torchax_lowering(self):
        module, _, _ = _export(
            lambda q, k, v: _sdpa(q, k, v, dropout_p=0.5), (_Q, _K, _V)
        )
        self.assertNotIn("torch.scaled_dot_product_attention", module)

    # Known limitation: the attention lowering gives NaN for an unmasked -inf row.
    @unittest.expectedFailure
    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_unmasked_negative_infinity_row_is_zero(self):
        # The producer zeroes a row whose scores are all -inf.
        one = np.ones((1, 1, 1, 1), np.float32)
        module, _, _ = _export(lambda q, k, v: _sdpa(q, k, v, scale=1.0), (one,) * 3)
        result = _execute(module, (np.full_like(one, -np.inf), one, one))
        np.testing.assert_array_equal(result, np.zeros_like(one))

    # Known limitation: the attention lowering zeroes a fully masked NaN row.
    @unittest.expectedFailure
    @unittest.skipIf(iree_runtime is None, "iree.runtime not available")
    def test_all_false_mask_with_nan_query_is_nan(self):
        # A NaN score row is not all -inf, so it stays NaN under a full mask.
        one = np.ones((1, 1, 1, 1), np.float32)
        hidden = np.zeros((1, 1), np.bool_)
        module, _, _ = _export(
            lambda q, k, v, m: _sdpa(q, k, v, attn_mask=m, scale=1.0),
            (one, one, one, hidden),
        )
        result = _execute(module, (np.full_like(one, np.nan), one, one, hidden))
        self.assertTrue(np.isnan(result).all())


def _add_case(name, fn, args):
    setattr(
        TorchaxAttentionTest,
        f"test_{name}",
        lambda self: self._check(fn, args),
    )


for _name, (_fn, _args) in _CASES.items():
    _add_case(_name, _fn, _args)


if __name__ == "__main__":
    unittest.main()
