# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import re
import sys
import unittest

_MISSING = [
    name
    for name in ("jax", "torch", "torchax")
    if importlib.util.find_spec(name) is None
]
if _MISSING:
    print(f"{', '.join(_MISSING)} not found, skipping torchax_export_test")
    sys.exit(77)

import jax
import numpy as np
import torch
import torchax

from iree_torchax import torchax_export


class _UnusedConstants(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(4, 5))
        self.weight = torch.nn.Parameter(torch.arange(3.0))
        self.register_buffer("table", torch.ones(5, 2))

    def forward(self, x):
        return x * self.weight


class _TorchDtypes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4))
        self.register_buffer(
            "scale", torch.tensor([[0.5, 2.0], [4.0, 8.0]], dtype=torch.float64)
        )

    def forward(self, x, index):
        return x @ self.weight, index + 1, self.scale * 2


class _TiedWeights(torch.nn.Module):
    # The embedding and the projection share one weight, as in tied language
    # models.
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 6)
        self.head = torch.nn.Linear(6, 11, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        return self.head(self.embed(ids))


class _BufferScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Buffer(torch.tensor([0.5, 0.25, 0.125]), persistent=False)

    def forward(self, x):
        return x * self.scale


class _ConstantsAndBuffers(torch.nn.Module):
    # Decomposition puts the lifted constant before the submodule's buffer in
    # the program's inputs.
    def __init__(self):
        super().__init__()
        self.inner = _BufferScale()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        return self.inner(x) * self.weight + torch.tensor([10.0, 20.0, 30.0])


class _Random(torch.nn.Module):
    def __init__(self, draw):
        super().__init__()
        self.draw = draw

    def forward(self, x):
        return self.draw(x)


def _run_with_jax(module, *inputs):
    exported = torch.export.export(module, tuple(map(torch.from_numpy, inputs)))
    program = torchax_export.exported_program_to_jax(exported)
    with jax.enable_x64(True):
        _, outputs = jax.jit(program.func)(program.constants, program.mutable, inputs)
    return [np.asarray(output) for output in outputs]


class TorchaxExportTest(unittest.TestCase):
    def test_stablehlo_keeps_torch_dtypes_and_matmul_precision(self):
        # Torch's int64 and f64 survive, f32 matmuls ask for full precision,
        # and the caller's JAX configuration is left as it was.
        exported = torch.export.export(
            _TorchDtypes(),
            (torch.ones(2, 3), torch.arange(10, dtype=torch.int64).reshape(2, 5)),
        )
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(
            [str(aval.dtype) for aval in artifact.in_avals],
            ["float32", "float64", "float32", "int64"],
        )
        self.assertEqual(
            [str(aval.dtype) for aval in artifact.out_avals],
            ["float32", "int64", "float64"],
        )
        self.assertIn("precision = [HIGHEST, HIGHEST]", str(artifact.mlir_module()))
        self.assertFalse(jax.config.jax_enable_x64)
        self.assertIsNone(jax.config.jax_default_matmul_precision)

    def test_stablehlo_keeps_unused_constants(self):
        # main takes every constant, used or not, in the documented order.
        exported = torch.export.export(_UnusedConstants(), (torch.ones(2, 3),))
        program, artifact = torchax_export.exported_program_to_stablehlo(exported)
        self.assertEqual(program.constant_names, ["unused", "weight", "table"])
        self.assertEqual(
            sorted(artifact.module_kept_var_idx), list(range(len(artifact.in_avals)))
        )

    def test_lifted_constants_and_buffers_keep_their_inputs(self):
        x = np.array([1.0, -2.0, 4.0], np.float32)
        (actual,) = _run_with_jax(_ConstantsAndBuffers(), x)
        expected = _ConstantsAndBuffers()(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(actual, expected)

    def test_tied_weight_is_one_constant(self):
        torch.manual_seed(0)
        module = _TiedWeights()
        ids = np.array([[1, 4, 9], [10, 0, 3]], np.int64)
        exported = torch.export.export(module, (torch.from_numpy(ids),))
        self.assertEqual(
            len([s for s in exported.graph_signature.input_specs if s.target]), 2
        )
        program = torchax_export.exported_program_to_jax(exported)
        self.assertEqual(len(program.constant_names), 1)
        (actual,) = _run_with_jax(module, ids)
        expected = module(torch.from_numpy(ids)).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_random_operations_name_the_missing_random_state(self):
        x = np.ones((3, 4), np.float32)
        cases = [
            ("rand", "aten.rand.default", lambda x: x + torch.rand(x.shape)),
            ("randn_like", "aten.randn_like.default", torch.randn_like),
            (
                "dropout",
                "aten.rand.default",
                lambda x: torch.nn.functional.dropout(x, 0.5, training=True),
            ),
        ]
        for name, op, draw in cases:
            with self.subTest(name), self.assertRaisesRegex(
                NotImplementedError, re.escape(op) + ".*random state"
            ):
                _run_with_jax(_Random(draw), x)

    def test_dropout_outside_training_exports(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        draw = lambda x: torch.nn.functional.dropout(x, 0.5, training=False)
        (result,) = _run_with_jax(_Random(draw), x)
        np.testing.assert_array_equal(result, x)

    def test_torchax_handler_in_torch_runs_in_the_environment(self):
        # Decomposition removes in-place ops, whose Torchax handlers are
        # written in Torch, so only a graph built by hand reaches one.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(torch.ops.aten.sigmoid_.default, (x,)))
        module = torch.fx.GraphModule(torch.nn.Module(), graph)
        x = np.linspace(-2.0, 2.0, 15, dtype=np.float32).reshape(3, 5)
        result = torchax_export._Interpreter(module, torchax.tensor.Environment()).run(
            jax.numpy.asarray(x), enable_io_processing=False
        )
        expected = torch.sigmoid(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
