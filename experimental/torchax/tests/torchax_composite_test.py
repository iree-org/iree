# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import jax
import numpy as np
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

from iree_torchax import composite


def _composite_signature(module_text):
    with mlir.make_ir_context():
        module = ir.Module.parse(module_text)
        assert module.operation.verify()
        symbols = ir.SymbolTable(module.operation)
        found = []

        def visit(operation):
            if operation.name == "stablehlo.composite":
                body = symbols[operation.attributes["decomposition"].value]
                signature = ir.FunctionType(
                    ir.TypeAttr(body.attributes["function_type"]).value
                )
                found.append(
                    (
                        [str(value.type) for value in operation.operands],
                        [str(type_) for type_ in signature.inputs],
                    )
                )
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)
        assert len(found) == 1
        return found[0]


class CompositeTest(unittest.TestCase):
    def test_symbolic_dimensions_stay_inside_decomposition(self):
        (n,) = jax.export.symbolic_shape("n")
        double = composite(lambda x: x * 2, name="example.double")
        module = jax.export.export(jax.jit(double))(
            jax.ShapeDtypeStruct((n, 4), np.float32)
        ).mlir_module()
        operands, arguments = _composite_signature(module)
        self.assertEqual(operands, ["tensor<?x4xf32>"])
        self.assertEqual(arguments, operands)

    def test_captured_constants_remain_operands(self):
        (n,) = jax.export.symbolic_shape("n")
        offset = jax.numpy.arange(4, dtype=np.float32)
        add = composite(lambda x: x + offset, name="example.add")
        module = jax.export.export(jax.jit(add))(
            jax.ShapeDtypeStruct((n, 4), np.float32)
        ).mlir_module()
        operands, arguments = _composite_signature(module)
        self.assertEqual(operands, ["tensor<4xf32>", "tensor<?x4xf32>"])
        self.assertEqual(arguments, operands)
        x = np.ones((2, 4), np.float32)
        np.testing.assert_array_equal(jax.jit(add)(x), x + np.arange(4))

    def test_unsolvable_operand_dimensions_preserve_upstream_signature(self):
        n, m = jax.export.symbolic_shape("n, m")
        double = composite(lambda x: x * 2, name="example.product_shape")

        def fn(x, a, b):
            return double(x), a, b

        module = jax.export.export(jax.jit(fn))(
            jax.ShapeDtypeStruct((n * m,), np.float32),
            jax.ShapeDtypeStruct((n,), np.float32),
            jax.ShapeDtypeStruct((m,), np.float32),
        ).mlir_module()
        operands, arguments = _composite_signature(module)
        self.assertEqual(operands, ["tensor<i32>", "tensor<i32>", "tensor<?xf32>"])
        self.assertEqual(arguments, operands)

    def test_used_platform_selector_preserves_fallback(self):
        select = composite(
            lambda x: jax.lax.platform_dependent(
                x, cpu=lambda x: x + 1, cuda=lambda x: x + 2
            ),
            name="example.platform",
        )
        artifact = jax.export.export(jax.jit(select), platforms=("cpu", "cuda"))(
            jax.ShapeDtypeStruct((3,), np.float32)
        )
        operands, arguments = _composite_signature(artifact.mlir_module())
        self.assertEqual(operands, ["tensor<i32>", "tensor<3xf32>"])
        self.assertEqual(arguments, operands)
        x = np.arange(3, dtype=np.float32)
        np.testing.assert_array_equal(artifact.call(x), x + 1)


if __name__ == "__main__":
    unittest.main()
