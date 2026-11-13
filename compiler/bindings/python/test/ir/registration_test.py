# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir

# Just a simple test that dialects have been registered properly on the
# context.
with ir.Context() as ctx:
    input_module = ir.Module.parse(
        r"""
    builtin.module  {
      func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
  """
    )
    print(input_module)


# Torch input support is optional. When enabled, its Python dialect modules
# should load normally for both namespace access and generic IR construction.
with ir.Context() as ctx:
    if ctx.is_registered_operation("torch.constant.int"):
        from iree.compiler.dialects import torch, torch_c

        assert isinstance(ctx.dialects["torch"], torch._Dialect)
        assert isinstance(ctx.dialects["torch_c"], torch_c._Dialect)
        tensor_type = ir.Type.parse("!torch.vtensor<[1],f32>")
        for _ in range(3):
            assert ir.Type.parse("!torch.vtensor<[1],f32>") == tensor_type
        print("Torch Python dialect registration passed")
