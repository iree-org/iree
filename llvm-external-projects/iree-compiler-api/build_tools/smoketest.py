# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io

from iree.compiler import ir
from iree.compiler import passmanager
from iree.compiler.dialects import chlo
from iree.compiler.dialects import mhlo
from iree.compiler.dialects import iree as iree_dialect
from iree.compiler.dialects import builtin
from iree.compiler.dialects import std
from iree.compiler.dialects import linalg
from iree.compiler.dialects import linalg
from iree.compiler.dialects import math
from iree.compiler.dialects import memref
from iree.compiler.dialects import shape
from iree.compiler.dialects import tensor
from iree.compiler.dialects import tosa
from iree.compiler.dialects import vector

from iree.compiler.api import driver

with ir.Context() as ctx:
  chlo.register_chlo_dialect(ctx)
  mhlo.register_mhlo_dialect(ctx)
  iree_dialect.register_dialect(ctx)

  input_module = ir.Module.parse(r"""
    builtin.module  {
      builtin.func @fabs(%arg0: tensor<1x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
        %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
        %1 = "mhlo.abs"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %1 : tensor<4x4xf32>
      }
    }
  """)

  options = driver.CompilerOptions()
  options.set_input_dialect_mhlo()
  options.add_target_backend("cpu")
  pm = passmanager.PassManager()
  driver.build_iree_vm_pass_pipeline(options, pm)
  pm.run(input_module)

  print(input_module)
  bytecode_io = io.BytesIO()
  driver.translate_module_to_vm_bytecode(options, input_module, bytecode_io)
  print(f"Bytecode module len = {len(bytecode_io.getbuffer())}")
