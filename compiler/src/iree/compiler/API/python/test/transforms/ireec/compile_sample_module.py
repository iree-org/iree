# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io

from iree.compiler import ir
from iree.compiler import passmanager
from iree.compiler.transforms import ireec

# The compiler re-exports API access to a number of dialects. If one of these
# fails to import, it indicates a build issue.
from iree.compiler.dialects import arith
#from iree.compiler.dialects import chlo
#from iree.compiler.dialects import mhlo
from iree.compiler.dialects import iree_input
from iree.compiler.dialects import builtin
from iree.compiler.dialects import linalg
from iree.compiler.dialects import math
from iree.compiler.dialects import memref
from iree.compiler.dialects import pdl
from iree.compiler.dialects import shape
from iree.compiler.dialects import tensor
from iree.compiler.dialects import tosa
from iree.compiler.dialects import vector

# Test the compiler API.
with ir.Context() as ctx:
  ireec.register_all_dialects(ctx)

  input_module = ir.Module.parse(r"""
    builtin.module  {
      func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
  """)

  options = ireec.CompilerOptions("--iree-hal-target-backends=llvm-cpu")
  print(options)
  pm = passmanager.PassManager(anchor_op="builtin.module")
  ireec.build_iree_vm_pass_pipeline(options, pm)
  pm.run(input_module.operation)

  print(input_module)
  bytecode_io = io.BytesIO()
  ireec.translate_module_to_vm_bytecode(options, input_module, bytecode_io)
  print(f"Bytecode module len = {len(bytecode_io.getbuffer())}")
