# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import subprocess

from iree.compiler import ir
from iree.compiler import passmanager
from iree.compiler import version
from iree.compiler.dialects import arith
from iree.compiler.dialects import chlo
from iree.compiler.dialects import mhlo
from iree.compiler.dialects import iree_input
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

from iree.compiler.transforms import ireec

# Test the compiler API.
with ir.Context() as ctx:
  chlo.register_chlo_dialect(ctx)
  mhlo.register_mhlo_dialect(ctx)
  iree_input.register_dialect(ctx)

  input_module = ir.Module.parse(r"""
    builtin.module  {
      builtin.func @fabs(%arg0: tensor<1x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
        %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
        %1 = "mhlo.abs"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %1 : tensor<4x4xf32>
      }
    }
  """)

  options = ireec.CompilerOptions("--iree-input-type=mhlo",
                                  "--iree-hal-target-backends=cpu")
  print(options)
  pm = passmanager.PassManager()
  ireec.build_iree_vm_pass_pipeline(options, pm)
  pm.run(input_module)

  print(input_module)
  bytecode_io = io.BytesIO()
  ireec.translate_module_to_vm_bytecode(options, input_module, bytecode_io)
  print(f"Bytecode module len = {len(bytecode_io.getbuffer())}")

# Check version.
print(f"PACKAGE_SUFFIX={version.PACKAGE_SUFFIX}")
print(f"VERSION={version.VERSION}")
print(f"REVISIONS={version.REVISIONS!r}")

# Check console scripts.
subprocess.check_output(["ireec", "-help"])
subprocess.check_output(["iree-translate", "-help"])
