# Lint-as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
R'''
This module contains Python wrappers for various IREE command-line tools.

This top-level API provides access to the `iree-compiler` tool, which compiles
MLIR ASM via IREE's compiler to a supported output format (i.e. VM FlatBuffer, C
source code, etc).

Example
~~~~~~~

.. code-block:: python

  import iree.compiler.tools

  SIMPLE_MUL_ASM = """
  func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
      %0 = tosa.mul %arg0, %arg1 {shift = 0 : i8} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
  }
  """

  # Also see compile_file()
  # There are many keyword options available.
  # See iree.compiler.CompilerOptions
  binary = iree.compiler.tools.compile_str(
      SIMPLE_MUL_ASM, input_type="tosa", target_backends=["llvm-cpu"])
'''

from .core import *
from .debugging import TempFileSaver
from .binaries import CompilerToolError
