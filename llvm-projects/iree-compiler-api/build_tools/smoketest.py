# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.compiler.dialects import chlo
from iree.compiler.dialects import mhlo
from iree.compiler.dialects import iree_public
from iree.compiler.dialects import builtin
from iree.compiler.dialects import std
from iree.compiler.dialects import linalg
try:
  from iree.compiler.dialects import linalg
except ImportError as e:
  print("KNOWN ISSUE: Linalg has an absolute path dependency issue:", e)
from iree.compiler.dialects import math
from iree.compiler.dialects import memref
from iree.compiler.dialects import shape
from iree.compiler.dialects import tensor
from iree.compiler.dialects import tosa
from iree.compiler.dialects import vector

with ir.Context() as ctx:
  try:
    chlo.register_chlo_dialect(ctx)
  except ImportError as e:
    print(
        "KNOWN ISSUE: For hidden visibility builds extensions need "
        "an explicit dep on LLVMSupport (chlo):", e)
  try:
    mhlo.register_mhlo_dialect(ctx)
  except ImportError as e:
    print(
        "KNOWN ISSUE: For hidden visibility builds extensions need "
        "an explicit dep on LLVMSupport (mhlo):", e)
  iree_public.register_iree_public_dialect(ctx)
