# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler_backend import ir
from iree.compiler_backend.dialects import chlo
from iree.compiler_backend.dialects import mhlo
from iree.compiler_backend.dialects import iree_public
from iree.compiler_backend.dialects import builtin
from iree.compiler_backend.dialects import std
from iree.compiler_backend.dialects import linalg
try:
  from iree.compiler_backend.dialects import linalg
except ImportError as e:
  print("KNOWN ISSUE: Linalg has an absolute path dependency issue:", e)
from iree.compiler_backend.dialects import math
from iree.compiler_backend.dialects import memref
from iree.compiler_backend.dialects import shape
from iree.compiler_backend.dialects import tensor
from iree.compiler_backend.dialects import tosa
from iree.compiler_backend.dialects import vector

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
