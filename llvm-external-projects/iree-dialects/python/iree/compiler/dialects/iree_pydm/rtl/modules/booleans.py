# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runtime library functions relating to booleans."""

from .constants import *
from .macros import *
from ..base import RtlModule

RTL_MODULE = RtlModule("booleans")


@RTL_MODULE.export_pyfunc
def object_as_bool(v) -> bool:
  if is_type(v, TYPE_BOOL):
    return unbox_unchecked_bool(v)
  elif is_type(v, TYPE_NONE):
    return False
  elif is_type(v, TYPE_INTEGER):
    return raw_compare_ne(unbox_unchecked_integer(v), 0)
  elif is_type(v, TYPE_REAL):
    return raw_compare_ne(unbox_unchecked_real(v), 0.0)
  # TODO: List, Str, Bytes, Tuple, user objects, etc.
  return True
