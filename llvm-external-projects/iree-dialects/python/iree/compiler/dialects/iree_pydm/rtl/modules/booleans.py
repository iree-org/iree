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
  type_code = get_type_code(v)
  if unbox_i32(type_code) == TYPE_NONE:
    return False
  elif is_numeric_type_code(type_code):
    numeric_cat = get_type_code_numeric_category(type_code)
    if unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_BOOL:
      return unbox_bool(v)
    numeric_subtype = get_type_code_numeric_subtype(type_code)

    # Switch based on numeric category and subtype. Generally, we either
    # promote to the 32bit variant of a category, or if v is the 64bit
    # variant, then we compare directly with 64bit comparison functions.
    if unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_SIGNED:
      if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
        # 32bit comparison.
        return cmpnz_i32(v)
      if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
        # Do 64 bit comparison.
        return cmpnz_i64(v)
      else:
        # TODO: Support 8/16 bit promotion.
        return raise_value_error(False)
    elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
      if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
        return True
      elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        return True
      elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        return True
      else:
        # TODO: BF16?
        return raise_value_error(False)
    else:
      # TODO: Unsigned, apsigned, weak.
      return raise_value_error(False)

  # TODO: List, Str, Bytes, Tuple, user objects, etc.
  return raise_value_error(False)
