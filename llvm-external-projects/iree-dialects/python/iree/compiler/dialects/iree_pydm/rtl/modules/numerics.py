# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RTL functions for numeric operations (on scalars)."""

from .constants import *
from .macros import *
from ..base import RtlModule

RTL_MODULE = RtlModule("numerics")

# TODO: This has drifted with the full numeric hierarchy and needs to be
# rewritten.
# @RTL_MODULE.export_pyfunc
# def dynamic_binary_promote(left, right) -> tuple:
#   left_order = get_numeric_promotion_order(left)
#   right_order = get_numeric_promotion_order(right)
#   # Note that since we are defining the numeric promotion rules, we have to
#   # use raw functions to compare (or else we would be using the thing we are
#   # defining).
#   if raw_compare_eq(left_order, right_order):
#     return left, right
#   elif raw_compare_gt(left_order, right_order):
#     return left, _promote_to(get_type_code(left), right)
#   else:
#     return _promote_to(get_type_code(right), left), right

# @RTL_MODULE.internal_pyfunc
# def _promote_to(type_code: int, value):
#   if raw_compare_eq(type_code, TYPE_INTEGER):
#     return _promote_to_integer(value)
#   elif raw_compare_eq(type_code, TYPE_REAL):
#     return _promote_to_real(value)
#   return raise_value_error(None)

# @RTL_MODULE.internal_pyfunc
# def _promote_to_integer(value) -> int:
#   if is_type(value, TYPE_BOOL):
#     return promote_numeric_to_integer(unbox_unchecked_bool(value))
#   return raise_value_error(0)

# @RTL_MODULE.internal_pyfunc
# def _promote_to_real(value) -> float:
#   if is_type(value, TYPE_BOOL):
#     return promote_numeric_to_real(unbox_unchecked_bool(value))
#   elif is_type(value, TYPE_INTEGER):
#     return promote_numeric_to_real(unbox_unchecked_integer(value))
#   return raise_value_error(0.0)
