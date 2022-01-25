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


@RTL_MODULE.export_pyfunc
def dynamic_binary_promote(left, right) -> tuple:
  left_type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if is_numeric_type_code(left_type_code):
    if is_numeric_type_code(right_type_code):
      # TODO: Actually promote
      return left, right
  return raise_value_error(())


@RTL_MODULE.export_pyfunc
def apply_binary_add(left, right):
  type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if unbox_i32(type_code) == unbox_i32(right_type_code):
    if is_numeric_type_code(type_code):
      numeric_cat = get_type_code_numeric_category(type_code)
      numeric_subtype = get_type_code_numeric_subtype(type_code)
      if unbox_i32(numeric_cat) <= TYPE_NUMERIC_CATEGORY_SIGNED:
        # For bool, weak int, unsigned and signed, promote to the compute
        # size and perform the arithmetic.
        # TODO: This is a bit of a violation in that how unbox is defined
        # does not allow an implicit bitcast to a related type. However, we
        # could lift this restriction for low-level code such as this.
        # Investigate.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
          return unbox_i32(left) + unbox_i32(right)
        elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
          return unbox_i64(left) + unbox_i64(right)
        # TODO: Promote all others to the native compute type.
      elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
        # These must all be explicit.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
          return unbox_f32(left) + unbox_f32(right)
        # TODO: Implement the rest.
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        #   return unbox_f64(left) + unbox_f64(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        #   return unbox_f16(left) + unbox_f16(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_BF16:
        #   return unbox_bf16(left) + unbox_bf16(right)
  return raise_value_error(0)


# TODO: Do some template generation for these to avoid repetition.
@RTL_MODULE.export_pyfunc
def apply_binary_sub(left, right):
  type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if unbox_i32(type_code) == unbox_i32(right_type_code):
    if is_numeric_type_code(type_code):
      numeric_cat = get_type_code_numeric_category(type_code)
      numeric_subtype = get_type_code_numeric_subtype(type_code)
      if unbox_i32(numeric_cat) <= TYPE_NUMERIC_CATEGORY_SIGNED:
        # For bool, weak int, unsigned and signed, promote to the compute
        # size and perform the arithmetic.
        # TODO: This is a bit of a violation in that how unbox is defined
        # does not allow an implicit bitcast to a related type. However, we
        # could lift this restriction for low-level code such as this.
        # Investigate.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
          return unbox_i32(left) - unbox_i32(right)
        elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
          return unbox_i64(left) - unbox_i64(right)
        # TODO: Promote all others to the native compute type.
      elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
        # These must all be explicit.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
          return unbox_f32(left) - unbox_f32(right)
        # TODO: Implement the rest.
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        #   return unbox_f64(left) + unbox_f64(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        #   return unbox_f16(left) + unbox_f16(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_BF16:
        #   return unbox_bf16(left) + unbox_bf16(right)
  return raise_value_error(0)


# TODO: Do some template generation for these to avoid repetition.
@RTL_MODULE.export_pyfunc
def apply_binary_mul(left, right):
  type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if unbox_i32(type_code) == unbox_i32(right_type_code):
    if is_numeric_type_code(type_code):
      numeric_cat = get_type_code_numeric_category(type_code)
      numeric_subtype = get_type_code_numeric_subtype(type_code)
      if unbox_i32(numeric_cat) <= TYPE_NUMERIC_CATEGORY_SIGNED:
        # For bool, weak int, unsigned and signed, promote to the compute
        # size and perform the arithmetic.
        # TODO: This is a bit of a violation in that how unbox is defined
        # does not allow an implicit bitcast to a related type. However, we
        # could lift this restriction for low-level code such as this.
        # Investigate.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
          return unbox_i32(left) * unbox_i32(right)
        elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
          return unbox_i64(left) * unbox_i64(right)
        # TODO: Promote all others to the native compute type.
      elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
        # These must all be explicit.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
          return unbox_f32(left) * unbox_f32(right)
        # TODO: Implement the rest.
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        #   return unbox_f64(left) + unbox_f64(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        #   return unbox_f16(left) + unbox_f16(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_BF16:
        #   return unbox_bf16(left) + unbox_bf16(right)
  return raise_value_error(0)


# TODO: Do some template generation for these to avoid repetition.
@RTL_MODULE.export_pyfunc
def apply_compare_le(left, right) -> bool:
  type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if unbox_i32(type_code) == unbox_i32(right_type_code):
    if is_numeric_type_code(type_code):
      numeric_cat = get_type_code_numeric_category(type_code)
      numeric_subtype = get_type_code_numeric_subtype(type_code)
      if unbox_i32(numeric_cat) <= TYPE_NUMERIC_CATEGORY_SIGNED:
        # For bool, weak int, unsigned and signed, promote to the compute
        # size and perform the arithmetic.
        # TODO: This is a bit of a violation in that how unbox is defined
        # does not allow an implicit bitcast to a related type. However, we
        # could lift this restriction for low-level code such as this.
        # Investigate.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
          return unbox_i32(left) <= unbox_i32(right)
        elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
          return unbox_i64(left) <= unbox_i64(right)
        # TODO: Promote all others to the native compute type.
      elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
        # These must all be explicit.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
          return unbox_f32(left) <= unbox_f32(right)
        # TODO: Implement the rest.
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        #   return unbox_f64(left) + unbox_f64(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        #   return unbox_f16(left) + unbox_f16(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_BF16:
        #   return unbox_bf16(left) + unbox_bf16(right)
  return raise_value_error(False)


# TODO: Do some template generation for these to avoid repetition.
@RTL_MODULE.export_pyfunc
def apply_compare_eq(left, right) -> bool:
  type_code = get_type_code(left)
  right_type_code = get_type_code(right)
  if unbox_i32(type_code) == unbox_i32(right_type_code):
    if is_numeric_type_code(type_code):
      numeric_cat = get_type_code_numeric_category(type_code)
      numeric_subtype = get_type_code_numeric_subtype(type_code)
      if unbox_i32(numeric_cat) <= TYPE_NUMERIC_CATEGORY_SIGNED:
        # For bool, weak int, unsigned and signed, promote to the compute
        # size and perform the arithmetic.
        # TODO: This is a bit of a violation in that how unbox is defined
        # does not allow an implicit bitcast to a related type. However, we
        # could lift this restriction for low-level code such as this.
        # Investigate.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER32:
          return unbox_i32(left) == unbox_i32(right)
        elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_INTEGER64:
          return unbox_i64(left) == unbox_i64(right)
        # TODO: Promote all others to the native compute type.
      elif unbox_i32(numeric_cat) == TYPE_NUMERIC_CATEGORY_REAL:
        # These must all be explicit.
        if unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP32:
          return unbox_f32(left) == unbox_f32(right)
        # TODO: Implement the rest.
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP64:
        #   return unbox_f64(left) + unbox_f64(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_FP16:
        #   return unbox_f16(left) + unbox_f16(right)
        # elif unbox_i32(numeric_subtype) == TYPE_NUMERIC_SUBTYPE_BF16:
        #   return unbox_bf16(left) + unbox_bf16(right)
  return raise_value_error(False)


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
