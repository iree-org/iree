# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Macros that can be used freely when building RTL modules."""

from ...importer import (
    def_ir_macro_intrinsic,
    ImportStage,
)

from .... import iree_pydm as d
from ..... import ir


@def_ir_macro_intrinsic
def get_type_code(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Gets the TypeCode (see C++ BuiltinTypeCode) associated with a value."""
  return d.GetTypeCodeOp(d.IntegerType.get(), value).result


@def_ir_macro_intrinsic
def is_type(stage: ImportStage, value: ir.Value,
            type_code: ir.Value) -> ir.Value:
  """Efficiently checks whether a value has a given type code."""
  ic = stage.ic
  if not d.IntegerType.isinstance(type_code.type):
    ic.abort(f"is_type() macro must be called with a constant type_code. "
             f"Got {type_code}")
  actual_type_code = get_type_code(stage, value)
  cmp_result = d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("eq"),
                                type_code, actual_type_code).result
  return cmp_result


@def_ir_macro_intrinsic
def get_numeric_promotion_order(stage: ImportStage,
                                value: ir.Value) -> ir.Value:
  """Gets the numeric promotion order.

  See get_numeric_promotion_order op.
  """
  return d.GetNumericPromotionOrderOp(d.IntegerType.get(), value).result


@def_ir_macro_intrinsic
def promote_numeric_to_integer(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Promotes the value to IntegerType."""
  return d.PromoteNumericOp(d.IntegerType.get(), value).result


@def_ir_macro_intrinsic
def promote_numeric_to_real(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Promotes the value to RealType."""
  return d.PromoteNumericOp(d.RealType.get(), value).result


@def_ir_macro_intrinsic
def unbox_unchecked_bool(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Unboxes an object value to a bool, not checking for success."""
  return d.UnboxOp(d.ExceptionResultType.get(), d.BoolType.get(),
                   value).primitive


@def_ir_macro_intrinsic
def unbox_unchecked_integer(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Unboxes an object value to an integer, not checking for success."""
  return d.UnboxOp(d.ExceptionResultType.get(), d.IntegerType.get(),
                   value).primitive


@def_ir_macro_intrinsic
def unbox_unchecked_real(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Unboxes an object value to a real, not checking for success."""
  return d.UnboxOp(d.ExceptionResultType.get(), d.RealType.get(),
                   value).primitive


@def_ir_macro_intrinsic
def raw_compare_eq(stage: ImportStage, left: ir.Value,
                   right: ir.Value) -> ir.Value:
  """Emits an ApplyCompareOp for 'eq'."""
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("eq"), left,
                          right).result


@def_ir_macro_intrinsic
def raw_compare_gt(stage: ImportStage, left: ir.Value,
                   right: ir.Value) -> ir.Value:
  """Emits an ApplyCompareOp for 'gt'."""
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("gt"), left,
                          right).result


@def_ir_macro_intrinsic
def raw_compare_ge(stage: ImportStage, left: ir.Value,
                   right: ir.Value) -> ir.Value:
  """Emits an ApplyCompareOp for 'ge'."""
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("ge"), left,
                          right).result


@def_ir_macro_intrinsic
def raw_compare_ne(stage: ImportStage, left: ir.Value,
                   right: ir.Value) -> ir.Value:
  """Emits an ApplyCompareOp for 'ne'."""
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("ne"), left,
                          right).result


@def_ir_macro_intrinsic
def raise_value_error(stage: ImportStage, dummy_return: ir.Value) -> ir.Value:
  """Raises a value error.

  This needs to be changed completely when more capabilities are available.
  It is only enough now to get program flow correct.
  """
  exc_result = d.FailureOp(d.ExceptionResultType.get()).result
  d.RaiseOnFailureOp(exc_result)
  return dummy_return
