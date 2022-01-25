# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Macros that can be used freely when building RTL modules."""

from .constants import *

from ...importer import (
    def_ir_macro_intrinsic,
    ImportStage,
)

from .... import iree_pydm as d
from ..... import ir


def _constant_i32(value: int):
  """Emits a constant i32 value."""
  return d.ConstantOp(
      d.IntegerType.get_explicit(32),
      ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)).result


def _constant_i64(value: int):
  """Emits a constant i64 value."""
  return d.ConstantOp(
      d.IntegerType.get_explicit(64),
      ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)).result


def _unbox_i32(stage: ImportStage, value: ir.Value) -> ir.Value:
  i32_type = d.IntegerType.get_explicit(32)
  if d.ObjectType.isinstance(value.type):
    return d.UnboxOp(d.ExceptionResultType.get(), i32_type, value).primitive
  else:
    if value.type != i32_type:
      stage.ic.abort(
          f"Type error unbox a non object type -> integer<32>: {value.type}")
    return value


@def_ir_macro_intrinsic
def unbox_i32(stage: ImportStage, value: ir.Value) -> ir.Value:
  return _unbox_i32(stage, value)

def _unbox_i64(stage: ImportStage, value: ir.Value) -> ir.Value:
  i64_type = d.IntegerType.get_explicit(64)
  if d.ObjectType.isinstance(value.type):
    return d.UnboxOp(d.ExceptionResultType.get(), i64_type, value).primitive
  else:
    if value.type != i64_type:
      stage.ic.abort(
          f"Type error unbox a non object type -> integer<64>: {value.type}")
    return value

@def_ir_macro_intrinsic
def unbox_i64(stage: ImportStage, value: ir.Value) -> ir.Value:
  return _unbox_i64(stage, value)


@def_ir_macro_intrinsic
def unbox_f32(stage: ImportStage, value: ir.Value) -> ir.Value:
  t = d.RealType.get_explicit(ir.F32Type.get())
  return d.UnboxOp(d.ExceptionResultType.get(), t, value).primitive


@def_ir_macro_intrinsic
def get_type_code(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Gets the TypeCode (see C++ BuiltinTypeCode) associated with a value.

  This always returns an integer<32>, which is expected by macros which
  operate on type codes.
  """
  return d.GetTypeCodeOp(d.IntegerType.get_explicit(32), value).result


@def_ir_macro_intrinsic
def is_numeric_type_code(stage: ImportStage, type_code: ir.Value):
  """Determines whether the type code is part of the numeric hierarchy."""
  type_code_i32 = _unbox_i32(stage, type_code)
  t = type_code_i32.type
  shifted = d.ApplyBinaryOp(t, ir.StringAttr.get("rshift"), type_code_i32,
                            _constant_i32(TYPE_NUMERIC_SHIFT)).result
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("eq"), shifted,
                          _constant_i32(TYPE_NUMERIC_SHIFTED_VALUE)).result


@def_ir_macro_intrinsic
def get_type_code_numeric_category(stage: ImportStage, type_code: ir.Value):
  type_code_i32 = _unbox_i32(stage, type_code)
  t = type_code_i32.type
  masked = d.ApplyBinaryOp(t, ir.StringAttr.get("and"), type_code_i32,
                           _constant_i32(TYPE_NUMERIC_CATEGORY_MASK)).result
  shifted = d.ApplyBinaryOp(t, ir.StringAttr.get("rshift"), masked,
                            _constant_i32(TYPE_NUMERIC_CATEGORY_SHIFT)).result
  return shifted


@def_ir_macro_intrinsic
def get_type_code_numeric_subtype(stage: ImportStage, type_code: ir.Value):
  type_code_i32 = _unbox_i32(stage, type_code)
  t = type_code_i32.type
  return d.ApplyBinaryOp(t, ir.StringAttr.get("and"), type_code_i32,
                         _constant_i32(TYPE_NUMERIC_SUBTYPE_MASK)).result


@def_ir_macro_intrinsic
def unbox_bool(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Unboxes an object value to a bool, not checking for success."""
  return d.UnboxOp(d.ExceptionResultType.get(), d.BoolType.get(),
                   value).primitive


@def_ir_macro_intrinsic
def cmpnz_i32(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Promotes a numeric value to i32 and compares it to zero.

  Returns True if not zero.
  This should not be needed in the fullness of time but works around type
  inference limitations in low level code.
  """
  value_i32 = _unbox_i32(stage, value)
  zero = _constant_i32(0)
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("ne"), value_i32,
                          zero).result


@def_ir_macro_intrinsic
def cmpnz_i64(stage: ImportStage, value: ir.Value) -> ir.Value:
  """Promotes a numeric value to i64 and compares it to zero.

  Returns True if not zero.
  This should not be needed in the fullness of time but works around type
  inference limitations in low level code.
  """
  value_i64 = _unbox_i64(stage, value)
  zero = _constant_i64(0)
  return d.ApplyCompareOp(d.BoolType.get(), ir.StringAttr.get("ne"), value_i64,
                          zero).result


@def_ir_macro_intrinsic
def raise_value_error(stage: ImportStage, dummy_return: ir.Value) -> ir.Value:
  """Raises a value error.

  This needs to be changed completely when more capabilities are available.
  It is only enough now to get program flow correct.
  """
  exc_result = d.FailureOp(d.ExceptionResultType.get()).result
  d.RaiseOnFailureOp(exc_result)
  return dummy_return
