# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Disable PyType, it does not seem to like the specialization pattern used in
# MLIR.
# pytype: skip-file
try:
  from .. import ir
  from ..dialects import pdl
  from ._ods_common import extend_opview_class as _ods_extend_opview_class, segmented_accessor as _ods_segmented_accessor, equally_sized_accessor as _ods_equally_sized_accessor, get_default_loc_context as _ods_get_default_loc_context, get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
  from typing import Optional, Sequence, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e
BoolArg = Optional[Union[bool, ir.BoolAttr]]
IntArg = Optional[Union[int, ir.IntegerAttr]]
IntListArg = Optional[Union[Sequence[int], ir.ArrayAttr]]
IntListListArg = Optional[Union[Sequence[Union[Sequence[int], ir.ArrayAttr]],
                                ir.ArrayAttr]]
StringArg = Optional[Union[str, ir.StringAttr]]
StringListArg = Optional[Union[Sequence[str], ir.ArrayAttr]]


def _defaulted_ensure(f):

  def inner(value, default=None):
    assert value is not None or default is not None
    return f(default if value is None else value)

  return inner


@_defaulted_ensure
def _ensure_int_array_attr(value: IntListArg):
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(value, Sequence):
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i64, i) for i in value])
  return value


@_defaulted_ensure
def _ensure_string_array_attr(value: StringListArg):
  if isinstance(value, Sequence):
    return ir.ArrayAttr.get([ir.StringAttr.get(str(i)) for i in value])
  return value


@_defaulted_ensure
def _ensure_array_of_array_attr(value: IntListListArg):
  if isinstance(value, Sequence):
    return ir.ArrayAttr.get([_ensure_int_array_attr(inner) for inner in value])
  return value


@_defaulted_ensure
def _ensure_int_attr(value: IntArg):
  if isinstance(value, int):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
  return value


@_defaulted_ensure
def _ensure_bool_attr(value: BoolArg):
  if isinstance(value, bool):
    return ir.BoolAttr.get(value)
  return value


@_defaulted_ensure
def _ensure_string_attr(value: StringArg):
  if isinstance(value, str):
    return ir.StringAttr.get(value)
  return value


def _count_expected_loops(tile_sizes: ir.ArrayAttr) -> int:
  # Number of loops = number of tile sizes != 0
  zero = _ensure_int_attr(0)
  return len(list(tile_sizes)) - list(tile_sizes).count(zero)


##===----------------------------------------------------------------------===##
## LinalgExt specific transforms
##===----------------------------------------------------------------------===##


class TileToLinalgExtTileOp:
  """Specialization for the TileToLinalgExtTileOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               sizes: IntListArg = None,
               loc=None,
               ip=None):
    sizes = _ensure_int_array_attr(sizes, [])
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, sizes, loc=loc, ip=ip)


class FuseIntoContainingOp:
  """Specialization for the FuseIntoContainingOp class."""

  def __init__(self,
               producerOp: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               containingOp: Union[ir.Value, ir.Operation, ir.OpView],
               loc=None,
               ip=None):
    super().__init__([], producerOp, containingOp, loc=loc, ip=ip)
