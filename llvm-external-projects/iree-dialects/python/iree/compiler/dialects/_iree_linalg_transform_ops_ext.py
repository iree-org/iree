#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

try:
  from .. import ir
  from ..dialects import pdl
  from typing import Optional, Sequence, Union
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e

BoolArg = Optional[Union[bool, ir.BoolAttr]]
IntArg = Optional[Union[int, ir.IntegerAttr]]
IntListArg = Optional[Union[Sequence[int], ir.ArrayAttr]]
IntListListArg = Optional[Union[Sequence[Union[Sequence[int], ir.ArrayAttr]],
                                ir.ArrayAttr]]
StringArg = Optional[Union[str, ir.StringAttr]]


def _defaulted_ensure(f):

  def inner(value, default=None):
    assert value is not None or default is not None
    return f(default if value is None else value)

  return inner


@_defaulted_ensure
def _ensure_array_attr(value: IntListArg):
  i64 = ir.IntegerType.get_signless(64)
  if isinstance(value, Sequence):
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i64, i) for i in value])
  return value


@_defaulted_ensure
def _ensure_array_of_array_attr(value: IntListListArg):
  if isinstance(value, Sequence):
    return ir.ArrayAttr.get([_ensure_array_attr(inner) for inner in value])
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


class MatchOp:
  """Specialization for the MatchOp class."""

  def __init__(self, target: Union[str, ir.FlatSymbolRefAttr]):
    if isinstance(target, str):
      target = ir.FlatSymbolRefAttr.get(target)

    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target)


class LowerVectorsOp:
  """Specialization for the LowerVectorsOp class."""

  def __init__(self,
               *,
               stages: IntListArg = None,
               contraction_lowering: StringArg = None,
               multireduction_lowering: StringArg = None,
               split_transfers: StringArg = None,
               unroll_vector_transfers: BoolArg = None,
               transpose_lowering: StringArg = None,
               transpose_avx2_lowering: BoolArg = None,
               loc=None,
               ip=None):
    stages = _ensure_array_attr(stages, [0, 1, 2, 3, 4, 5, 6])
    contraction_lowering = _ensure_string_attr(contraction_lowering,
                                               "outerproduct")
    multireduction_lowering = _ensure_string_attr(multireduction_lowering,
                                                  "innerparallel")
    split_transfers = _ensure_string_attr(split_transfers, "linalg-copy")
    unroll_vector_transfers = _ensure_bool_attr(unroll_vector_transfers, True)
    transpose_lowering = _ensure_string_attr(transpose_lowering, "eltwise")
    transpose_avx2_lowering = _ensure_bool_attr(transpose_avx2_lowering, False)

    super().__init__(stages,
                     contraction_lowering,
                     multireduction_lowering,
                     split_transfers,
                     unroll_vector_transfers,
                     transpose_lowering,
                     transpose_avx2_lowering,
                     loc=loc,
                     ip=ip)


class LowerToLLVMOp:
  """Specialization for the LowerToLLVMOp class."""

  def __init__(self,
               *,
               reassociate_fp_reductions: BoolArg = None,
               enable_index_optimizations: BoolArg = None,
               enable_arm_neon: BoolArg = None,
               enable_arm_sve: BoolArg = None,
               enable_amx: BoolArg = None,
               enable_x86vector: BoolArg = None,
               enable_async: BoolArg = None,
               loc=None,
               ip=None):
    super().__init__(_ensure_bool_attr(reassociate_fp_reductions, False),
                     _ensure_bool_attr(enable_index_optimizations, False),
                     _ensure_bool_attr(enable_arm_neon, False),
                     _ensure_bool_attr(enable_arm_sve, False),
                     _ensure_bool_attr(enable_amx, False),
                     _ensure_bool_attr(enable_x86vector, False),
                     _ensure_bool_attr(enable_async, False),
                     loc=loc,
                     ip=ip)


class FuseOp:
  """Specialization for the FuseOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               tile_sizes: IntListArg = None,
               tile_interchange: IntListArg = None,
               loc=None,
               ip=None):
    tile_sizes = _ensure_array_attr(tile_sizes, [])
    tile_interchange = _ensure_array_attr(tile_interchange, [])
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type,
                     target,
                     tile_sizes,
                     tile_interchange,
                     loc=loc,
                     ip=ip)


class TileOp:
  """Specialization for the TileOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               sizes: IntListArg = None,
               interchange: IntListArg = None,
               peel: IntListArg = None,
               scalarize_dyn_dims: BoolArg = None,
               loc=None,
               ip=None):
    sizes = _ensure_array_attr(sizes, [])
    interchange = _ensure_array_attr(interchange, [])
    peel = _ensure_array_attr(peel, [])
    scalarize_dyn_dims = _ensure_bool_attr(scalarize_dyn_dims, False)
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type,
                     target,
                     sizes,
                     interchange,
                     peel,
                     scalarize_dyn_dims,
                     loc=loc,
                     ip=ip)


class PadOp:
  """Specialization for the PadOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               pack_paddings: IntListArg = None,
               hoist_paddings: IntListArg = None,
               transpose_paddings: IntListListArg = None,
               loc=None,
               ip=None):
    pack_paddings = _ensure_array_attr(pack_paddings, [])
    hoist_paddings = _ensure_array_attr(hoist_paddings, [])
    transpose_paddings = _ensure_array_of_array_attr(transpose_paddings, [])
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type,
                     target,
                     pack_paddings,
                     hoist_paddings,
                     transpose_paddings,
                     loc=loc,
                     ip=ip)


class GeneralizeOp:
  """Specialization for the GeneralizeOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type, target, loc=loc, ip=ip)


class InterchangeOp:
  """Specialization for the InterchangeOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               iterator_interchange: IntListArg = None,
               loc=None,
               ip=None):
    iterator_interchange = _ensure_array_attr(iterator_interchange, [])
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type,
                     target,
                     iterator_interchange,
                     loc=loc,
                     ip=ip)


class VectorizeOp:

  def __init__(self,
               target: Optional[Union[ir.Value, ir.Operation,
                                      ir.OpView]] = None,
               *,
               vectorize_padding: BoolArg = None,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()

    super().__init__(operation_type if target is not None else None,
                     target,
                     _ensure_bool_attr(vectorize_padding, False),
                     loc=loc,
                     ip=ip)


class GetParentLoopOp:

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               num_loops: IntArg = None,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()
    num_loops = _ensure_int_attr(num_loops, 1)
    super().__init__(operation_type, target, num_loops, loc=loc, ip=ip)


class UnrollLoopOp:

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               factor: Union[int, ir.IntegerAttr],
               loc=None,
               ip=None):
    # Factor must not be None, do not provide the default value here.
    factor = _ensure_int_attr(factor)
    super().__init__(target, factor, loc=loc, ip=ip)


class PipelineLoopOp:

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               iteration_interval: IntArg,
               read_latency: IntArg,
               loc=None,
               ip=None):
    iteration_interval = _ensure_int_attr(iteration_interval, 1)
    read_latency = _ensure_int_attr(read_latency, 10)
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type,
                     target,
                     iteration_interval,
                     read_latency,
                     loc=loc,
                     ip=ip)


class OutlineLoopOp:

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               func_name: StringArg,
               loc=None,
               ip=None):
    # Function name must not be None, do not provide the default value.
    func_name = _ensure_string_attr(func_name)
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, func_name, loc=loc, ip=ip)


class SequenceOp:

  def __init__(self, *, loc=None, ip=None):
    super().__init__(loc=loc, ip=ip)
    self.body.blocks.append()


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
    sizes = _ensure_array_attr(sizes, [])
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, sizes, loc=loc, ip=ip)


class RewriteLinalgExtTileToScfForOp:
  """Specialization for the RewriteLinalgExtTileToScfForOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, loc=loc, ip=ip)


class RewriteLinalgExtTileToInParallelOp:
  """Specialization for the RewriteLinalgExtTileToInParallelOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, loc=loc, ip=ip)


class RewriteLinalgExtInParallelToScfForOp:
  """Specialization for the RewriteLinalgExtInParallelToScfForOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, loc=loc, ip=ip)


class RewriteLinalgExtInParallelToAsyncOp:
  """Specialization for the RewriteLinalgExtInParallelToAsyncOp class."""

  def __init__(self,
               target: Union[ir.Value, ir.Operation, ir.OpView],
               *,
               loc=None,
               ip=None):
    operation_type = pdl.OperationType.get()
    super().__init__(operation_type, target, loc=loc, ip=ip)
