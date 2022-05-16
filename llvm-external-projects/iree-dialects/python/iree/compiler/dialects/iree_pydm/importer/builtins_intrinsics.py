# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Sequence

from .util import ImportContext, ImportStage, Intrinsic
from ... import cf as cf_d
from ... import iree_pydm as d
from .... import ir


@Intrinsic.make_singleton
class py_builtin_print(Intrinsic):

  # TODO: Obviously not right.
  def emit_immediate(self, ic: ImportContext) -> ir.Value:
    return d.NoneOp(d.NoneType.get(ic.context), ip=ic.ip, loc=ic.loc).result


@Intrinsic.make_singleton
class py_builtin_len(Intrinsic):

  def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                keywords: Sequence[Any]) -> ir.Value:
    ic = stage.ic
    if keywords or len(args) != 1:
      stage.ic.abort("the builtin `len` expects a single argument")
    with ic.loc, ic.ip:
      result = d.LenOp(result=d.IntegerType.get(), target=args[0]).result
    return ic.box(result)

  def __repr__(self):
    return "__len__"


@Intrinsic.make_singleton
class py_builtin_range(Intrinsic):

  def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                keywords: Sequence[Any]) -> ir.Value:
    ic = stage.ic
    # Range only accepts positional arguments of either:
    #   stop
    #   stop, start
    #   stop, start, step
    arity = len(args)
    if keywords or arity == 0 or arity > 3:
      stage.ic.abort(
          "the builtin `range` expects between one and three positional "
          "arguments")
    start = None
    step = None
    if arity == 1:
      stop = args[0]
    elif arity == 2:
      start = args[0]
      stop = args[1]
    else:
      start = args[0]
      stop = args[1]
      step = args[2]

    zero = stage.ic.emit_constant(0)

    # If step was explicitly provided. Ensure it is not zero.
    # Per https://docs.python.org/3/library/stdtypes.html#range
    # we should dynamically validate that step != 0 and raise a ValueError
    # if it is.
    if step is not None:
      predecessor_block = ic.ip.block
      fail_block = predecessor_block.create_after()
      successor_block = fail_block.create_after()
      with ic.loc, ic.ip:
        object_type = d.ObjectType.get()
        cmp_zero, cmp_step = d.DynamicBinaryPromoteOp(
            left=zero,
            right=step,
            left_prime=object_type,
            right_prime=object_type).results
        step_eq_zero = d.ApplyCompareOp(dunder_name=ir.StringAttr.get("eq"),
                                        left=cmp_zero,
                                        right=cmp_step,
                                        result=d.BoolType.get()).result
        step_eq_zero_pred = d.BoolToPredOp(ir.IntegerType.get_signless(1),
                                           step_eq_zero).result
        cf_d.CondBranchOp(condition=step_eq_zero_pred,
                          trueDestOperands=[],
                          falseDestOperands=[],
                          trueDest=fail_block,
                          falseDest=successor_block)
      with ic.scoped_ip(ir.InsertionPoint(fail_block)):
        with ic.loc, ic.ip:
          # TODO: Use constants for error code (-4 == ValueError).
          exc_result = d.FailureOp(exc_result=d.ExceptionResultType.get(),
                                   code=ir.IntegerAttr.get(
                                       ir.IntegerType.get_signless(32), -4))
          d.ReturnErrorOp(exc_result)

      # Continue with successor.
      ic.reset_ip(ir.InsertionPoint(successor_block))

    # Emit MakeRangeOp.
    with ic.loc, ic.ip:
      # Default values.
      if start is None:
        start = zero
      if step is None:
        step = stage.ic.emit_constant(1)

      # Start with a completely generic range type and let type inference
      # figure it out.
      range_type = d.RangeType.get()
      range_value = d.MakeRangeOp(range=range_type,
                                  stop=stop,
                                  start=start,
                                  step=step).result
    return ic.box(range_value)
