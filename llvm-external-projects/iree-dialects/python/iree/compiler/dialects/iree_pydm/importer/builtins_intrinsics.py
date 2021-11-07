# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Sequence

from .util import ImportContext, ImportStage, Intrinsic
from ... import iree_pydm as d
from .... import ir


@Intrinsic.make_singleton
class print(Intrinsic):

  # TODO: Obviously not right.
  def emit_immediate(self, ic: ImportContext) -> ir.Value:
    return d.NoneOp(d.NoneType.get(ic.context), ip=ic.ip, loc=ic.loc).result


@Intrinsic.make_singleton
class range(Intrinsic):
  # TODO: Implement emit_immediate to allow binding to the range function
  # itself.

  def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                keywords: Sequence[Any]) -> ir.Value:
    ic = stage.ic
    with ic.loc, ic.ip:
      if keywords:
        ic.abort("the 'range' builtin does not support keyword arguments")
      start = None
      end = None
      step = None
      arity = len(args)
      if arity == 1:
        end = args[0]
      elif arity == 2:
        start, end = args
      elif arity == 3:
        start, end, step = args
      else:
        ic.abort("the 'range' builtin expects 1, 2 or 3 arguments")
      if start is None:
        start = ic.emit_constant(0)
      if step is None:
        step = ic.emit_constant(1)
      return d.MakeRangeOp(d.RangeType.get(), start, end, step).result
