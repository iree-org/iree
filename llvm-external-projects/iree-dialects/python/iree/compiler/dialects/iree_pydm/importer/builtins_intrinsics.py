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
class py_builtin_print(Intrinsic):

  # TODO: Obviously not right.
  def emit_immediate(self, ic: ImportContext) -> ir.Value:
    return d.NoneOp(d.NoneType.get(ic.context), ip=ic.ip, loc=ic.loc).result


@Intrinsic.make_singleton
class py_builtin_len(Intrinsic):

  def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                keywords: Sequence[Any]) -> ir.Value:
    if keywords or len(args) != 1:
      stage.ic.abort("the builtin `len` expects a single argument")
    result = d.LenOp(result=d.IntegerType.get(), target=args[0]).result
    return result

  def __repr__(self):
    return "__len__"
