# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

from .util import ImportContext, Intrinsic
from ... import iree_pydm as d
from .... import ir


@Intrinsic.make_singleton
class print(Intrinsic):

  # TODO: Obviously not right.
  def emit_immediate(self, ic: ImportContext) -> ir.Value:
    return d.NoneOp(d.NoneType.get(ic.context), ip=ic.ip, loc=ic.loc).result
