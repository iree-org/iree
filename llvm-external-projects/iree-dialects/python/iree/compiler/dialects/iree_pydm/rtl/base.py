# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base helpers for the RTL DSL."""

from typing import List, Optional

import functools

from ..importer import (
    create_context,
    def_pyfunc_intrinsic,
    DefaultImportHooks,
    FuncProvidingIntrinsic,
    ImportContext,
    ImportStage,
)

from ... import builtin as builtin_d
from .... import ir


class RtlModule:
  """Declares a runtime library module.

  This is typically included in a Python module which exports functions as:

  ```
  RTL_MODULE = RtlModule("booleans")

  @RTL_MODULE.export_pyfunc
  def object_as_bool(v) -> bool:
    ...
  ```
  """

  def __init__(self, name: str, symbol_prefix: str = "pydmrtl$"):
    self.name = name
    self.symbol_prefix = symbol_prefix
    self.exported_funcs: List[FuncProvidingIntrinsic] = []

  def export_pyfunc(self,
                    f=None,
                    *,
                    symbol: Optional[str] = None,
                    visibility: Optional[str] = None) -> FuncProvidingIntrinsic:
    """Marks a python function for export in the created module.

    This is typically used as a decorator and returns a FuncProvidingIntrinsic.
    """
    if f is None:
      return functools.partial(self.export_pyfunc,
                               symbol=symbol,
                               visibility=visibility)
    if symbol is None:
      symbol = f.__name__
    symbol = self.symbol_prefix + symbol
    intrinsic = def_pyfunc_intrinsic(f, symbol=symbol, visibility=visibility)
    self.exported_funcs.append(intrinsic)
    return intrinsic

  def internal_pyfunc(
      self,
      f=None,
      *,
      symbol: Optional[str] = None,
      visibility: Optional[str] = "private") -> FuncProvidingIntrinsic:
    if f is None:
      return functools.partial(self.internal_pyfunc,
                               symbol=symbol,
                               visibility=visibility)
    if symbol is None:
      symbol = f.__name__
    symbol = self.symbol_prefix + symbol
    intrinsic = def_pyfunc_intrinsic(f, symbol=symbol, visibility=visibility)
    return intrinsic


class RtlBuilder:
  """A build session for a combined runtime library."""

  def __init__(self, context: Optional[ir.Context] = None):
    self.context = context if context else create_context()
    # TODO: Use hooks that import constants, etc.
    self.hooks = DefaultImportHooks()
    self.root_module = ir.Module.create(
        ir.Location.unknown(context=self.context))
    self.module_op = self.root_module.operation

  def emit_module(self, rtl_module: RtlModule):
    root_body = self.module_op.regions[0].blocks[0]
    with ir.InsertionPoint(root_body), ir.Location.unknown():
      sub_module = builtin_d.ModuleOp()
      sub_module.sym_name = ir.StringAttr.get(rtl_module.name)
    ic = ImportContext(context=self.context, module=sub_module)
    stage = ImportStage(ic=ic, hooks=self.hooks)

    # Export functions.
    for f in rtl_module.exported_funcs:
      # Getting the symbol implies exporting it into the module.
      f.get_or_create_provided_func_symbol(stage)
