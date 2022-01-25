# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base helpers for the RTL DSL."""

from typing import Callable, List, Optional, Sequence

import functools
import threading

from ..importer import (
    create_context,
    def_pyfunc_intrinsic,
    DefaultImportHooks,
    FuncProvidingIntrinsic,
    ImportContext,
    ImportStage,
)

from ... import (
    builtin as builtin_d,
    iree_pydm as pydm_d,
)
from .... import (
    ir,
    passmanager,
    transforms as unused_transforms,
)


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
                    visibility: Optional[str] = None):
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

  def internal_pyfunc(self,
                      f=None,
                      *,
                      symbol: Optional[str] = None,
                      visibility: Optional[str] = "private"):
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

  @staticmethod
  def build_modules(rtl_modules: Sequence[RtlModule]) -> bytes:
    """One shot build modules and return assembly."""
    b = RtlBuilder()
    b.emit_modules(rtl_modules)
    b.optimize()
    return b.root_module.operation.get_asm(binary=True, enable_debug_info=True)

  @staticmethod
  def lazy_build_source_bundle(
      rtl_modules: Sequence[RtlModule]) -> Callable[[], pydm_d.SourceBundle]:
    """Returns a function to lazily build RTL modules.

    Modules will only be built once and cached for the life of the function.
    Since RTL asm is typically passed unsafely to compiler passes, caching
    forever is important.
    """
    rtl_modules = tuple(rtl_modules)
    cache = []
    lock = threading.Lock()

    def get() -> pydm_d.SourceBundle:
      with lock:
        if not cache:
          asm_blob = RtlBuilder.build_modules(rtl_modules)
          cache.append(pydm_d.SourceBundle.from_asm(asm_blob))
        return cache[0]

    return get

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

  def emit_modules(self, rtl_modules: Sequence[RtlModule]):
    for rtl_module in rtl_modules:
      self.emit_module(rtl_module)

  def optimize(self):
    """Optimizes the RTL modules by running through stage 1 compilation."""
    with self.context:
      pm = passmanager.PassManager.parse(
          "builtin.module(pydm-post-import-pipeline)")
      pm.run(self.root_module)
