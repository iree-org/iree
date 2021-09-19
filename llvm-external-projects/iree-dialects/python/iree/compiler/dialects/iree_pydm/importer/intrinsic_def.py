# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Sequence

import functools

from .importer import Importer
from .util import DefaultImportHooks, ImportStage, Intrinsic, FuncProvidingIntrinsic

from ... import iree_pydm as d
from .... import ir


def def_pyfunc_intrinsic(
    f=None,
    *,
    symbol: Optional[str] = None,
    visibility: Optional[str] = None,
) -> FuncProvidingIntrinsic:
  """Defines an intrinsic function that will be included in the module."""
  if f is None:
    return functools.partial(def_pyfunc_intrinsic,
                             symbol=symbol,
                             visibility=visibility)

  if symbol is None:
    symbol = f.__name__

  class PyIntrinsicFunc(FuncProvidingIntrinsic):
    """The intrinsic which will compile the func and emit calls to it."""

    def get_or_create_provided_func_symbol(self, stage: ImportStage) -> str:
      ic = stage.ic
      symbol_attr = ir.FlatSymbolRefAttr.get(symbol, context=ic.context)
      existing = ic.lookup_symbol(symbol_attr)
      if not existing:
        _import_global_function(stage, f, symbol=symbol, visibility=visibility)
      return symbol

    def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                  keywords: Sequence[Any]) -> ir.Value:
      ic = stage.ic
      if keywords:
        ic.abort(f"{self} only supports positional arguments")
      resolved_symbol = self.get_or_create_provided_func_symbol(stage)
      with ic.ip, ic.loc:
        exc_result, call_result = d.DynamicCallOp(
            d.ExceptionResultType.get(), d.ObjectType.get(),
            ir.FlatSymbolRefAttr.get(resolved_symbol), args).results
        d.RaiseOnFailureOp(exc_result)
        return call_result

    def __call__(self, *args, **kwargs):
      return f(*args, **kwargs)

    def __repr__(self):
      return f"<py intrinsic {symbol}>"

  return PyIntrinsicFunc()


def def_ir_macro_intrinsic(f=None):
  """Defines an IR macro intrinsic.

  The decorated function must take as positional arguments the
  ImportStage followed by *`ir.Value` instances corresponding with the
  call and return a single `ir.Value`.

  The function will be evaluated in an MLIR with context including
  context, location and ip.
  """
  if f is None:
    return functools.partial(def_ir_macro_intrinsic)

  class IrIntrinsicMacro(Intrinsic):

    def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                  keywords: Sequence[Any]) -> ir.Value:
      ic = stage.ic
      if keywords:
        ic.abort(f"{self} only supports positional arguments")

      # TODO: Apply pre-conditions on number of arguments, etc, for nicer
      # error messages.
      with ic.loc, ic.ip:
        result = f(stage, *args)
        if not isinstance(result, ir.Value):
          ic.abort(f"compiler intrinsic macro must return an IR Value: {f}")
        return result

    def __call__(self, *args, **kwargs):
      return f(*args, **kwargs)

    def __repr__(self):
      return f"<IR macro {self}>"

  return IrIntrinsicMacro()


def def_pattern_call_intrinsic(match_generic: Sequence[Any] = (),
                               match_specific: Sequence[Any] = ()):
  """Defines a multi-function call intrinsic."""

  def _extract_symbol_intrinsics(
      matches: Sequence[Any]) -> Sequence[FuncProvidingIntrinsic]:
    names = []
    for m in matches:
      assert isinstance(m, FuncProvidingIntrinsic), (
          f"Match functions for a def_multi_func_intrinsic must be "
          f"a FuncProvidingIntrinsic. Got: {m}")
      names.append(m)
    return names

  generic_intrinsics = _extract_symbol_intrinsics(match_generic)
  specific_intrinsics = _extract_symbol_intrinsics(match_specific)

  class IrPatternCallIntrinsic(Intrinsic):

    def emit_call(self, stage: ImportStage, args: Sequence[ir.Value],
                  keywords: Sequence[Any]) -> ir.Value:
      ic = stage.ic
      if keywords:
        ic.abort(f"{self} only supports positional arguments")

      generic_symbol_names = [
          i.get_or_create_provided_func_symbol(stage)
          for i in generic_intrinsics
      ]
      specific_symbol_names = [
          i.get_or_create_provided_func_symbol(stage)
          for i in specific_intrinsics
      ]

      with ic.ip, ic.loc:
        generic_attrs = ir.ArrayAttr.get(
            [ir.FlatSymbolRefAttr.get(s) for s in generic_symbol_names])
        specific_attrs = ir.ArrayAttr.get(
            [ir.FlatSymbolRefAttr.get(s) for s in specific_symbol_names])
        exc_result, call_result = d.PatternMatchCallOp(
            d.ExceptionResultType.get(), d.ObjectType.get(), generic_attrs,
            specific_attrs, args).results
        d.RaiseOnFailureOp(exc_result)
        return call_result

    def __repr__(self):
      return (f"<pattern call generic={generic_intrinsics}, "
              f"specific={specific_intrinsics}>")

  return IrPatternCallIntrinsic()


def _import_global_function(parent_stage: ImportStage,
                            f,
                            *,
                            symbol: str,
                            visibility: Optional[str] = None) -> d.FuncOp:
  """In a fresh import context, import a global function."""
  # Note that we are bringing out own hooks, since intrinsics are compiled with
  # defaults.
  importer = Importer(parent_stage.ic, hooks=DefaultImportHooks())
  return importer.import_global_function(f,
                                         symbol=symbol,
                                         visibility=visibility)
