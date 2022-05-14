# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base helpers for the Simple Lang DSL.

This DSL is sufficient for in-process testing of relatively simple
functions. It presently contains many ABI and interface things that
should be mainlined into regular APIs.
"""

from typing import Union
from typing import List, Optional

import io
import functools
import os
import sys

from iree.compiler.transforms import ireec
from iree.compiler.dialects.iree_pydm.importer import (
    create_context,
    def_pyfunc_intrinsic,
    DefaultImportHooks,
    FuncProvidingIntrinsic,
    ImportContext,
    ImportStage,
)
from iree.compiler.dialects.iree_pydm.rtl import (
    get_std_rtl_source_bundle,)

from iree.compiler.dialects import builtin as builtin_d
from iree.compiler.dialects import iree_pydm as pydm_d
from iree.compiler import (ir, passmanager, transforms as unused_transforms)
from iree import runtime as iree_runtime
from iree.runtime.system_api import load_vm_module

_DebugMode = Optional[Union[bool, int]]


def _resolve_debug_mode(debug: _DebugMode) -> _DebugMode:
  if debug is not None:
    return debug
  env_debug = os.environ.get("PYDM_DEBUG")
  if not env_debug:
    return False
  try:
    env_debug = int(env_debug)
  except ValueError:
    print(f"WARNING: PYDM_DEBUG malformed (expected int). Ignoring.")
    return False
  return env_debug


def jit(f=None, debug: _DebugMode = None):
  """Jit compiles a single function to IREE."""
  debug = _resolve_debug_mode(debug)
  if not f:
    return functools.partial(jit, debug=debug)

  def wrapped(*args, **kwargs):
    return getattr(wrapped.module.exports, "main")(*args, **kwargs)

  wrapped.module = SimpleModule(debug=debug)
  wrapped.module.export_pyfunc(f, symbol="main")
  # Important: Make sure it acts like the original (incl. docstring, et al).
  functools.update_wrapper(wrapped, f)
  return wrapped


class SimpleModule:
  """Declares a runtime library module.

  This is typically included in a Python module which exports functions as:

  ```
  M = SimpleModule("foo")

  @M.export_pyfunc
  def object_as_bool(v) -> bool:
    ...
  ```
  """

  def __init__(self, name: str = "module", debug: _DebugMode = False):
    self.name = name
    self.debug = debug
    self.exported_funcs: List[FuncProvidingIntrinsic] = []
    self._compiled_binary = None
    self._loaded_module = None
    self._exports = None

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
    intrinsic = def_pyfunc_intrinsic(f, symbol=symbol, visibility=visibility)
    return intrinsic

  def compile(self, context: Optional[ir.Context] = None) -> "Compiler":
    """Compiles the module given the exported and internal functions."""
    compiler = Compiler(context, debug=self.debug)
    compiler.import_module(self)
    compiler.compile()
    self._compiled_binary = compiler.translate()
    return compiler

  @property
  def compiled_binary(self):
    if self._compiled_binary is None:
      self.compile()
    return self._compiled_binary

  @property
  def loaded_module(self):
    if self._loaded_module is None:
      # TODO: This API in IREE needs substantial ergonomic work for loading
      # a module from a memory image.
      system_config = _get_global_config()
      vm_module = iree_runtime._binding.VmModule.from_flatbuffer(
          self.compiled_binary)
      self._loaded_module = load_vm_module(vm_module, system_config)
    return self._loaded_module

  @property
  def exports(self):
    if self._exports:
      return self._exports
    self._exports = PyWrapperModule()
    native_module = self.loaded_module
    for name in native_module.vm_module.function_names:
      setattr(self._exports, name,
              _create_py_wrapper(getattr(native_module, name)))
    return self._exports

  def save(self, filename: str):
    with open(filename, "wb") as f:
      f.write(self.compiled_binary)


class PyWrapperModule:
  ...


def _create_py_wrapper(native_func):
  """Wraps a native func so that it does arg/result translation."""

  def invoke(*args, **kwargs):
    try:
      exc_code, result = native_func(*args, **kwargs)
    except IndexError as e:
      # The VM raises this on an out of bounds list access, which happens
      # when a variables is read before assignment. This suits us for now,
      # since error reporting is better through the VM than gencode right now.
      # This is a bit imprecise, though.
      raise UnboundLocalError() from e

    msg = "Raised from compiled function"
    if exc_code == 0:
      return result
    elif exc_code == -1:
      raise StopIteration()
    elif exc_code == -2:
      raise StopAsyncIteration()
    elif exc_code == -3:
      raise RuntimeError(msg)
    elif exc_code == -4:
      raise ValueError(msg)
    elif exc_code == -5:
      raise NotImplementedError(msg)
    elif exc_code == -6:
      raise KeyError(msg)
    elif exc_code == -7:
      raise IndexError(msg)
    elif exc_code == -8:
      raise AttributeError(msg)
    elif exc_code == -9:
      raise TypeError(msg)
    elif exc_code == -10:
      raise UnboundLocalError(msg)
    else:
      raise RuntimeError(f"Unmapped native exception code: {exc_code}")

  return invoke


class Compiler:
  """A module being compiled."""

  def __init__(self,
               context: Optional[ir.Context] = None,
               debug: _DebugMode = False):
    self.debug = debug
    self.context = context if context else create_context(debug=debug)
    self.hooks = DefaultImportHooks()
    self.root_module = ir.Module.create(
        ir.Location.unknown(context=self.context))
    self.module_op = self.root_module.operation
    self.rtl_source_bundle = get_std_rtl_source_bundle()

    # IREE compiler options.
    self.options = ireec.CompilerOptions("--iree-hal-target-backends=cpu")

  def __str__(self):
    return str(self.root_module)

  def import_module(self, m: SimpleModule):
    with self.context:
      root_body = self.module_op.regions[0].blocks[0]
      self.module_op.attributes["sym_name"] = ir.StringAttr.get(m.name)
    ic = ImportContext(context=self.context, module=self.module_op)
    stage = ImportStage(ic=ic, hooks=self.hooks)

    # Export functions.
    for f in m.exported_funcs:
      # Getting the symbol implies exporting it into the module.
      f.get_or_create_provided_func_symbol(stage)
    if not self.root_module.operation.verify():
      raise RuntimeError(
          f"Imported Python module did not verify: {self.root_module}")

  def compile(self):
    """Compiles the module."""
    with self.context:
      if self.debug:
        print("Run 'pydm-post-import-pipeline' pipeline...", file=sys.stderr)
      pm = passmanager.PassManager.parse("pydm-post-import-pipeline")
      if self.debug > 1:
        pm.enable_ir_printing()
      lowering_options = pydm_d.LoweringOptions()
      lowering_options.link_rtl(self.rtl_source_bundle)
      if self.debug:
        print("Run 'lower-to-iree' pipeline...", file=sys.stderr)
      pydm_d.build_lower_to_iree_pass_pipeline(pm, lowering_options)
      pm.run(self.root_module)
      if self.debug:
        self.root_module.operation.print(enable_debug_info=True)

      pm = passmanager.PassManager()
      # if self.debug:
      #   pm.enable_ir_printing()
      ireec.build_iree_vm_pass_pipeline(self.options, pm)
      pm.run(self.root_module)

  def translate(self):
    """Translates to a binary, returning a buffer."""
    bytecode_io = io.BytesIO()
    ireec.translate_module_to_vm_bytecode(self.options, self.root_module,
                                          bytecode_io)
    return bytecode_io.getbuffer()


# TODO: This is hoaky and needs to go in a real runtime layer.
_cached_global_config: Optional[iree_runtime.system_api.Config] = None


def _get_global_config() -> iree_runtime.system_api.Config:
  global _cached_global_config
  if not _cached_global_config:
    _cached_global_config = iree_runtime.system_api.Config("dylib")
  return _cached_global_config
