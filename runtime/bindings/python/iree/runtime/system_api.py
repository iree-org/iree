# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Top-level python system API.

This facility layers on top of the underlying binding native facilities and
exposes them in a way that allows general operation against contexts, modules
and functions.
"""

# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

from __future__ import annotations

__all__ = [
    "load_vm_flatbuffer",
    "load_vm_flatbuffer_file",
    "load_vm_module",
    "load_vm_modules",
    "normalize_value",
    "Config",
    "SystemContext",
    "TARGET_BACKEND_TO_DRIVER",
]

import logging
import os
import sys

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from . import _binding
from .function import FunctionInvoker
from .system_setup import get_first_device
from . import tracing

import numpy as np

# Mapping from IREE target backends to their corresponding drivers.
TARGET_BACKEND_TO_DRIVER = {
    "llvm-cpu": "local-task",
    "vmvx": "local-task",
    "vulkan-spirv": "vulkan",
}


class Config:
  """System configuration."""

  device: _binding.HalDevice
  vm_instance: _binding.VmInstance
  default_vm_modules: Tuple[_binding.VmModule, ...]
  tracer: Optional[tracing.Tracer]

  def __init__(self,
               driver_name: Optional[str] = None,
               *,
               device: Optional[_binding.HalDevice] = None,
               tracer: Optional[tracing.Tracer] = None):
    # Either use an explicit device or auto config based on driver names.
    if device is not None and driver_name is not None:
      raise ValueError(
          "Either 'device' or 'driver_name' can be specified (not both)")
    if device is not None:
      self.device = device
    else:
      self.device = get_first_device(
          driver_name.split(",") if driver_name is not None else None)

    self.vm_instance = _binding.VmInstance()
    hal_module = _binding.create_hal_module(self.vm_instance, self.device)
    self.default_vm_modules = (hal_module,)
    self.tracer = tracer or tracing.get_default_tracer()
    if self.tracer and self.tracer.enabled:
      logging.info("IREE runtime tracing calls to path: %s",
                   self.tracer.trace_path)
    else:
      self.tracer = None


def _bool_to_int8(
    array: Any) -> Optional[Union[np.ndarray, List[Any], Tuple[Any]]]:
  if not isinstance(array, np.ndarray):
    return array

  # IREE models booleans as i8s.
  # TODO(#5359): This cast should be moved into the function abi.
  if array.dtype == bool:
    array = array.astype(np.int8)
  return array


def normalize_value(
    value: Any) -> Optional[Union[np.ndarray, List[Any], Tuple[Any]]]:
  """Normalizes the given value for input to (or comparison with) IREE."""
  if value is None:
    # Exclude None from falling through to blanket np.asarray conversion.
    return value

  if isinstance(value, (list, tuple, dict)):
    return value

  array = np.asarray(value)
  # TODO(#5359): Move into the function abi.
  if isinstance(value, (bool, int, float)):
    # Manually convert ints and floats to 32 bits.
    if array.dtype == np.float64:
      array = array.astype(np.float32)
    elif array.dtype == np.int64:
      array = array.astype(np.int32)

  return array


def _convert_lists_to_tuples(pytree):
  if isinstance(pytree, Sequence):
    return tuple(_convert_lists_to_tuples(leaf) for leaf in pytree)
  elif isinstance(pytree, Mapping):
    for key in pytree:
      pytree[key] = _convert_lists_to_tuples(pytree[key])
    return pytree
  else:
    return pytree


class BoundModule:
  """Wraps a VmModule with its context and provides nice python accessors.

  Resolves item access (["foo"]) as function resolution.
  """

  def __init__(self, context: SystemContext, vm_module: _binding.VmModule):
    self._context = context
    self._tracer = self._context._config.tracer
    self._vm_module = vm_module
    self._lazy_functions = dict()

    # Let the tracing infra create a traced module.
    self.traced_module = None
    if self._tracer:
      self.traced_module = self._tracer.persist_vm_module(vm_module)

  @property
  def name(self):
    return self._vm_module.name

  @property
  def vm_module(self):
    return self._vm_module

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __getitem__(self, name):
    vm_function = self._lazy_functions.get(name)
    if vm_function is not None:
      return vm_function

    vm_function = self._vm_module.lookup_function(name)
    if vm_function is None:
      raise KeyError(f"Function '{name}' not found in module '{self}'")

    # TODO: Needing to know the precise device to allocate on here is bad
    # layering and will need to be fixed in some fashion if/when doing
    # heterogenous dispatch.
    return FunctionInvoker(self._context.vm_context,
                           self._context.config.device, vm_function,
                           self._context._tracer)

  def __repr__(self):
    return f"<BoundModule {repr(self._vm_module)}>"


class BoundModules(dict):
  """Provides nice python accessors for a dict of BoundModules."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)


class SystemContext:
  """Global system."""

  def __init__(self, vm_modules=None, config: Optional[Config] = None):
    self._config = config if config is not None else Config()
    self._is_dynamic = vm_modules is None
    if self._is_dynamic:
      init_vm_modules = None
    else:
      init_vm_modules = self._config.default_vm_modules + tuple(vm_modules)

    self._vm_context = _binding.VmContext(instance=self._config.vm_instance,
                                          modules=init_vm_modules)

    if self._is_dynamic:
      self._vm_context.register_modules(self._config.default_vm_modules)
      self._bound_modules = BoundModules([
          (m.name, BoundModule(self, m))
          for m in self._config.default_vm_modules
      ])
    else:
      self._bound_modules = BoundModules([
          (m.name, BoundModule(self, m)) for m in init_vm_modules
      ])

    self._tracer = None  # type: Optional[tracing.ContextTracer]
    if self._config.tracer:
      self._tracer = tracing.ContextTracer(
          self._config.tracer,
          is_dynamic=self._is_dynamic,
          modules=[bm.traced_module for bm in self._bound_modules.values()])

  @property
  def vm_context(self) -> _binding.VmContext:
    return self._vm_context

  @property
  def is_dynamic(self) -> bool:
    return self._is_dynamic

  @property
  def config(self) -> Config:
    return self._config

  @property
  def instance(self) -> _binding.VmInstance:
    return self._config.vm_instance

  @property
  def modules(self) -> BoundModules:
    return self._bound_modules

  def add_module_dependency(self, name, minimum_version=0):
    resolved_module = _binding.VmModule.resolve_module_dependency(
        self._config.vm_instance, name, minimum_version)
    self._vm_context.register_modules([resolved_module])

  def add_vm_modules(self, vm_modules):
    assert self._is_dynamic, "Cannot 'add_module' on a static context"
    for m in vm_modules:
      if m.name in self._bound_modules:
        raise ValueError(f"Attempt to register duplicate VmModule: '{m.name}'")
      bound_module = BoundModule(self, m)
      self._bound_modules[m.name] = bound_module
      if self._tracer:
        self._tracer.add_module(bound_module.traced_module)
    self._vm_context.register_modules(vm_modules)

  def add_vm_module(self, vm_module):
    self.add_vm_modules((vm_module,))


def load_vm_modules(*vm_modules, config: Optional[Config] = None):
  """Loads VmModules into a new SystemContext and returns them."""
  context = SystemContext(vm_modules=vm_modules, config=config)
  bound_modules = [context.modules[m.name] for m in vm_modules]
  return bound_modules


def load_vm_module(vm_module, config: Optional[Config] = None):
  """Loads a VmModule into a new SystemContext and returns it."""
  return load_vm_modules(vm_module, config=config)[0]


def load_vm_flatbuffer(vm_flatbuffer: bytes,
                       *,
                       driver: Optional[str] = None,
                       backend: Optional[str] = None) -> BoundModule:
  """Loads a VM Flatbuffer into a callable module.

  Either 'driver' or 'backend' must be specified.
  """
  if driver is None and backend is None:
    raise ValueError("Either 'driver' or 'backend' must be specified, but got "
                     "'None' for both.")
  if backend is not None and driver is not None:
    raise ValueError("Cannot specify both 'driver' and a 'backend' to infer "
                     "the driver from.")
  if backend is not None:
    driver = TARGET_BACKEND_TO_DRIVER[backend]
  config = Config(driver)
  vm_module = _binding.VmModule.from_flatbuffer(config.vm_instance,
                                                vm_flatbuffer)
  bound_module = load_vm_module(vm_module, config)
  return bound_module


# TODO: There should be an API for mmap'ing the file which should be used
# instead of reading into memory.
def load_vm_flatbuffer_file(path: str,
                            *,
                            driver: Optional[str] = None,
                            backend: Optional[str] = None) -> BoundModule:
  """Loads a file containing a VM Flatbuffer into a callable module.

  Either 'driver' or 'backend' must be specified.
  """
  with open(path, "rb") as f:
    vm_flatbuffer = f.read()
  return load_vm_flatbuffer(vm_flatbuffer, driver=driver, backend=backend)
