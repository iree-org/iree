# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top-level python system API.

This facility layers on top of the underlying binding native facilities and
exposes them in a way that allows general operation against contexts, modules
and functions.
"""

# pylint: disable=protected-access
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test

__all__ = ["load_module", "load_modules", "Config", "SystemContext"]

import os
import sys

from typing import Optional, Sequence, Tuple

from . import binding as _binding

# Typing aliases (largely used for documentation).
AnyModule = _binding.VmModule

# Environment key for a comma-delimitted list of drivers to try to load.
PREFERRED_DRIVER_ENV_KEY = "IREE_DEFAULT_DRIVER"

# Default value for IREE_DRIVER
DEFAULT_IREE_DRIVER_VALUE = "vulkan,vmla"


def _create_default_iree_driver(
    driver_names: Optional[Sequence[str]] = None) -> _binding.HalDriver:
  """Returns a default driver based on environment settings."""
  # TODO(laurenzo): Ideally this should take a module and join any explicitly
  # provided driver list with environmental constraints and what the module
  # was compiled for.
  if driver_names is None:
    # Read from environment.
    driver_names = os.environ.get(PREFERRED_DRIVER_ENV_KEY)
    if driver_names is None:
      driver_names = DEFAULT_IREE_DRIVER_VALUE
    driver_names = driver_names.split(",")
  available_driver_names = _binding.HalDriver.query()
  driver_exceptions = {}
  for driver_name in driver_names:
    if driver_name not in available_driver_names:
      print(
          "Could not create driver %s (not registered)" % driver_name,
          file=sys.stderr)
      continue
    try:
      driver = _binding.HalDriver.create(driver_name)
      # TODO(laurenzo): Remove these prints to stderr (for now, more information
      # is better and there is no better way to report it yet).
    except Exception as ex:  # pylint: disable=broad-except
      print(
          "Could not create default driver %s: %r" % (driver_name, ex),
          file=sys.stderr)
      driver_exceptions[driver_name] = ex
      continue

    # Sanity check creation of the default device and skip the driver if
    # this fails (this works around issues where the driver is present
    # but there are no devices). This default initialization scheme needs
    # to be improved.
    try:
      device = driver.create_default_device()
    except Exception as ex:
      print(
          "Could not create default driver device %s: %r" % (driver_name, ex),
          file=sys.stderr)
      driver_exceptions[driver_name] = ex
      continue

    print("Created IREE driver %s: %r" % (driver_name, driver), file=sys.stderr)
    return driver

  # All failed.
  raise RuntimeError("Could not create any requested driver "
                     "%r (available=%r) : %r" %
                     (driver_names, available_driver_names, driver_exceptions))


class Config:
  """System configuration."""

  driver: _binding.HalDriver
  device: _binding.HalDevice
  vm_instance: _binding.VmInstance
  host_type_factory: _binding.HostTypeFactory
  default_modules: Tuple[AnyModule]

  def __init__(self, driver_name: Optional[str] = None):
    self.vm_instance = _binding.VmInstance()
    self.driver = _create_default_iree_driver(
        driver_name.split(",") if driver_name is not None else None)
    self.device = self.driver.create_default_device()
    hal_module = _binding.create_hal_module(self.device)
    strings_module = _binding.create_strings_module()
    tensorlist_module = _binding.create_tensorlist_module()
    self.host_type_factory = _binding.HostTypeFactory.get_numpy()
    self.default_modules = (hal_module, strings_module, tensorlist_module)


_global_config = None


def _get_global_config():
  global _global_config
  if _global_config is None:
    _global_config = Config()
  return _global_config


class BoundFunction:
  """Wraps a VmFunction, VmContext and ABI into a pythonic function."""

  def __init__(self, context: "SystemContext",
               vm_function: _binding.VmFunction):
    self._context = context
    self._vm_function = vm_function
    self._abi = context.create_function_abi(vm_function)

  def __call__(self, *args):
    # NOTE: This is just doing sync dispatch right now. In the future,
    # this should default to async and potentially have some kind of policy
    # flag that can allow it to be overridden.
    inputs = self._abi.raw_pack_inputs(args)
    results = self._abi.allocate_results(inputs, static_alloc=False)
    self._context._vm_context.invoke(self._vm_function, inputs, results)
    unpacked_results = self._abi.raw_unpack_results(results)
    # TODO(laurenzo): When switching from 'raw' to structured pack/unpack,
    # the ABI should take care of this one-arg special case.
    if len(unpacked_results) == 1:
      return unpacked_results[0]
    elif len(unpacked_results) == 0:
      return None
    else:
      return unpacked_results

  def __repr__(self):
    return "<BoundFunction %r (%r)>" % (
        self._abi,
        self._vm_function,
    )


class BoundModule:
  """Wraps a VmModule with its context and provides nice python accessors.

  Resolves item access (["foo"]) as function resolution.
  """

  def __init__(self, context: "SystemContext", vm_module: AnyModule):
    self._context = context
    self._vm_module = vm_module
    self._lazy_functions = dict()

  @property
  def name(self):
    return self._vm_module.name

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
      raise KeyError("Function '%s' not found in module '%s'" %
                     (name, self.name))
    bound_function = BoundFunction(self._context, vm_function)
    self._lazy_functions[name] = bound_function
    return bound_function

  def __repr__(self):
    return "<BoundModule %r>" % (self._vm_module,)


class Modules(dict):
  """Provides nice python accessors for a dict of modules."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)


class SystemContext:
  """Global system."""

  def __init__(self, modules=None, config: Optional[Config] = None):
    self._config = config if config is not None else _get_global_config()
    print("SystemContext driver=%r" % self._config.driver, file=sys.stderr)
    self._is_dynamic = modules is None
    if not self._is_dynamic:
      init_modules = self._config.default_modules + tuple(modules)
    else:
      init_modules = None

    self._vm_context = _binding.VmContext(
        instance=self._config.vm_instance, modules=init_modules)

    if self._is_dynamic:
      self._vm_context.register_modules(self._config.default_modules)
      self._modules = Modules([
          (m.name, BoundModule(self, m)) for m in self._config.default_modules
      ])
    else:
      self._modules = Modules([
          (m.name, BoundModule(self, m)) for m in init_modules
      ])

  @property
  def is_dynamic(self) -> bool:
    return self._is_dynamic

  @property
  def config(self) -> Config:
    return self._config

  @property
  def instance(self) -> _binding.VmInstance:
    return self._instance

  @property
  def modules(self) -> Modules:
    return self._modules

  def create_function_abi(self, f: _binding.VmFunction) -> _binding.FunctionAbi:
    return self._vm_context.create_function_abi(self._config.device,
                                                self._config.host_type_factory,
                                                f)

  def add_modules(self, modules):
    assert self._is_dynamic, "Cannot 'add_module' on a static context"
    for m in modules:
      name = m.name
      if name in self._modules:
        raise ValueError("Attempt to register duplicate module: '%s'" % (name,))
      self._modules[m.name] = BoundModule(self, m)
    self._vm_context.register_modules(modules)

  def add_module(self, module):
    self.add_modules((module,))


def load_modules(*modules, config: Optional[Config] = None):
  """Loads modules into a new or shared context and returns them."""
  context = SystemContext(modules=modules, config=config)
  bound_modules = [context.modules[m.name] for m in modules]
  return bound_modules


def load_module(module, **kwargs):
  """Loads a module into a new or shared context and returns them."""
  return load_modules(module, **kwargs)[0]
