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

from typing import Tuple

from . import binding as _binding

# Typing aliases (largely used for documentation).
AnyModule = _binding.vm.VmModule


class Config:

  vm_instance: _binding.vm.VmInstance
  host_type_factory: _binding.host_types.HostTypeFactory
  default_modules: Tuple[AnyModule]


class _GlobalConfig(Config):
  """Singleton of globally configured instances."""

  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance._static_init()
    return cls._instance

  def _static_init(self):
    self.vm_instance = _binding.vm.VmInstance()
    self.driver_names = _binding.hal.HalDriver.query()
    # TODO(laurenzo): More flexible selection of driver and device.
    self.driver = _binding.hal.HalDriver.create("vulkan")
    self.device = self.driver.create_default_device()
    self.hal_module = _binding.vm.create_hal_module(self.device)
    self.host_type_factory = _binding.host_types.HostTypeFactory.get_numpy()
    self.default_modules = (self.hal_module,)


class BoundFunction:
  """Wraps a VmFunction, VmContext and ABI into a pythonic function."""

  def __init__(self, context: "SystemContext",
               vm_function: _binding.vm.VmFunction):
    self._context = context
    self._vm_function = vm_function
    self._abi = context.create_function_abi(vm_function)

  def __call__(self, *args):
    # NOTE: This is just doing sync dispatch right now. In the future,
    # this should default to async and potentially have some kind of policy
    # flag that can allow it to be overriden.
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

  def __init__(self, modules=None, config: Config = None):
    self._config = config if config is not None else _GlobalConfig()
    self._is_dynamic = modules is None
    if not self._is_dynamic:
      init_modules = self._config.default_modules + tuple(modules)
    else:
      init_modules = None

    self._vm_context = _binding.vm.VmContext(
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
  def instance(self) -> _binding.vm.VmInstance:
    return self._instance

  @property
  def modules(self) -> Modules:
    return self._modules

  def create_function_abi(
      self, f: _binding.vm.VmFunction) -> _binding.function_abi.FunctionAbi:
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


def load_modules(*modules):
  """Loads modules into a new or shared context and returns them."""
  context = SystemContext(modules=modules)
  context_modules = context.modules
  bound_modules = [context_modules[m.name] for m in modules]
  return bound_modules


def load_module(module):
  """Loads a module into a new or shared context and returns them."""
  return load_modules(module)[0]
