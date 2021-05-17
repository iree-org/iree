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

# TODO(#4131) python>=3.7: Use postponed type annotations.

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

import os
import sys

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from . import binding as _binding
from .function import FunctionInvoker

import numpy as np

# Environment key for a comma-delimitted list of drivers to try to load.
PREFERRED_DRIVER_ENV_KEY = "IREE_DEFAULT_DRIVER"

# Default value for IREE_DRIVER
DEFAULT_IREE_DRIVER_VALUE = "dylib,vulkan,vmvx"

# Mapping from IREE target backends to their corresponding drivers.
TARGET_BACKEND_TO_DRIVER = {
    "dylib-llvm-aot": "dylib",
    "vmvx": "vmvx",
    "vulkan-*": "vulkan",
}


def _create_default_iree_driver(
    driver_names: Optional[Sequence[str]] = None) -> _binding.HalDriver:
  """Returns a default driver based on environment settings."""
  # TODO(laurenzo): Ideally this should take a VmModule and join any explicitly
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
      print(f"Could not create driver {driver_name} (not registered)",
            file=sys.stderr)
      continue
    try:
      driver = _binding.HalDriver.create(driver_name)
      # TODO(laurenzo): Remove these prints to stderr (for now, more information
      # is better and there is no better way to report it yet).
    except Exception as ex:  # pylint: disable=broad-except
      print(f"Could not create default driver {driver_name}: {repr(ex)}",
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
      print(f"Could not create default driver device {driver_name}: {repr(ex)}",
            file=sys.stderr)
      driver_exceptions[driver_name] = ex
      continue

    print(f"Created IREE driver {driver_name}: {repr(driver)}", file=sys.stderr)
    return driver

  # All failed.
  raise RuntimeError(
      f"Could not create any requested driver {repr(driver_names)} (available="
      f"{repr(available_driver_names)}) : {repr(driver_exceptions)}")


class Config:
  """System configuration."""

  driver: _binding.HalDriver
  device: _binding.HalDevice
  vm_instance: _binding.VmInstance
  host_type_factory: _binding.HostTypeFactory
  default_vm_modules: Tuple[_binding.VmModule, ...]

  def __init__(self, driver_name: Optional[str] = None):
    self.vm_instance = _binding.VmInstance()
    self.driver = _create_default_iree_driver(
        driver_name.split(",") if driver_name is not None else None)
    self.device = self.driver.create_default_device()
    hal_module = _binding.create_hal_module(self.device)
    strings_module = _binding.create_strings_module()
    tensorlist_module = _binding.create_tensorlist_module()
    self.host_type_factory = _binding.HostTypeFactory.get_numpy()
    self.default_vm_modules = (hal_module, strings_module, tensorlist_module)


_global_config = None


def _get_global_config():
  global _global_config
  if _global_config is None:
    _global_config = Config()
  return _global_config


def _bool_to_int8(
    array: Any) -> Optional[Union[np.ndarray, List[Any], Tuple[Any]]]:
  if not isinstance(array, np.ndarray):
    return array

  # IREE models booleans as i8s.
  # TODO(#5359): This cast should be moved into the function abi.
  if array.dtype == np.bool:
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


class BoundFunction:
  """Wraps a VmFunction, VmContext and ABI into a pythonic function."""

  def __init__(self, context: "SystemContext",
               vm_function: _binding.VmFunction):
    self._context = context
    self._vm_function = vm_function
    self._abi = context.create_function_abi(vm_function)
    self._serialized_inputs = None
    self._serialized_outputs = None

  def __call__(self, *args, **kwargs):
    # Convert tensors, device arrays, ints, ... to IREE-friendly inputs.
    args = [normalize_value(value) for value in args]
    kwargs = {k: normalize_value(v) for k, v in kwargs.items()}
    args = [_bool_to_int8(value) for value in args]
    kwargs = {k: _bool_to_int8(v) for k, v in kwargs.items()}

    # NOTE: This is just doing sync dispatch right now. In the future,
    # this should default to async and potentially have some kind of policy
    # flag that can allow it to be overridden.
    inputs = self._abi.pack_inputs(*args, **kwargs)
    self._serialized_inputs = tuple(self._abi.serialize_vm_list(inputs))
    results = self._abi.allocate_results(inputs, static_alloc=False)
    self._context._vm_context.invoke(self._vm_function, inputs, results)
    self._serialized_outputs = tuple(self._abi.serialize_vm_list(results))
    unpacked_results = self._abi.unpack_results(results)

    # TODO(#5359): Add support for list and tuple return types.
    # The SIP signature used by the runtime bindings cannot differentiate
    # between Lists and Tuples, as it only has a single 'sequence' type.
    # The function abi uses py::list when unpacking the results according to the
    # SIP signature. The most common instance of a returned Sequence in Python
    # however is multiple return values, and that is represented by a tuple.
    # We manually change the return types of all Sequences to Tuple in order to
    # match the semantics of this case.
    unpacked_results = _convert_lists_to_tuples(unpacked_results)

    return unpacked_results

  def __repr__(self):
    return f"<BoundFunction {repr(self._abi)} ({repr(self._vm_function)})>"

  def get_serialized_values(self):
    if self._serialized_inputs is None:
      raise RuntimeError("Attempted to call get_serialized_values() before "
                         "any values were passed.")
    return self._serialized_inputs, self._serialized_outputs


class BoundModule:
  """Wraps a VmModule with its context and provides nice python accessors.

  Resolves item access (["foo"]) as function resolution.
  """

  def __init__(self, context: "SystemContext", vm_module: _binding.VmModule):
    self._context = context
    self._vm_module = vm_module
    self._lazy_functions = dict()

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

    # TODO: Remove this fork and delete the local BoundFunction once SIP is
    # removed. We take the new path if there is a native IREE ABI attribute
    # or no SIP ('f') attribute.
    reflection_dict = vm_function.reflection
    if "iree.abi" in reflection_dict or "f" not in reflection_dict:
      # TODO: Needing to know the precise device to allocate on here is bad
      # layering and will need to be fixed in some fashion if/when doing
      # heterogenous dispatch.
      return FunctionInvoker(self._context.vm_context,
                             self._context.config.device, vm_function)

    # Legacy SIP path.
    bound_function = BoundFunction(self._context, vm_function)
    self._lazy_functions[name] = bound_function
    return bound_function

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
    self._config = config if config is not None else _get_global_config()
    print(f"SystemContext driver={repr(self._config.driver)}", file=sys.stderr)
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
    return self._instance

  @property
  def modules(self) -> BoundModules:
    return self._bound_modules

  def create_function_abi(self, f: _binding.VmFunction) -> _binding.FunctionAbi:
    return self._vm_context.create_function_abi(self._config.device,
                                                self._config.host_type_factory,
                                                f)

  def add_vm_modules(self, vm_modules):
    assert self._is_dynamic, "Cannot 'add_module' on a static context"
    for m in vm_modules:
      if m.name in self._bound_modules:
        raise ValueError(f"Attempt to register duplicate VmModule: '{m.name}'")
      self._bound_modules[m.name] = BoundModule(self, m)
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
  vm_module = _binding.VmModule.from_flatbuffer(vm_flatbuffer)
  config = Config(TARGET_BACKEND_TO_DRIVER[backend])
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
