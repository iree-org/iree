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
"""Utilities interop with TensorFlow."""

# pylint: disable=protected-access

import collections
import os
import random
import re
import tempfile
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

from absl import flags
from absl import logging
import numpy as np
from pyiree import rt
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


def set_random_seed(seed: int = 0) -> None:
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def uniform(shape: Sequence[int], dtype: np.dtype = np.float32) -> np.ndarray:
  return np.random.uniform(size=shape).astype(dtype)


def ndarange(shape: Sequence[int], dtype: np.dtype = np.float32) -> np.ndarray:
  return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def to_mlir_type(dtype: np.dtype) -> str:
  """Returns a string that denotes the type 'dtype' in MLIR style."""
  bits = dtype.itemsize * 8
  if np.issubdtype(dtype, np.integer):
    return f"i{bits}"
  elif np.issubdtype(dtype, np.floating):
    return f"f{bits}"
  else:
    raise TypeError(f"Expected integer or floating type, but got {dtype}")


def get_shape_and_dtype(array: np.ndarray,
                        allow_non_mlir_dtype: bool = False) -> str:
  shape_dtype = [str(dim) for dim in list(array.shape)]
  if np.issubdtype(array.dtype, np.number):
    shape_dtype.append(to_mlir_type(array.dtype))
  elif allow_non_mlir_dtype:
    shape_dtype.append(f"<dtype '{array.dtype}'>")
  else:
    raise TypeError(f"Expected integer or floating type, but got {array.dtype}")
  return "x".join(shape_dtype)


def save_input_values(inputs: Sequence[np.ndarray],
                      artifacts_dir: str = None) -> str:
  """Saves input values with IREE tools format if 'artifacts_dir' is set."""
  result = []
  for array in inputs:
    shape_dtype = get_shape_and_dtype(array)
    values = " ".join([str(x) for x in array.flatten()])
    result.append(f"{shape_dtype}={values}")
  result = "\n".join(result)
  if artifacts_dir is not None:
    inputs_path = os.path.join(artifacts_dir, "inputs.txt")
    logging.info("Saving IREE input values to: %s", inputs_path)
    with open(inputs_path, "w") as f:
      f.write(result)
      f.write("\n")
  return result


def _setup_mlir_crash_reproducer(
    function: Callable[[Any], Any],
    artifacts_dir: str,
    backend_id: str,
) -> Callable[[Any], Any]:
  """Wraps `function` so that it a MLIR crash reproducer is saved if it crashes.

  Writes to `artifacts_dir/reproducer__{backend}.mlir` in the case of a crash.

  Args:
    function: The callable to decorate.
    artifacts_dir: The directory to write the reproducer to.
    backend_id: The unique backend name to use when writting the reproducer.

  Returns:
    A function with the same API as the passed function.
  """

  def decorator(*args, **kwargs):
    # Set up a crash reproducer for debugging.
    if artifacts_dir is not None:
      compiler.Context.default_crash_reproducer_path = os.path.join(
          artifacts_dir, f"reproducer__{backend_id}.mlir")
    try:
      results = function(*args, **kwargs)
    except Exception:  # pylint: disable=broad-except
      # Disable the crash reproducer (to avoid inadvertently overwriting it).
      if artifacts_dir is not None:
        compiler.Context.default_crash_reproducer_path = None
      raise
    return results

  return decorator


def _incrementally_lower_compiler_module(
    compiler_module: compiler.Module,
    backend_info: "BackendInfo",
    artifacts_dir: str,
) -> Tuple[compiler.binding.OpaqueBlob, Union[str, None]]:
  """Lowers a MLIR compiler module incrementally and saves its outputs.

  If artifacts_dir is provided then the following artifacts will be saved:
    tf_input.mlir:
      MLIR for the module in TF's input dialect.
    iree_input.mlir:
      The MLIR above translated to IREE via compiler.TF_IMPORT_PASS_PIPELINE.
    backend_id/compiled.vmfb:
      A VM FlatBuffer compiled to the target backend from the IREE MLIR above.

  Args:
    compiler_module: A compiler.Module to lower.
    backend_info: BackendInfo with the details for lowering compiler_module to
      IREE.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved. No compilation artifacts will be saved if this is not
      provided.
  """
  if artifacts_dir is not None:
    os.makedirs(artifacts_dir, exist_ok=True)
    tf_mlir_path = os.path.join(artifacts_dir, "tf_input.mlir")
    logging.info("Saving raw TF input MLIR to: %s", tf_mlir_path)
    with open(tf_mlir_path, "w") as f:
      f.write(compiler_module.to_asm())

  # Manually run the passes that tf_module_to_compiler_module usually would.
  compiler_module.run_pass_pipeline(compiler.TF_IMPORT_PASS_PIPELINE)

  if artifacts_dir is not None:
    iree_mlir_path = os.path.join(artifacts_dir, "iree_input.mlir")
    logging.info("Saving IREE input MLIR to: %s", iree_mlir_path)
    with open(iree_mlir_path, "w") as f:
      f.write(compiler_module.to_asm())

  compiled_module = compiler_module.compile(
      target_backends=backend_info.compiler_targets)

  compiled_path = None
  if artifacts_dir is not None:
    backend_dir = os.path.join(artifacts_dir, backend_info.backend_id)
    os.makedirs(backend_dir, exist_ok=True)
    compiled_path = os.path.join(backend_dir, "compiled.vmfb")
    logging.info("Saving compiled IREE module to: %s", compiled_path)
    with open(compiled_path, "wb") as f:
      f.write(compiled_module)
  return compiled_module, compiled_path


def _incrementally_compile_tf_module(
    module: Type[tf.Module],
    backend_info: "BackendInfo",
    exported_names: Sequence[str] = (),
    artifacts_dir: str = None,
) -> Tuple[compiler.binding.OpaqueBlob, Union[str, None]]:
  """Compiles a TensorFlow tf.Module and optionally saves compilation artifacts.

  The module blob this creates is not callable. See IreeCompiledModule for an
  API that returns a module that can be called without any further steps.

  See _incrementally_lower_compiler_module's docstring for details about which
  artifacts will be saved.

  Args:
    module: A tf.Module.
    backend_info: BackendInfo with the details for compiling module to IREE.
    exported_names: Optional sequence representing the exported names to keep.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved. No compilation artifacts will be saved if this is not
      provided.

  Returns:
    A compiled IREE module blob and the path to the compiled VM FlatBuffer if
    artifacts_dir is provided.
  """

  def _compile_module(module, exported_names, backend_info, artifacts_dir):
    compiler_module = compiler.tf_module_to_compiler_module(module,
                                                            exported_names,
                                                            pass_pipeline=())
    return _incrementally_lower_compiler_module(compiler_module, backend_info,
                                                artifacts_dir)

  _compile_module = _setup_mlir_crash_reproducer(_compile_module, artifacts_dir,
                                                 backend_info.backend_id)
  return _compile_module(module, exported_names, backend_info, artifacts_dir)


class CompiledModule(object):
  """Base class for the TF and IREE compiled modules."""

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      compiled_paths: Dict[str, str],
  ):
    """Shared base constructor – not useful on its own.

    Args:
      module_name: A name for this compiled module. In most cases this will be
        the name of the tf.Module subclass or instance that is compiled.
      backend_info: BackendInfo with the details about compiling this module.
      compiled_paths: A dictionary mapping compiled method names to file paths
        corresponding to their serialized representations.
    """
    self.module_name = module_name
    self.backend_info = backend_info
    self.compiled_paths = compiled_paths

  def reinitialize(self):
    """Reinitializes all stateful variables."""
    raise NotImplementedError()

  @classmethod
  def create_from_class(cls,
                        module_class: Type[tf.Module],
                        backend_info: "BackendInfo",
                        exported_names: Sequence[str] = (),
                        artifacts_dir: str = None):
    """Compile a tf.Module subclass to the target backend in backend_info.

    Args:
      module_class: The tf.Module subclass to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    raise NotImplementedError()

  @classmethod
  def create_from_instance(cls,
                           module_instance: tf.Module,
                           backend_info: "BackendInfo",
                           exported_names: Sequence[str] = (),
                           artifacts_dir: str = None):
    """Compile a tf.Module instance to the target backend in backend_info.

    This is only implemented for IreeCompiledModule.

    Args:
      module_instance: The tf.Module instance to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    raise NotImplementedError()

  def iree_serializable(self):
    return False

  def tflite_serializable(self):
    return False


class _FunctionWrapper(object):

  def get_serialized_values(self) -> Tuple[Tuple[str], Tuple[str]]:
    """Dummy function to match _IreeFunctionWrapper's API."""
    return (), ()


class _IreeFunctionWrapper(_FunctionWrapper):
  """Wraps an IREE function, making it callable."""

  def __init__(self, context: rt.SystemContext, f: rt.system_api.BoundFunction):
    self._context = context
    self._f = f

  def __call__(self, *args):
    return self._f(*args)

  def get_serialized_values(self) -> Tuple[Tuple[str], Tuple[str]]:
    """Get cxx serialized inputs and outputs for this function."""
    return self._f.get_serialized_values()


class IreeCompiledModule(CompiledModule):
  """Iree compiled module."""

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      compiled_paths: Dict[str, str],
      vm_module: rt.VmModule,
      config: rt.Config,
  ):
    """Base constructor – Use one of the named constructors instead.

    Args:
      module_name: A name for this compiled module. In most cases this will be
        the name of the tf.Module subclass or instance that is compiled.
      backend_info: BackendInfo with the details about compiling this module.
      compiled_paths: A dictionary mapping compiled method names to file paths
        corresponding to their serialized representations.
      vm_module: A rt.VmModule containing compilation info to wrap.
      config: A rt.Config containing compilation info to wrap.
    """
    super().__init__(module_name, backend_info, compiled_paths)
    self._vm_module = vm_module
    self._config = config
    self.reinitialize()

  @classmethod
  def create_from_class(cls,
                        module_class: Type[tf.Module],
                        backend_info: "BackendInfo",
                        exported_names: Sequence[str] = (),
                        artifacts_dir: str = None):
    """Compile a tf.Module subclass to the target backend in backend_info.

    Args:
      module_class: The tf.Module subclass to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    set_random_seed()
    module_instance = module_class()
    return cls.create_from_instance(module_instance, backend_info,
                                    exported_names, artifacts_dir)

  @classmethod
  def create_from_instance(cls,
                           module_instance: tf.Module,
                           backend_info: "BackendInfo",
                           exported_names: Sequence[str] = (),
                           artifacts_dir: str = None):
    """Compile a tf.Module instance to the target backend in backend_info.

    Args:
      module_instance: The tf.Module instance to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    module_blob, compiled_path = _incrementally_compile_tf_module(
        module=module_instance,
        backend_info=backend_info,
        exported_names=exported_names,
        artifacts_dir=artifacts_dir)
    vm_module = rt.VmModule.from_flatbuffer(module_blob)
    config = rt.Config(driver_name=backend_info.driver)

    compiled_paths = None
    if compiled_path is not None:
      # IREE bundles every compiled method into the same compiled module.
      compiled_paths = collections.defaultdict(lambda: compiled_path)

    module_name = type(module_instance).__name__

    return cls(module_name, backend_info, compiled_paths, vm_module, config)

  def reinitialize(self):
    """Reinitializes all stateful variables."""
    # set_random_seed is not needed here because the model_class.__init__ is not
    # called.
    self._context = rt.SystemContext(modules=[self._vm_module],
                                     config=self._config)

  def __getattr__(self, attr: str) -> _IreeFunctionWrapper:
    # Try to resolve it as a function.
    m = self._context.modules[self._vm_module.name]
    f = m[attr]
    return _IreeFunctionWrapper(self._context, f)

  def iree_serializable(self) -> bool:
    return self.compiled_paths is not None


def _normalize_numpy(result: np.ndarray):
  """Normalizes TF and TFLite's outputs to match IREE's"""
  if np.isscalar(result):
    result = np.array(result)
  if result.dtype == np.bool:
    # IREE interprets bools as int8s, so we modify this for comparison.
    result = result.astype(dtype=np.int8)
  return result


class _TfFunctionWrapper(_FunctionWrapper):
  """Wraps a TF function, normalizing it to numpy."""

  def __init__(self, f: Callable[..., Any]):
    self._f = f

  def _convert_to_numpy(self, tensor: Any) -> Any:
    if not isinstance(tensor, tf.Tensor):
      return tensor
    return _normalize_numpy(tensor.numpy())

  def __call__(self, *args, **kwargs):
    # TensorFlow will auto-convert all inbound args.
    results = self._f(*args, **kwargs)
    # Then unmarshal them to numpy in the same way that the other backends do.
    # Handle single result (technically ambiguous with return of a tuple,
    # which is sad).
    if not isinstance(results, tuple):
      results = (results,)
    return tf.nest.map_structure(self._convert_to_numpy,
                                 *results,
                                 check_types=False)


class TfCompiledModule(CompiledModule):
  """TensorFlow 'compiled' module.

  This facade exists to provide a complimentary API to IreeCompiledModule and
  normalize TensorFlow's output to Numpy.
  """

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      constructor: Callable[[], tf.Module],
      exported_names: Sequence[str],
  ):
    """Base constructor – Use one of the named constructors instead.

    Args:
      module_name: A name for this compiled module. In most cases this will be
        the name of the tf.Module subclass or instance that is compiled.
      backend_info: BackendInfo with the details about compiling this module.
      constructor: A callable (class or function) which returns the tf.Module
        subclass instance to wrap.
      exported_names: an optional iterable of strings representing which of the
        tf.Module subclass instance's functions should be callable. If
        exported_names is empty then all functions will be callable.
    """
    super().__init__(module_name, backend_info, compiled_paths=None)
    self._constructor = constructor
    self._exported_names = exported_names
    self.reinitialize()

  @classmethod
  def create_from_class(cls,
                        module_class: Type[tf.Module],
                        backend_info: "BackendInfo",
                        exported_names: Sequence[str] = (),
                        artifacts_dir: str = None):
    """Compile a tf.Module subclass to the target backend in backend_info.

    Args:
      module_class: The tf.Module subclass to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    module_name = module_class.__name__
    constructor = module_class
    return cls(module_name, backend_info, constructor, exported_names)

  def reinitialize(self):
    """Reinitializes all stateful variables."""
    set_random_seed()
    self._tf_module = self._constructor()

  def __getattr__(self, attr: str) -> _TfFunctionWrapper:
    # Try to resolve it as a function.
    exported = not self._exported_names or attr in self._exported_names
    if not hasattr(self._tf_module, attr) or not exported:
      raise AttributeError(f"The TensorFlow module does not have attr '{attr}'")
    f = getattr(self._tf_module, attr)
    if not f or not hasattr(f, "__call__"):
      raise AttributeError(
          f"The TensorFlow module does not have a callable attr '{attr}'")
    return _TfFunctionWrapper(f)


def _get_non_inhereted_function_names(cls):
  """Gets all methods that cls has that its parents don't have."""
  names = set(dir(cls))
  for parent in cls.__bases__:
    names -= set(dir(parent))
  return list(names)


def _get_concrete_functions(module_class: Type[tf.Module],
                            exported_names: Sequence[str] = ()):
  """Get concrete functions from non-inherited methods or exported_names."""
  if not len(exported_names):
    # Get all method names on 'module_class' that aren't on 'tf.Module'.
    exported_names = _get_non_inhereted_function_names(module_class)
  instance = module_class()
  functions = []
  for name in exported_names:
    functions.append(instance.__getattribute__(name).get_concrete_function())
  return functions, exported_names


def tf_module_to_tflite_interpreters(
    module_class: Type[tf.Module],
    exported_names: Sequence[str] = (),
    artifacts_dir: str = None
) -> Tuple[Dict[str, tf.lite.Interpreter], Union[Dict[str, str]], None]:
  """Compile a tf.Module to TFLite interpreters for each of its methods.

  Args:
    module_class: A tf.Module subclass to compile with TFLite. If module_class
      has an attr get_legacy_tflite_saved_model_converter_kwargs then it will
      be compiled using tf.compat.v1.lite. It's best not to use this, however.
    exported_names: an optional iterable of strings representing which of the
      module_class's functions should be callable. If exported_names is empty
      then all functions will be callable.
    artifacts_dir: an optional path to save compilation artifacts to.

  Returns:
    A dictionary of function names to TFLite interpreters and a dictionary of
    function names to compiled tflite graph paths (or None if artifacts_dir)
    is None.
  """
  interpreters = dict()
  compiled_paths = None
  if artifacts_dir is not None:
    compiled_paths = dict()

  def _interpret_bytes(tflite_module: bytes, base_dir: str):
    """Save compiled TFLite module bytes and convert into an interpreter."""
    tflite_dir = os.path.join(base_dir, "tflite")
    os.makedirs(tflite_dir, exist_ok=True)
    tflite_path = os.path.join(tflite_dir, f"{name}.tflite")
    with open(tflite_path, "wb") as f:
      f.write(tflite_module)

    interpreters[name] = tf.lite.Interpreter(tflite_path)
    if artifacts_dir is not None:
      compiled_paths[name] = tflite_path

  # Convert module_class's methods into TFLite module byte-strings.
  tflite_modules = []
  functions, names = _get_concrete_functions(module_class, exported_names)
  for function in functions:
    converter = tf.lite.TFLiteConverter.from_concrete_functions([function])
    tflite_modules.append(converter.convert())

  # Load each of the converted methods above into tf.lite.Interpreters.
  for name, tflite_module in zip(names, tflite_modules):
    if artifacts_dir is None:
      with tempfile.TemporaryDirectory() as base_dir:
        _interpret_bytes(tflite_module, base_dir)
    else:
      _interpret_bytes(tflite_module, artifacts_dir)

  return interpreters, compiled_paths


class _TfLiteFunctionWrapper(_FunctionWrapper):
  """Wraps a TFLite interpreter and makes it behave like a python function."""

  def __init__(self, interpreter: tf.lite.Interpreter):
    self._interpreter = interpreter

  def __call__(self, *args, **kwargs) -> Tuple[Any]:
    if len(kwargs):
      raise ValueError("kwargs are not supported, but the following kwargs "
                       f"were provided {kwargs}")

    # Set up and run the function.
    self._interpreter.allocate_tensors()
    for arg, detail in zip(args, self._interpreter.get_input_details()):
      self._interpreter.set_tensor(detail["index"], arg)
    self._interpreter.invoke()

    # Extract the outputs from the TFLite interpreter.
    outputs = []
    is_dict = False
    for detail in self._interpreter.get_output_details():
      value = _normalize_numpy(self._interpreter.get_tensor(detail["index"]))
      name = detail["name"]
      if name != "Identity":
        # If the name of any output is "Identity" then we expect the entire
        # output to be a single array or tuple of arrays.
        if len(outputs) and not is_dict:
          raise ValueError(
              f"Encountered a named output '{name}' after {len(outputs)} "
              "non-named outputs")
        is_dict = True
        outputs.append([name, value])
      else:
        outputs.append(value)

    # Process them to match the output of the tf.Module.
    if not is_dict:
      outputs = tuple(outputs)
      if len(outputs) == 1:
        outputs = outputs[0]
    else:
      outputs = dict(outputs)
    return outputs


class TfLiteCompiledModule(CompiledModule):
  """Compiles a tf.Module with TFLite and allows it to be called."""

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      compiled_paths: Dict[str, str],
      interpreters: Dict[str, tf.lite.Interpreter],
  ):
    """Base constructor – Use one of the named constructors instead.

    Args:
      module_name: A name for this compiled module. In most cases this will be
        the name of the tf.Module subclass or instance that is compiled.
      backend_info: BackendInfo with the details about compiling this module.
      compiled_paths: A dictionary mapping compiled method names to file paths
        corresponding to their serialized representations.
      interpreters: A dict of tf.lite.Interpreters to make callable.
    """
    super().__init__(module_name, backend_info, compiled_paths)
    self._interpreters = interpreters

  @classmethod
  def create_from_class(cls,
                        module_class: Type[tf.Module],
                        backend_info: "BackendInfo",
                        exported_names: Sequence[str] = (),
                        artifacts_dir: str = None):
    """Compile a tf.Module subclass to the target backend in backend_info.

    Args:
      module_class: The tf.Module subclass to compile.
      backend_info: BackendInfo with the details for compiling module to IREE.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    set_random_seed()
    interpreters, compiled_paths = tf_module_to_tflite_interpreters(
        module_class, exported_names, artifacts_dir)
    module_name = module_class.__name__
    return cls(module_name, backend_info, compiled_paths, interpreters)

  def reinitialize(self):
    """Reinitializes all stateful variables."""
    # This is a noop because TFLite (mostly) doesn't support stateful modules.
    pass

  def __getattr__(self, attr: str) -> _TfLiteFunctionWrapper:
    # Try to resolve it as an interpreter.
    if not attr in self._interpreters:
      raise AttributeError(
          f"The TFLite module does not have an interpreter for '{attr}'")
    return _TfLiteFunctionWrapper(self._interpreters[attr])

  def tflite_serializable(self) -> bool:
    return self.compiled_paths is not None


class BackendInfo:
  """Contains information for compiling the specified backend."""

  _name_to_info = {
      "tf": {
          "compiled_module_class": TfCompiledModule,
          "driver": None,
          "compiler_targets": None,
      },
      "tflite": {
          "compiled_module_class": TfLiteCompiledModule,
          "driver": None,
          "compiler_targets": None,
      },
      "iree_vmla": {
          "compiled_module_class": IreeCompiledModule,
          "driver": "vmla",
          "compiler_targets": ["vmla"]
      },
      "iree_llvmjit": {
          "compiled_module_class": IreeCompiledModule,
          "driver": "llvm",
          "compiler_targets": ["llvm-ir"]
      },
      "iree_vulkan": {
          "compiled_module_class": IreeCompiledModule,
          "driver": "vulkan",
          "compiler_targets": ["vulkan-*"]
      },
  }

  def __init__(self, backend_name: str, backend_id: str = None):
    """Creates a BackendInfo with the compilation details for backend_name.

    Args:
      backend_name: a str specifying which backend to use. Should be one of
        'tf', 'iree_vmla', 'iree_llvmjit', 'iree_vulkan'.
      backend_id: an optional str specifying what name to use when saving
        compiled artifacts. Must satisfy `backend_id.startswith(backend_name)`.

    Raises:
      KeyError: if backend_name is not one of ['tf', 'iree_vmla',
      'iree_llvmjit', 'iree_vulkan'].
      ValueError: if backend_id doesn't start with backend_name.
    """
    if backend_name not in self._name_to_info:
      raise KeyError(
          "Expected backend_name to be one of "
          f"{list(self._name_to_info.keys())} but got '{backend_name}'.")
    if backend_id is not None and not backend_id.startswith(backend_name):
      raise ValueError(f"Expected backend_id to start with '{backend_name}' "
                       f"but got '{backend_id}'.")

    self.backend_name = backend_name
    self.backend_id = backend_name if backend_id is None else backend_id

    info = self._name_to_info[backend_name]
    self._compiled_module_class = info["compiled_module_class"]
    self.driver = info["driver"]
    self.compiler_targets = info["compiler_targets"]

  def compile_from_class(self,
                         module_class: Type[tf.Module],
                         exported_names: Sequence[str] = (),
                         artifacts_dir: str = None) -> CompiledModule:
    """Creates a 'CompiledModule' for this backend."""
    return self._compiled_module_class.create_from_class(
        module_class, self, exported_names, artifacts_dir)

  @classmethod
  def get_all_backends(cls) -> Sequence["BackendInfo"]:
    """Returns a list of all BackendInfo configurations."""
    return [BackendInfo(backend_name) for backend_name in cls._name_to_info]
