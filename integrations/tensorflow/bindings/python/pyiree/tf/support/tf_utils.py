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
from typing import Any, Callable, Dict, Sequence, Set, Tuple, Type, Union

from absl import logging
import numpy as np
from pyiree import rt
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf


def set_random_seed(seed: int = 0) -> None:
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


InputGeneratorType = Callable[[Sequence[int], Union[tf.DType, np.dtype]],
                              np.ndarray]


def uniform(shape: Sequence[int],
            dtype: Union[tf.DType, np.dtype] = np.float32,
            low: float = -1.0,
            high: float = 1.0) -> np.ndarray:
  """np.random.uniform with simplified API and dtype and bool support."""
  dtype = dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
  if dtype == np.bool:
    return np.random.choice(2, shape).astype(np.bool)
  else:
    values = np.random.uniform(size=shape, low=low, high=high)
    if np.issubdtype(dtype, np.integer):
      values = np.round(values)
    return values.astype(dtype)


def ndarange(shape: Sequence[int],
             dtype: Union[tf.DType, np.dtype] = np.float32) -> np.ndarray:
  """np.ndarange for arbitrary input shapes."""
  dtype = dtype.as_numpy_dtype if isinstance(dtype, tf.DType) else dtype
  return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def random_permutation(
    shape: Sequence[int],
    dtype: Union[tf.DType, np.dtype] = np.float32) -> np.ndarray:
  """Returns a random permutation of 'np.prod(shape)' numbers."""
  values = ndarange(shape, dtype)
  np.random.shuffle(values)
  return values


def apply_function(values, function):
  """Applies 'function' recursively to the inputted values."""
  if isinstance(values, list):
    return [apply_function(v, function) for v in values]
  elif isinstance(values, tuple):
    return tuple([apply_function(v, function) for v in values])
  elif isinstance(values, dict):
    return {k: apply_function(v, function) for k, v in values.items()}
  else:
    return function(values)


def generate_inputs(
    spec,  # Union[Sequence[tf.TensorSpec], tf.TensorSpec]
    input_generator: InputGeneratorType,
) -> Sequence[np.ndarray]:
  """Generates inputs for a given input signature using 'input_generator'."""
  make_static = lambda shape: [dim if dim is not None else 2 for dim in shape]
  generate = lambda spec: input_generator(make_static(spec.shape), spec.dtype)
  return apply_function(spec, generate)


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
    function: Any,  # pytype doesn't support arbitrary Callable[*args, **kwargs]
    artifacts_dir: str,
    backend_id: str,
) -> Any:  # Callable[Any, Any]
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
  """Compile a TensorFlow tf.Module and optionally save compilation artifacts.

  The module blob this creates is not callable. See IreeCompiledModule for an
  API that returns a module that can be called without any further steps.

  See _incrementally_lower_compiler_module's docstring for details about which
  artifacts will be saved.

  Args:
    module: A tf.Module.
    backend_info: BackendInfo with the details for compiling this module.
    exported_names: Optional sequence representing the exported names to keep.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved. No compilation artifacts will be saved if this is not
      provided.

  Returns:
    A compiled IREE module blob and the path to the compiled VM FlatBuffer if
    artifacts_dir is provided.
  """

  def _compile_module(module, backend_info, exported_names, artifacts_dir):
    compiler_module = compiler.tf_module_to_compiler_module(module,
                                                            exported_names,
                                                            pass_pipeline=())
    return _incrementally_lower_compiler_module(compiler_module, backend_info,
                                                artifacts_dir)

  _compile_module = _setup_mlir_crash_reproducer(_compile_module, artifacts_dir,
                                                 backend_info.backend_id)
  return _compile_module(module, backend_info, exported_names, artifacts_dir)


def _incrementally_compile_tf_signature_def_saved_model(
    saved_model_dir: str, saved_model_tags: Set[str],
    backend_info: "BackendInfo", exported_name: str, artifacts_dir: str):
  """Compile a SignatureDef SavedModel and optionally save compilation artifacts.

  The module blob this creates is not callable. See IreeCompiledModule for an
  API that returns a module that can be called without any further steps.

  See _incrementally_lower_compiler_module's docstring for details about which
  artifacts will be saved.

  Args:
    saved_model_dir: Directory of the saved model.
    saved_model_tags: Optional set of tags to use when loading the model.
    backend_info: BackendInfo with the details for compiling the saved model.
    exported_name: A str representing the signature on the saved model to
      compile.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved. No compilation artifacts will be saved if this is not
      provided.

  Returns:
    A compiled IREE module blob and the path to the compiled VM FlatBuffer if
    artifacts_dir is provided.
  """

  def _compile_module(saved_model_dir, saved_model_tags, backend_info,
                      exported_name, artifacts_dir):
    # Convert the tf_module into raw TF input MLIR.
    compiler_module = compiler.tf_signature_def_saved_model_to_compiler_module(
        saved_model_dir, saved_model_tags, [exported_name], pass_pipeline=())
    return _incrementally_lower_compiler_module(compiler_module, backend_info,
                                                artifacts_dir)

  _compile_module = _setup_mlir_crash_reproducer(_compile_module, artifacts_dir,
                                                 backend_info.backend_id)
  return _compile_module(saved_model_dir, saved_model_tags, backend_info,
                         exported_name, artifacts_dir)


class _FunctionWrapper(object):

  def __call__(self, *args, **kwargs):
    raise NotImplementedError()

  def get_serialized_values(self) -> Tuple[Tuple[str], Tuple[str]]:
    """Dummy function to match _IreeFunctionWrapper's API."""
    return ("",), ("",)


class CompiledModule(object):
  """Base class for the TF and IREE compiled modules."""

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      compiled_paths: Union[Dict[str, str], None],
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
      backend_info: BackendInfo with the details for compiling this module.
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

  @classmethod
  def create_from_signature_def_saved_model(cls,
                                            saved_model_dir: str,
                                            saved_model_tags: Set[str],
                                            module_name: str,
                                            backend_info: "BackendInfo",
                                            exported_name: str,
                                            input_names: Sequence[str],
                                            output_names: Sequence[str],
                                            artifacts_dir: str = None):
    """Compile a SignatureDef SavedModel to the target backend in backend_info.

    Args:
      saved_model_dir: Directory of the saved model.
      saved_model_tags: Optional set of tags to use when loading the model.
      module_name: A name for this compiled module.
      backend_info: BackendInfo with the details for compiling the saved model.
      exported_name: A str representing the signature on the saved model to
        compile.
      input_names: A sequence of kwargs to feed to the saved model.
      output_names: A sequence of named outputs to extract from the saved model.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    raise NotImplementedError()

  def __getattr__(self, attr: str) -> _FunctionWrapper:
    raise NotImplementedError()

  def iree_serializable(self):
    return False

  def tflite_serializable(self):
    return False


class _IreeFunctionWrapper(_FunctionWrapper):
  """Wraps an IREE function, making it callable."""

  def __init__(self, context: rt.SystemContext, f: rt.system_api.BoundFunction):
    self._context = context
    self._f = f

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)

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

  @classmethod
  def create_from_signature_def_saved_model(cls,
                                            saved_model_dir: str,
                                            saved_model_tags: Set[str],
                                            module_name: str,
                                            backend_info: "BackendInfo",
                                            exported_name: str,
                                            input_names: Sequence[str],
                                            output_names: Sequence[str],
                                            artifacts_dir: str = None):
    """Compile a SignatureDef SavedModel to the target backend in backend_info.

    Args:
      saved_model_dir: Directory of the saved model.
      saved_model_tags: Optional set of tags to use when loading the model.
      module_name: A name for this compiled module.
      backend_info: BackendInfo with the details for compiling the saved model.
      exported_name: A str representing the signature on the saved model to
        compile.
      input_names: A sequence of kwargs to feed to the saved model.
      output_names: A sequence of named outputs to extract from the saved model.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    del input_names  # Unused.
    del output_names  # Unused.
    module_blob, compiled_path = _incrementally_compile_tf_signature_def_saved_model(
        saved_model_dir, saved_model_tags, backend_info, exported_name,
        artifacts_dir)
    vm_module = rt.VmModule.from_flatbuffer(module_blob)
    config = rt.Config(driver_name=backend_info.driver)

    compiled_paths = None
    if compiled_path is not None:
      # IREE bundles every compiled method into the same compiled module :)
      compiled_paths = collections.defaultdict(lambda: compiled_path)

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


def _convert_to_numpy(tensor: Any) -> Any:
  if not isinstance(tensor, tf.Tensor):
    return tensor
  return _normalize_numpy(tensor.numpy())


def convert_to_numpy(values: Any) -> Any:
  """Converts any tf.Tensor in values to numpy."""
  return apply_function(values, _convert_to_numpy)


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
    return convert_to_numpy(results)


def _convert_inputs_to_tensors(function):

  def decorator(*args, **kwargs):
    args = [tf.convert_to_tensor(arg) for arg in args]
    kwargs = {k: tf.convert_to_tensor(v) for k, v in kwargs.items()}
    return function(*args, **kwargs)

  return decorator


class SignatureDefSavedModelWrapper(object):
  """Wraps a SavedModel to imitate a tf.Module with a method 'exported_name'."""

  def __init__(self, saved_model_dir: str, saved_model_tags: Set[str],
               exported_name: str):
    self.saved_model = tf.saved_model.load(saved_model_dir,
                                           tags=saved_model_tags)
    inference_func = self.saved_model.signatures[exported_name]
    inference_func = _convert_inputs_to_tensors(inference_func)
    self.__setattr__(exported_name, inference_func)


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
      backend_info: BackendInfo with the details for compiling this module.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    module_name = module_class.__name__
    constructor = module_class
    return cls(module_name, backend_info, constructor, exported_names)

  @classmethod
  def create_from_signature_def_saved_model(cls,
                                            saved_model_dir: str,
                                            saved_model_tags: Set[str],
                                            module_name: str,
                                            backend_info: "BackendInfo",
                                            exported_name: str,
                                            input_names: Sequence[str],
                                            output_names: Sequence[str],
                                            artifacts_dir: str = None):
    """Compile a SignatureDef SavedModel to the target backend in backend_info.

    Args:
      saved_model_dir: Directory of the saved model.
      saved_model_tags: Optional set of tags to use when loading the model.
      module_name: A name for this compiled module.
      backend_info: BackendInfo with the details for compiling the saved model.
      exported_name: A str representing the signature on the saved model to
        compile.
      input_names: A sequence of kwargs to feed to the saved model.
      output_names: A sequence of named outputs to extract from the saved model.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    constructor = lambda: SignatureDefSavedModelWrapper(
        saved_model_dir, saved_model_tags, exported_name)
    return cls(module_name, backend_info, constructor, [exported_name])

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
    functions.append(getattr(instance, name).get_concrete_function())
  return functions, exported_names


def tf_module_to_tflite_module_bytes(
    module_class: Type[tf.Module], exported_names: Sequence[str] = ()
) -> Dict[str, bytes]:
  """Compiles a tf.Module's methods with TFLite.

  Args:
    module_class: A tf.Module subclass to compile with TFLite.
    exported_names: an optional iterable of strings representing which of the
      module_class's functions should be compiled. If exported_names is empty
      then all functions will be compiled.

  Returns:
    A dict mapping method names to compiled TFLite module bytes.
  """
  tflite_modules = []
  methods, method_names = _get_concrete_functions(module_class, exported_names)
  for method in methods:
    converter = tf.lite.TFLiteConverter.from_concrete_functions([method])
    tflite_modules.append(converter.convert())
  return dict(zip(method_names, tflite_modules))


def tf_signature_def_saved_model_to_tflite_module_bytes(
    saved_model_dir: str,
    saved_model_tags: Set[str],
    exported_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> Dict[str, bytes]:
  """Compiles a SignatureDef SavedModel signature with TFLite.

  Args:
    saved_model_dir: Directory of the saved model.
    saved_model_tags: Optional set of tags to use when loading the model.
    exported_name: A str representing the signature on the saved model to
      compile.
    input_names: A sequence of kwargs to feed to the saved model.
    output_names: A sequence of named outputs to extract from the saved model.

  Returns:
    A dict mapping the signature name to the compiled TFLite module bytes.
  """
  converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
      saved_model_dir,
      tag_set=saved_model_tags,
      signature_key=exported_name,
      input_arrays=input_names,
      output_arrays=output_names)
  tflite_module = converter.convert()
  return dict([[exported_name, tflite_module]])


def tflite_module_bytes_to_tflite_interpreters(
    tflite_module_bytes: Dict[str, bytes],
    artifacts_dir: str = None
) -> Tuple[Dict[str, tf.lite.Interpreter], Union[Dict[str, str], None]]:
  """Compile a dict of TFLite compiled bytes to  TFLite interpreters.

  Args:
    tflite_module_bytes: A dict mapping method names to compiled TFLite byte
      strings.
    artifacts_dir: an optional path to save compilation artifacts to.

  Returns:
    A dictionary mapping method names to TFLite interpreters and a dictionary
    mapping method names to compiled tflite graph paths (or None if
    artifacts_dir is None).
  """
  interpreters = dict()
  compiled_paths = None
  if artifacts_dir is not None:
    compiled_paths = dict()

  def _interpret_bytes(method_name: str, tflite_module: bytes, base_dir: str):
    """Save compiled TFLite module bytes and convert into an interpreter."""
    tflite_dir = os.path.join(base_dir, "tflite")
    os.makedirs(tflite_dir, exist_ok=True)
    tflite_path = os.path.join(tflite_dir, f"{method_name}.tflite")
    with open(tflite_path, "wb") as f:
      f.write(tflite_module)

    interpreters[method_name] = tf.lite.Interpreter(tflite_path)
    if artifacts_dir is not None:
      compiled_paths[method_name] = tflite_path

  # Load each of the converted methods above into tf.lite.Interpreters.
  for method_name, tflite_module in tflite_module_bytes.items():
    if artifacts_dir is None:
      with tempfile.TemporaryDirectory() as base_dir:
        _interpret_bytes(method_name, tflite_module, base_dir)
    else:
      _interpret_bytes(method_name, tflite_module, artifacts_dir)

  return interpreters, compiled_paths


class _TfLiteFunctionWrapper(_FunctionWrapper):
  """Wraps a TFLite interpreter and makes it behave like a python function."""

  def __init__(self, interpreter: tf.lite.Interpreter,
               output_names: Sequence[str]):
    self._interpreter = interpreter
    self._output_names = output_names

  def __call__(self, *args,
               **kwargs) -> Union[Dict[str, Any], Tuple[Any], np.ndarray]:
    if len(args) and len(kwargs):
      raise ValueError("Passing both args and kwargs is not supported by "
                       "_TfLiteFunctionWrapper")

    # Set up and run the function.
    self._interpreter.allocate_tensors()

    if len(args):
      # Specifically to get TFLite to work with keras models that take a list of
      # inputs instead of a sequence of args as their inputs, because it decides
      # to change the input signature but it still technically works if you
      # ignore that it does that.
      if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

      for arg, detail in zip(args, self._interpreter.get_input_details()):
        self._interpreter.set_tensor(detail["index"], arg)
    else:
      for detail in self._interpreter.get_input_details():
        self._interpreter.set_tensor(detail["index"], kwargs[detail["name"]])

    self._interpreter.invoke()

    # Extract the outputs from the TFLite interpreter.
    outputs = []
    for detail in self._interpreter.get_output_details():
      value = _normalize_numpy(self._interpreter.get_tensor(detail["index"]))
      if self._output_names is not None:
        name = detail["name"]
        if name not in self._output_names:
          raise ValueError(f"Expected '{name}' to be in {self._output_names}")
        outputs.append([detail["name"], value])
      else:
        outputs.append(value)

    # Process them to match the output of the tf.Module.
    if self._output_names is not None:
      return dict(outputs)
    else:
      if len(outputs) == 1:
        return outputs[0]
      return tuple(outputs)


class TfLiteCompiledModule(CompiledModule):
  """Compiles a tf.Module with TFLite and allows it to be called."""

  def __init__(
      self,
      module_name: str,
      backend_info: "BackendInfo",
      compiled_paths: Dict[str, str],
      interpreters: Dict[str, tf.lite.Interpreter],
      output_names: Sequence[str] = None,
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
    self._output_names = output_names

  @classmethod
  def create_from_class(cls,
                        module_class: Type[tf.Module],
                        backend_info: "BackendInfo",
                        exported_names: Sequence[str] = (),
                        artifacts_dir: str = None):
    """Compile a tf.Module subclass to the target backend in backend_info.

    Args:
      module_class: The tf.Module subclass to compile.
      backend_info: BackendInfo with the details for compiling this module.
      exported_names: Optional sequence representing the exported names to keep.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    set_random_seed()
    tflite_module_bytes = tf_module_to_tflite_module_bytes(
        module_class, exported_names)
    interpreters, compiled_paths = tflite_module_bytes_to_tflite_interpreters(
        tflite_module_bytes, artifacts_dir)
    module_name = module_class.__name__
    return cls(module_name, backend_info, compiled_paths, interpreters)

  @classmethod
  def create_from_signature_def_saved_model(cls,
                                            saved_model_dir: str,
                                            saved_model_tags: Set[str],
                                            module_name: str,
                                            backend_info: "BackendInfo",
                                            exported_name: str,
                                            input_names: Sequence[str],
                                            output_names: Sequence[str],
                                            artifacts_dir: str = None):
    """Compile a SignatureDef SavedModel to the target backend in backend_info.

    Args:
      saved_model_dir: Directory of the saved model.
      saved_model_tags: Optional set of tags to use when loading the model.
      module_name: A name for this compiled module.
      backend_info: BackendInfo with the details for compiling the saved model.
      exported_name: A str representing the signature on the saved model to
        compile.
      input_names: A sequence of kwargs to feed to the saved model.
      output_names: A sequence of named outputs to extract from the saved model.
      artifacts_dir: An optional string pointing to where compilation artifacts
        should be saved. No compilation artifacts will be saved if this is not
        provided.
    """
    tflite_module_bytes = tf_signature_def_saved_model_to_tflite_module_bytes(
        saved_model_dir, saved_model_tags, exported_name, input_names,
        output_names)
    interpreters, compiled_paths = tflite_module_bytes_to_tflite_interpreters(
        tflite_module_bytes, artifacts_dir)
    return cls(module_name, backend_info, compiled_paths, interpreters,
               output_names)

  def reinitialize(self):
    """Reinitializes all stateful variables."""
    # This is a noop because TFLite (mostly) doesn't support stateful modules.
    pass

  def __getattr__(self, attr: str) -> _TfLiteFunctionWrapper:
    # Try to resolve it as an interpreter.
    if not attr in self._interpreters:
      raise AttributeError(
          f"The TFLite module does not have an interpreter for '{attr}'")
    return _TfLiteFunctionWrapper(self._interpreters[attr], self._output_names)

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

  def compile_signature_def_saved_model(
      self,
      saved_model_dir: str,
      saved_model_tags: Set[str],
      module_name: str,
      exported_name: str,
      input_names: Sequence[str],
      output_names: Sequence[str],
      artifacts_dir: str = None) -> CompiledModule:
    return self._compiled_module_class.create_from_signature_def_saved_model(
        saved_model_dir, saved_model_tags, module_name, self, exported_name,
        input_names, output_names, artifacts_dir)

  @classmethod
  def get_all_backends(cls) -> Sequence["BackendInfo"]:
    """Returns a list of all BackendInfo configurations."""
    return [BackendInfo(backend_name) for backend_name in cls._name_to_info]
