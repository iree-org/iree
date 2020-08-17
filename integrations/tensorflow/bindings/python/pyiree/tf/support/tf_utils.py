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

import os
import random
import re
import tempfile
from typing import Any, Callable, Sequence, Type

from absl import flags
from absl import logging
import numpy as np
from pyiree import rt
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf

flags.DEFINE_bool("keep_saved_model", False,
                  "Keep the SavedModel used by compile_tf_module on disk.")
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
  """Returns a string that denotes the type `dtype` in MLIR style."""
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
  """Saves input values with IREE tools format if `artifacts_dir` is set."""
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


def backends_to_str(backend_infos: Sequence["BackendInfo"]) -> str:
  """Creates a normalized string representing the provided backends."""
  normalized_names = []
  for backend_info in backend_infos:
    # Remove unusual characters and ensure names don't end or start in "_".
    name = re.sub("[^0-9a-zA-Z_]+", "_", backend_info.name)
    normalized_names.append(name.strip("_"))
  return "__".join(normalized_names)


def _get_backends_path(artifact_name: str,
                       backend_infos: Sequence["BackendInfo"],
                       artifacts_dir: str) -> str:
  """Gets the path to save artifact_name under for the specified backend(s)."""
  backends_string = backends_to_str(backend_infos)
  # Put the artifact in a directory if there's only one backend.
  if len(backend_infos) == 1:
    backend_dir = os.path.join(artifacts_dir, backends_string)
    os.makedirs(backend_dir, exist_ok=True)
    return os.path.join(artifacts_dir, backends_string, artifact_name)
  else:
    return os.path.join(artifacts_dir, f"{artifact_name}__{backends_string}")


def compile_tf_module(tf_module: Type[tf.Module],
                      backend_infos: Sequence["BackendInfo"] = (),
                      exported_names: Sequence[str] = (),
                      artifacts_dir: str = None) -> compiler.binding.OpaqueBlob:
  """Compiles a TensorFlow tf.Module and optionally saves compilation artifacts.

  The artifact this creates is not callable. See IreeCompiledModule for an API
  that returns a module that can be called without any further steps.

  If artifacts_dir is provided then the following artifacts will be saved:
    backend_name/saved_model:
      A TF SavedModel directory containing the files used translate the
      tf.Module into an IREE module. Only saved if '--keep_saved_model=True'.
    tf_input.mlir:
      MLIR for the module in TF's input dialect.
    iree_input.mlir:
      The MLIR above translated to IREE via compiler.TF_IMPORT_PASS_PIPELINE.
    backend_name/compiled.vmfb:
      A VM FlatBuffer compiled to the target backends from the IREE MLIR above.

  If multiple backends are specified, then instead of saving the SavedModel and
  compiled 'vmfb' under 'backend_name/', they will be saved as follows:
    - 'saved_model__{backends}'
    - 'compiled__{backends}.vmfb'
  where 'backends' is a '__' delimited list (e.g. iree_vmla__iree_llvmjit).

  Args:
    tf_module: A tf.Module.
    backend_infos: Iterable of BackendInfo names to compile for.
    exported_names: Iterable of dotted function names to consider for
      compilation.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved.

  Returns:
    A compiled IREE module blob.
  """

  def _compile_from_path(sm_path: str) -> compiler.binding.OpaqueBlob:
    """Helper function for compile_tf_module."""
    if artifacts_dir is not None:
      # Set up a crash reproducer for debugging.
      compiler.Context.default_crash_reproducer_path = os.path.join(
          artifacts_dir, f"reproducer__{backends_string}.mlir")
    try:
      # We break up the compilation here so we can save intermediary artifacts.
      compiler_context = compiler.Context()

      # Convert the tf_module into raw TF input MLIR.
      compiler_module = compiler.tf_load_saved_model(
          sm_path,
          exported_names=exported_names,
          compiler_context=compiler_context,
          pass_pipeline=())

      if artifacts_dir is not None:
        tf_mlir_path = os.path.join(artifacts_dir, "tf_input.mlir")
        logging.info("Saving raw TF input MLIR to: %s", tf_mlir_path)
        with open(tf_mlir_path, "w") as f:
          f.write(compiler_module.to_asm())

      # Now run the passes manually that tf_load_saved_model would usually do.
      compiler_module.run_pass_pipeline(compiler.TF_IMPORT_PASS_PIPELINE)

      if artifacts_dir is not None:
        iree_mlir_path = os.path.join(artifacts_dir, "iree_input.mlir")
        logging.info("Saving IREE input MLIR to: %s", iree_mlir_path)
        with open(iree_mlir_path, "w") as f:
          f.write(compiler_module.to_asm())

      target_backends = []
      for backend_info in backend_infos:
        target_backends.extend(backend_info.compiler_targets)
      compiled_module = compiler_module.compile(target_backends=target_backends)

      if artifacts_dir is not None:
        compiled_path = _get_backends_path("compiled", backend_infos,
                                           artifacts_dir)
        compiled_path = f"{compiled_path}.vmfb"
        logging.info("Saving compiled IREE module to: %s", compiled_path)
        with open(compiled_path, "wb") as f:
          f.write(compiled_module)

      return compiled_module
    except Exception:  # pylint: disable=broad-except
      if artifacts_dir is not None:
        # Disable the crash reproducer (to avoid inadvertently overwriting it).
        compiler.Context.default_crash_reproducer_path = None
      raise

  options = tf.saved_model.SaveOptions(save_debug_info=True)
  backends_string = backends_to_str(backend_infos)
  if artifacts_dir is not None and FLAGS.keep_saved_model:
    # Create a saved model for these target backends to avoid a race condition
    # when running a test suite.
    # TODO(meadowlark): Remove this once we have a TfLiteCompiledModule.
    sm_path = _get_backends_path("saved_model", backend_infos, artifacts_dir)
    tf.saved_model.save(tf_module, sm_path, options=options)
    return _compile_from_path(sm_path)
  else:
    # Round-trip the saved model through a temporary directory.
    with tempfile.TemporaryDirectory() as sm_path:
      tf.saved_model.save(tf_module, sm_path, options=options)
      return _compile_from_path(sm_path)


class CompiledModule(object):
  """Base class for the TF and IREE compiled modules."""

  def __init__(self, module_class: Type[tf.Module], backend_info: "BackendInfo",
               exported_names: Sequence[str], artifacts_dir: str):
    """Shared base constructor – not useful on its own."""
    self._module_class = module_class
    self._backend_info = backend_info
    self._exported_names = exported_names
    self._artifacts_dir = artifacts_dir

    # Public attributes:
    self.backend = self._backend_info.name
    self.module_name = self._module_class.__name__

  def create_reinitialized(self):
    """Duplicates this module with its initial state without recompiling."""
    raise NotImplementedError()


class _IreeFunctionWrapper(object):
  """Wraps an IREE function, making it callable."""

  def __init__(self, context: rt.SystemContext, f: rt.system_api.BoundFunction):
    self._context = context
    self._f = f

  def __call__(self, *args):
    return self._f(*args)


class IreeCompiledModule(CompiledModule):
  """Iree compiled module."""

  def __init__(self,
               module_class: Type[tf.Module],
               backend_info: "BackendInfo",
               exported_names: Sequence[str] = (),
               artifacts_dir: str = None,
               _create_reinitialized_args: Sequence[Any] = None):
    """Compile a tf.Module to the target backend in backend_info.

    Args:
      module_class: the tf.Module subclass to compile.
      backend_info: an element of BackendInfo corresponding to the IREE backend
        to compile to.
      exported_names: an optional iterable of strings representing which of the
        module_class's functions to compile. If exported_names is empty all
        functions will be compiled.
      artifacts_dir: an optional path to save compilation artifacts to.
      _create_reinitialized_args: used internally.
    """
    super().__init__(module_class, backend_info, exported_names, artifacts_dir)

    if _create_reinitialized_args is None:
      set_random_seed()
      self._module_blob = compile_tf_module(
          tf_module=module_class(),
          backend_infos=[backend_info],
          exported_names=exported_names,
          artifacts_dir=artifacts_dir)
      self._module = rt.VmModule.from_flatbuffer(self._module_blob)
      self._config = rt.Config(driver_name=backend_info.driver)
    else:
      # Called from self.create_reinitialized()
      self._module_blob, self._module, self._config = _create_reinitialized_args

    # Holds all of the module's mutable state.
    self._context = rt.SystemContext(
        modules=[self._module], config=self._config)

  def create_reinitialized(self) -> "IreeCompiledModule":
    """Duplicates this module with its initial state without recompiling."""
    default_args = [
        self._module_class, self._backend_info, self._exported_names,
        self._artifacts_dir
    ]
    create_reinitialized_args = [self._module_blob, self._module, self._config]
    return IreeCompiledModule(*default_args, create_reinitialized_args)

  def __getattr__(self, attr: str) -> _IreeFunctionWrapper:
    # Try to resolve it as a function.
    m = self._context.modules[self._module.name]
    f = m[attr]
    return _IreeFunctionWrapper(self._context, f)


class _TfFunctionWrapper(object):
  """Wraps a TF function, normalizing it to numpy."""

  def __init__(self, f: Callable[..., Any]):
    self._f = f

  def _convert_to_numpy(self, tensor: Any) -> Any:
    if not isinstance(tensor, tf.Tensor):
      return tensor
    result = tensor.numpy()
    if np.isscalar(result):
      # convert_to_tensor isn't reversible via .numpy()
      result = np.array(result)
    return result

  def __call__(self, *args, **kwargs):
    # TensorFlow will auto-convert all inbound args.
    results = self._f(*args, **kwargs)
    # Then unmarshal them to numpy in the same way that the other backends do.
    # Handle single result (technically ambiguous with return of a tuple,
    # which is sad).
    if not isinstance(results, tuple):
      results = (results,)
    return tf.nest.map_structure(
        self._convert_to_numpy, *results, check_types=False)


class TfCompiledModule(CompiledModule):
  """TensorFlow 'compiled' module.

  This facade exists to provide a complimentary API to IreeCompiledModule and
  normalize TensorFlow's output to Numpy.
  """

  def __init__(self,
               module_class: Type[tf.Module],
               backend_info: "BackendInfo",
               exported_names: Sequence[str] = (),
               artifacts_dir: str = None):
    """Wrap a tf.Module in a TFCompiledModule facade.

    Args:
      module_class: the tf.Module subclass to 'compile'.
      backend_info: one of the 'tf*' elements in BackendInfo.
      exported_names: an optional iterable of strings representing which of the
        module_class's functions should be callable. If exported_names is empty
        then all functions will be callable.
      artifacts_dir: an optional path to save compilation artifacts to. Has no
        effect for this subclass as nothing is compiled.
    """
    super().__init__(module_class, backend_info, exported_names, artifacts_dir)
    set_random_seed()
    self._tf_module = module_class()

  def create_reinitialized(self) -> "TfCompiledModule":
    """Duplicates this module with the starting state of module_class."""
    return TfCompiledModule(self._module_class, self._backend_info,
                            self._exported_names, self._artifacts_dir)

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


class BackendInfo:
  """Contains information for compiling the specified backend."""

  _name_to_info = {
      "tf": {
          "compiled_module_class": TfCompiledModule,
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

  def __init__(self, backend_name: str, artifact_name: str = None):
    """Creates a BackendInfo with the compilation details for backend_name.

    Args:
      backend_name: a str specifying which backend to use. Should be one of
        'tf', 'iree_vmla', 'iree_llvmjit', 'iree_vulkan'.
      artifact_name: an optional str specifying what name to use when saving
        compiled artifacts.

    Raises:
      KeyError: if backend_name is not one of ['tf', 'iree_vmla',
      'iree_llvmjit', 'iree_vulkan'].
    """
    if backend_name not in self._name_to_info:
      raise KeyError(
          "Expected backend_name to be one of "
          f"{list(self._name_to_info.keys())} but got '{backend_name}'.")
    info = self._name_to_info[backend_name]
    self._compiled_module_class = info["compiled_module_class"]
    self.driver = info["driver"]
    self.compiler_targets = info["compiler_targets"]
    self.name = backend_name if artifact_name is None else artifact_name

  def compile(self,
              module: Type[tf.Module],
              exported_names: Sequence[str] = (),
              artifacts_dir: str = None) -> CompiledModule:
    """Creates a `CompiledModule` for this backend."""
    return self._compiled_module_class(module, self, exported_names,
                                       artifacts_dir)

  @classmethod
  def get_all_backends(cls) -> Sequence["BackendInfo"]:
    """Returns a list of all BackendInfo configurations."""
    return [BackendInfo(backend_name) for backend_name in cls._name_to_info]
