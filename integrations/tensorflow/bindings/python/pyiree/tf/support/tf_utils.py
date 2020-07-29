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
import copy
import os
import random
import re
import sys
import tempfile

from absl import flags
from absl import logging
import numpy as np
from pyiree import rt
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS
NUMPY_LINEWIDTH = 120


def set_random_seed(seed=0):
  """Set random seed for tf, np and random."""
  tf.random.set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def backends_to_str(target_backends):
  """Creates a flattened and normalized string representing target_backends."""
  normalized_backends = []
  for backend in target_backends:
    # Remove unusual characters and ensure names don't end or start in "_".
    backend = re.sub("[^0-9a-zA-Z_]+", "_", backend)
    normalized_backends.append(backend.strip("_"))
  return "__".join(normalized_backends)


def compile_tf_module(tf_module,
                      target_backends=(),
                      exported_names=(),
                      artifacts_dir=None):
  """Compiles a TensorFlow tf.Module and optionally saves compilation artifacts.

  The artifact this creates is not callable. See IreeCompiledModule for an API
  that returns a module that can be called without any further steps.

  If artifacts_dir is provided then the following artifacts will be saved:
    saved_model:
      A TF SavedModel directory containing the files used translate the
      tf.Module into an IREE module.
    tf_input.mlir:
      MLIR for the module in TF's input dialect.
    iree_input.mlir:
      The MLIR above translated to IREE via compiler.TF_IMPORT_PASS_PIPELINE.
    compiled__backends.vmfb:
      A VM FlatBuffer compiled to the target backends from the IREE MLIR above.
  Here 'backends' is a '__' delimited list of iree backends (e.g. vmla__llvm_ir)

  Args:
    tf_module: A tf.Module.
    target_backends: Iterable of string backend names to compile for.
    exported_names: Iterable of dotted function names to consider for
      compilation.
    artifacts_dir: An optional string pointing to where compilation artifacts
      should be saved.

  Returns:
    A compiled IREE module blob.
  """

  def _compile_from_path(sm_path):
    """Helper function for compile_tf_module."""
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

    compiled_module = compiler_module.compile(target_backends=target_backends)
    if artifacts_dir is not None:
      compiled_name = f"compiled__{backends_to_str(target_backends)}.vmfb"
      compiled_path = os.path.join(artifacts_dir, compiled_name)
      logging.info("Saving compiled IREE module to: %s", compiled_path)
      with open(compiled_path, "wb") as f:
        f.write(compiled_module)

    return compiled_module

  options = tf.saved_model.SaveOptions(save_debug_info=True)
  if artifacts_dir is not None:
    # Save the saved model alongside the other compilation artifacts.
    sm_path = os.path.join(artifacts_dir, "saved_model")
    tf.saved_model.save(tf_module, sm_path, options=options)
    return _compile_from_path(sm_path)
  else:
    # Round-trip the saved model through a temporary directory.
    with tempfile.TemporaryDirectory() as sm_path:
      tf.saved_model.save(tf_module, sm_path, options=options)
      return _compile_from_path(sm_path)


class CompiledModule(object):
  """Base class for the TF and IREE compiled modules."""

  def __init__(self, module_class, backend_info, exported_names, artifacts_dir):
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


class IreeCompiledModule(CompiledModule):
  """Iree compiled module."""

  def __init__(self,
               module_class,
               backend_info,
               exported_names=[],
               artifacts_dir=None,
               _create_reinitialized_args=None):
    """Compile a tf.Module to the target backend in backend_info.

    Args:
      module_class: the tf.Module subclass to compile.
      backend_info: an element of BackendInfo corresponding to the IREE backend
        to compile to.
      exported_names: an optional iterable of strings representing which of the
        module_class's functions to compile. If exported_names is empty all
        functions will be compiled.
      artifacts_dir: an optional path to save compilation artifacts to.
    """
    super().__init__(module_class, backend_info, exported_names, artifacts_dir)

    if _create_reinitialized_args is None:
      self._module_blob = compile_tf_module(
          tf_module=module_class(),
          target_backends=backend_info.iree_compiler_targets,
          exported_names=exported_names,
          artifacts_dir=artifacts_dir)
      self._module = rt.VmModule.from_flatbuffer(self._module_blob)
      self._config = rt.Config(driver_name=backend_info.iree_driver)
    else:
      # Called from self.create_reinitialized()
      self._module_blob, self._module, self._config = _create_reinitialized_args

    # Holds all of the module's mutable state.
    self._context = rt.SystemContext(
        modules=[self._module], config=self._config)

  def create_reinitialized(self):
    """Duplicates this module with its initial state without recompiling."""
    default_args = [
        self._module_class, self._backend_info, self._exported_names,
        self._artifacts_dir
    ]
    create_reinitialized_args = [self._module_blob, self._module, self._config]
    return IreeCompiledModule(*default_args, create_reinitialized_args)

  def __getattr__(self, attr):
    # Try to resolve it as a function.
    m = self._context.modules[self._module.name]
    f = m[attr]
    return _IreeFunctionWrapper(self._context, f)


class _IreeFunctionWrapper(object):
  """Wraps an IREE function, making it callable."""

  def __init__(self, context, f):
    self._context = context
    self._f = f

  def __call__(self, *args):
    return self._f(*args)


class TfCompiledModule(CompiledModule):
  """TensorFlow 'compiled' module.

  This facade exists to provide a complimentary API to IreeCompiledModule and
  normalize TensorFlow's output to Numpy.
  """

  def __init__(self,
               module_class,
               backend_info,
               exported_names=[],
               artifacts_dir=None):
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
    self._tf_module = module_class()

  def create_reinitialized(self):
    """Duplicates this module with the starting state of module_class."""
    return TfCompiledModule(self._module_class, self._backend_info,
                            self._exported_names, self._artifacts_dir)

  def __getattr__(self, attr):
    # Try to resolve it as a function.
    exported = len(self._exported_names) == 0 or attr in self._exported_names
    if not hasattr(self._tf_module, attr) or not exported:
      raise AttributeError(f"The TensorFlow module does not have attr '{attr}'")
    f = getattr(self._tf_module, attr)
    if not f or not hasattr(f, "__call__"):
      raise AttributeError(
          f"The TensorFlow module does not have a callable attr '{attr}'")
    return _TfFunctionWrapper(f)


class _TfFunctionWrapper(object):
  """Wraps a TF function, normalizing it to numpy."""

  def __init__(self, f):
    self._f = f

  def __call__(self, *args, **kwargs):
    # TensorFlow will auto-convert all inbound args.
    results = self._f(*args, **kwargs)
    # Then unmarshal them to numpy in the same way that the other backends do.
    # Handle single result (technically ambiguous with return of a tuple,
    # which is sad).
    if not isinstance(results, tuple):
      results = (results,)
    return tf.nest.map_structure(
        lambda t: t.numpy() if isinstance(t, tf.Tensor) else t,
        *results,
        check_types=False)


class BackendInfo(
    collections.namedtuple(
        "BackendInfo",
        ["name", "CompiledModule", "iree_driver", "iree_compiler_targets"])):
  """Info object describing a backend."""

  # All BackendInfo entries by name.
  ALL = {}

  @classmethod
  def add(cls, **kwargs):
    backend_info = cls(**kwargs)
    cls.ALL[backend_info.name] = backend_info


BackendInfo.add(
    name="tf",
    CompiledModule=TfCompiledModule,
    iree_driver=None,
    iree_compiler_targets=None)
# tf_also is used for checking test consistency
# to catch any initialization/randomization issues between model runs
BackendInfo.add(
    name="tf_also",
    CompiledModule=TfCompiledModule,
    iree_driver=None,
    iree_compiler_targets=None)
BackendInfo.add(
    name="iree_vmla",
    CompiledModule=IreeCompiledModule,
    iree_driver="vmla",
    iree_compiler_targets=["vmla"])
BackendInfo.add(
    name="iree_vulkan",
    CompiledModule=IreeCompiledModule,
    iree_driver="vulkan",
    iree_compiler_targets=["vulkan-*"])
BackendInfo.add(
    name="iree_llvmjit",
    CompiledModule=IreeCompiledModule,
    iree_driver="llvm",
    iree_compiler_targets=["llvm-ir"])


def _indent(input_str, indentation=2):
  """Indents a string by the specified number of spaces, defaulting to 2."""
  spaces = " " * indentation
  lines = input_str.split("\n")
  # Prepend spaces to each non-empty line.
  lines = [f"{spaces}{line}" if len(line) else line for line in lines]
  return "\n".join(lines)


class ModuleCall:

  def __init__(self, method_name, inputs, outputs, rtol=1e-6, atol=1e-6):
    """Records the details of a call to a CompiledModule."""
    self.method = method_name

    # Deepcopy to safegard against mutation.
    self.inputs = copy.deepcopy(inputs)
    if outputs is not None:
      outputs = copy.deepcopy(outputs)
    else:
      outputs = tuple()
    self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)

    self.rtol = rtol
    self.atol = atol

  def get_tolerances(self):
    """Gets the floating point tolerances associated with this call."""
    return self.rtol, self.atol

  def __str__(self):
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=NUMPY_LINEWIDTH)

    header = f"Method: {self.method}"
    inputs = "\n".join(_indent(str(value)) for value in self.inputs)
    outputs = "\n".join(_indent(str(value)) for value in self.outputs)
    tolerances = _indent(f"rtol={self.rtol}, atol={self.atol}")
    body = f"Inputs:\n{inputs}\nOutputs:\n{outputs}\nTolerances:\n{tolerances}"
    result = f"{header}\n{_indent(body)}"

    np.set_printoptions(**prior_printoptions)
    return result


class TracedModule:

  def __init__(self, module, trace_function=None):
    """Wraps a CompiledModule so that all inputs and outputs are traced.

    The TracedModule returned will have an API almost identical to that of the
    passed CompiledModule. The only change is that if the keywords `rtol` or
    `atol` are passed to one of the CompiledModule's functions, they will be
    used to set the tolerance for comparing that call to the same call in
    another trace. They will not be passed to the CompiledModule's method.
    So for example, calling `trace.add(a, b, rtol=1e-8)` would be the same as
    calling `module.add(a, b)`.

    Args:
      module: the CompiledModule to trace.
      trace_function: an optional function accepting a TracedModule to run this
        module through. Useful for comparing backends on the same set of calls.
    """
    self.module = module
    self.trace_function = trace_function
    self.trace_name = "unnamed_trace"  # Used for saving.
    self.calls = []
    if self.trace_function is not None:
      self.trace_name = self.trace_function.__name__
      self.trace_function(self)

  def _trace_call(self, method, method_name):
    """Decorates a CompiledModule method to capture its inputs and outputs."""

    def call(*args, **kwargs):
      # Pop manually specified tolerances from the kwargs (if any).
      tolerances = {}
      tolerances["rtol"] = kwargs.pop("rtol", None)
      tolerances["atol"] = kwargs.pop("atol", None)
      # Only pass these to ModuleCall if they were specified by the user.
      tolerances = {k: v for k, v in tolerances.items() if v is not None}

      # Run the method and record the details of the call.
      outputs = method(*args, **kwargs)
      self.calls.append(ModuleCall(method_name, args, outputs, **tolerances))
      return outputs

    return call

  def __getattr__(self, attr):
    # Try to resolve it as an attr on self.module.
    if not hasattr(self.module, attr):
      raise AttributeError(f"The compiled module does not have attr '{attr}'")
    module_attr = getattr(self.module, attr)
    if not hasattr(module_attr, "__call__"):
      # e.g. trace.backend_name
      return module_attr
    else:
      # e.g. trace.simple_mul(a, b)
      return self._trace_call(module_attr, method_name=attr)

  def __str__(self):
    header = f"Trace of {self.module_name} compiled to '{self.backend}' "
    if self.trace_name != "unnamed_trace":
      header += f"on function '{self.trace_name}':"
    # Give each call a number so it's easier to compare between multiple traces.
    calls = [f"{i + 1}. {str(call)}" for i, call in enumerate(self.calls)]
    calls = _indent("\n".join(calls))
    return f"{header}\n{calls}"

  def save_plaintext(self, trace_dir, summarize=True):
    """Saves a human-readable string representation of this trace to disk.

    Args:
      trace_dir: the directory to save the trace in.
      summarize: a bool controlling whether numpy should summarize the inputs
        and outputs if they're large. Setting this to False is very slow for
        large outputs.
    """
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(
        linewidth=NUMPY_LINEWIDTH,
        threshold=None if summarize else sys.maxsize,
        edgeitems=10)  # Can show more items since they won't clutter the logs.

    path = os.path.join(trace_dir, f"{self.trace_name}__{self.backend}.txt")
    with open(path, "w") as f:
      f.write(str(self))

    np.set_printoptions(**prior_printoptions)

  def __iter__(self):
    for call in self.calls:
      yield call
