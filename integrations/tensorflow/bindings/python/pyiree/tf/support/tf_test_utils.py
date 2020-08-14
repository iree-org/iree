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
"""Test utilities interop with TensorFlow."""

# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=unsupported-assignment-operation

# This file uses the following abbreviations:
#   ref: reference – for the reference CompiledModule
#   tar: target - for one of the target CompiledModules

import copy
import inspect
import os
import sys
import tempfile

from absl import flags
from absl import logging
import numpy as np
from pyiree.tf import compiler
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_string("reference_backend", "tf",
                    "The backend to treat as a source of truth.")
flags.DEFINE_string("target_backends", None,
                    "Explicit comma-delimited list of target backends.")
flags.DEFINE_string(
    "artifacts_dir", None,
    "Specifies a directory to dump compilation artifacts and traces to. "
    "Defaults to the OS's tempdir.")
flags.DEFINE_bool(
    "summarize", True,
    "Summarize the inputs and outputs of each module trace logged to disk.")
flags.DEFINE_bool("log_all_traces", False,
                  "Log all traces to logging.info, even if comparison passes.")
FLAGS = flags.FLAGS
NUMPY_LINEWIDTH = 120


def _setup_artifacts_dir(module_name):
  parent_dir = FLAGS.artifacts_dir
  if parent_dir is None:
    parent_dir = os.path.join(tempfile.gettempdir(), "iree", "modules")
  artifacts_dir = os.path.join(parent_dir, module_name)
  logging.info("Saving compilation artifacts and traces to '%s'", artifacts_dir)
  tf_utils._makedirs(artifacts_dir)
  return artifacts_dir


def _parse_target_backends():
  """Decodes --target_backends and creates unique names for their artifacts."""
  backend_names = FLAGS.target_backends.split(",")
  backend_to_index = {k: 0 for k in backend_names if backend_names.count(k) > 1}
  artifact_names = []

  # If there are multiple copies of the same backend_name, index them. e.g.
  # backend_names = ["tf", "iree_vmla", "tf"]
  # --> artifact_names = ["tf_0", "iree_vmla", "tf_1"]
  for backend_name in backend_names:
    if backend_name in backend_to_index:
      artifact_names.append(f"{backend_name}_{backend_to_index[backend_name]}")
      backend_to_index[backend_name] += 1
    else:
      artifact_names.append(backend_name)

  return backend_names, artifact_names


def get_target_backends():
  """Gets the BackendInfo instances to compare with the reference backend.

  By default all backends in BackendInfo will be used. Specific backends to
  run on can be specified using the `--target_backends` flag.

  Returns:
    Sequence of BackendInfo that should be used.
  """
  if FLAGS.target_backends is not None:
    logging.info("Using backends from command line: %s", FLAGS.target_backends)
    backend_names, names = _parse_target_backends()
    backends = [
        tf_utils.BackendInfo(backend, name)
        for backend, name in zip(backend_names, names)
    ]
  else:
    # If no backends are specified, use them all.
    backends = tf_utils.BackendInfo.get_all_backends()
  return backends


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

  def _get_shape_and_dtype(self, value):
    if isinstance(value, np.ndarray):
      return tf_utils.get_shape_and_dtype(value, allow_non_mlir_dtype=True)
    else:
      return str(type(value))

  def __str__(self):
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=NUMPY_LINEWIDTH)

    header = f"Method: {self.method}"
    inputs = "\n".join(_indent(str(value)) for value in self.inputs)
    input_shapes = ", ".join(
        self._get_shape_and_dtype(value) for value in self.inputs)

    outputs = "\n".join(_indent(str(value)) for value in self.outputs)
    output_shapes = ", ".join(
        self._get_shape_and_dtype(value) for value in self.outputs)

    tolerances = _indent(f"rtol={self.rtol}, atol={self.atol}")
    body = (f"Inputs: {input_shapes}\n{inputs}\n"
            f"Outputs: {output_shapes}\n{outputs}"
            f"\nTolerances:\n{tolerances}")
    result = f"{header}\n{_indent(body)}"

    np.set_printoptions(**prior_printoptions)
    return result


class Trace:
  """Stores the inputs and outputs of a series of calls to a module."""

  def __init__(self, module, function):
    """Extracts metadata from module and function and initializes.

    Example usage:
      def forward_pass(...):
        ...
      module = IreeCompiledModule(...)
      trace = Trace(module, forward_pass)
      forward_pass(TracedModule(module, trace))

    Args:
      module: the module who's outputs this trace will record.
      function: the function that module will be traced on.
    """
    # Extract metadata from module and function.
    self.module_name = module.module_name
    self.backend = module.backend
    self.function_name = function.__name__
    self.function_sourcefile = inspect.getsourcefile(function)
    source, start_line = inspect.getsourcelines(function)
    self.function_line_numbers = (start_line, start_line + len(source))
    self.function_source = "".join(source)

    self.calls = []

  def __str__(self):
    header = (f"Trace of {self.module_name} compiled to '{self.backend}' "
              f"on function '{self.function_name}':")
    # Give each call a number so it's easier to compare between multiple traces.
    calls = [f"{i + 1}. {str(call)}" for i, call in enumerate(self.calls)]
    calls = _indent("\n".join(calls))
    return f"{header}\n{calls}"

  def __iter__(self):
    for call in self.calls:
      yield call

  @staticmethod
  def compare_traces(ref_trace, tar_trace):
    traces_match = True

    # Check that all method invocations match.
    ref_methods = [(call.method, call.rtol, call.atol) for call in ref_trace]
    tar_methods = [(call.method, call.rtol, call.atol) for call in tar_trace]
    if ref_methods != tar_methods:
      # Raise a ValueError instead of returning False since this is an
      # unexpected error.
      raise ValueError(
          "The reference and target traces have different call structures:\n"
          f"Reference: {ref_methods}\nTarget:    {tar_methods}")

    for ref_call, tar_call in zip(ref_trace, tar_trace):
      logging.info("Comparing calls to '%s'", ref_call.method)
      rtol, atol = ref_call.get_tolerances()

      inputs_match = Trace._check_same(ref_call.inputs, tar_call.inputs, rtol,
                                       atol)
      if not inputs_match:
        logging.error("Inputs did not match.")
      outputs_match = Trace._check_same(ref_call.outputs, tar_call.outputs,
                                        rtol, atol)
      if not outputs_match:
        logging.error("Outputs did not match.")
      calls_match = inputs_match and outputs_match

      if not calls_match:
        logging.error("Comparision between '%s' and '%s' failed on method '%s'",
                      ref_trace.backend, tar_trace.backend, ref_call.method)
        logging.error("Reference call '%s':\n%s", ref_trace.backend, ref_call)
        logging.error("Target call '%s':\n%s", tar_trace.backend, tar_call)

      traces_match = traces_match and calls_match
    return traces_match

  @staticmethod
  def _check_same(ref, tar, rtol, atol):
    """Checks that ref and tar have identical datastructures and values."""
    # Check for matching types.
    if not isinstance(tar, type(ref)):
      logging.error(
          "Expected ref and tar to have the same type but got '%s' and '%s'",
          type(ref), type(tar))
      return False

    if ref is None:
      # Nothing to compare (e.g. the called method had no outputs).
      return True

    # Recursive check for dicts.
    if isinstance(ref, dict):
      if ref.keys() != tar.keys():
        logging.error(
            "Expected ref and tar to have the same keys, but got '%s' and '%s'",
            ref.keys(), tar.keys())
        return False
      # Check that all of the dictionaries' values are the same.
      for key in ref:
        if not Trace._check_same(ref[key], tar[key], rtol, atol):
          return False

    # Recursive check for iterables.
    elif isinstance(ref, list) or isinstance(ref, tuple):
      if len(ref) != len(tar):
        logging.error(
            "Expected ref and tar to have the same length, but got %s and %s",
            len(ref), len(tar))
        return False
      # Check that all of the iterables' values are the same.
      for i in range(len(ref)):
        if not Trace._check_same(ref[i], tar[i], rtol, atol):
          return False

    # Base check for numpy arrays.
    elif isinstance(ref, np.ndarray):
      if ref.dtype != tar.dtype:
        logging.error(
            "Expected ref and tar to have the same dtype, but got %s  and %s",
            ref.dtype, tar.dtype)
        return False
      if np.issubdtype(ref.dtype, np.floating):
        same = np.allclose(ref, tar, rtol=rtol, atol=atol)
        if not same:
          abs_diff = np.max(np.abs(ref - tar))
          rel_diff = np.max(np.abs(ref - tar) / np.max(tar))
          logging.error(
              "Floating point difference between ref and tar was too large. "
              "Max abs diff: %s, atol: %s, max relative diff: %s, rtol: %s",
              abs_diff, atol, rel_diff, rtol)
        return same
      else:
        return np.array_equal(ref, tar)

    # Base check for native number types.
    elif isinstance(ref, (int, float)):
      return ref == tar

    # If outputs end up here then an extra branch for that type should be added.
    else:
      raise TypeError(f"Encountered results with unexpected type {type(ref)}")
    return True

  def _get_trace_dir(self, artifacts_dir):
    trace_dir = os.path.join(artifacts_dir, self.backend, "traces",
                             self.function_name)
    tf_utils._makedirs(trace_dir)
    return trace_dir

  def save_plaintext(self, artifacts_dir, summarize=True):
    """Saves a human-readable string representation of this trace to disk.

    Args:
      artifacts_dir: the base directory to save the trace in.
      summarize: a bool controlling whether numpy should summarize the inputs
        and outputs if they're large. Setting this to False is very slow for
        large outputs.
    """
    prior_printoptions = np.get_printoptions()
    np.set_printoptions(
        linewidth=NUMPY_LINEWIDTH,
        threshold=None if summarize else sys.maxsize,
        edgeitems=10)  # Can show more items since they won't clutter the logs.

    trace_dir = self._get_trace_dir(artifacts_dir)
    path = os.path.join(trace_dir, "log.txt")
    with open(path, "w") as f:
      f.write(str(self))
      f.write("\n")

    np.set_printoptions(**prior_printoptions)


class TracedModule:

  def __init__(self, module, trace):
    """Wraps a CompiledModule so that all inputs and outputs are traced.

    The TracedModule returned will have an API almost identical to that of the
    passed CompiledModule. The only changes is that if the keywords `rtol` or
    `atol` are passed to one of the CompiledModule's methods, then they will be
    used to set the tolerance for comparing that call to the same call in
    another trace. So for example, calling `traced_module.add(a, b rtol=1e-8)`
    would be the same as calling `module.add(a, b)`.

    Args:
      module: the CompiledModule to trace.
      trace: the Trace to record calls to this module with.
    """
    self._module = module
    self._trace = trace

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
      self._trace.calls.append(
          ModuleCall(method_name, args, outputs, **tolerances))
      return outputs

    return call

  def __getattr__(self, attr):
    # Try to resolve it as an attr on self._module.
    if not hasattr(self._module, attr):
      raise AttributeError(f"The compiled module does not have attr '{attr}'")
    module_attr = getattr(self._module, attr)
    if not hasattr(module_attr, "__call__"):
      # e.g. traced_module.backend
      return module_attr
    else:
      # e.g. traced_module.simple_mul(a, b)
      return self._trace_call(module_attr, method_name=attr)


def compile_module(module_class, exported_names=()):
  """CompiledModuleTestCase decorator that compiles a tf.Module.

  A CompiledModule is created for each backend in --target_backends. They can
  be accessed individually via self.compiled_modules.backend_name or as a union
  via self.get_module().

  Args:
    module_class: the tf.Module subclass to compile.
    exported_names: optional iterable of strings representing which of
      module_class's functions to compile. If exported_names is empty all
      functions will be compiled.

  Returns:
    Class decorator function.
  """

  def decorator(cls):
    """Decorator Function."""
    if not issubclass(cls, TracedModuleTestCase):
      logging.exception(
          "The 'compile_module' decorator must be applied to a "
          "TracedModuleTestCase derived class, which %s is not.", cls)
    cls._module_class = module_class
    cls._exported_names = exported_names
    return cls

  return decorator


class TracedModuleTestCase(tf.test.TestCase):
  """Compiles a tf.Module to multiple backends to test their correctness."""
  # Will be initialized by the @compile_module decorator.
  _module_class = None
  _exported_names = ()

  # Will be initialized in setUpClass.
  _ref_module = None
  _tar_modules = None

  @classmethod
  def _compile(cls, backend_info):
    return backend_info.compile(cls._module_class, cls._exported_names,
                                cls._artifacts_dir)

  @classmethod
  def setUpClass(cls):
    # Ran before any of the unit tests.
    super().setUpClass()
    if cls._module_class is None:
      raise AttributeError(
          "setUpClass was called but no module was specified. Specify a module "
          "to compile via the @tf_test_utils.compile_module decorator.")

    # Setup the directory for saving compilation artifacts and traces.
    cls._artifacts_dir = _setup_artifacts_dir(cls._module_class.__name__)

    # Create a CompiledModule for the reference backend and each target backend.
    ref_backend_info = tf_utils.BackendInfo(FLAGS.reference_backend,
                                            f"{FLAGS.reference_backend}_ref")
    cls._ref_module = cls._compile(ref_backend_info)

    tar_backend_infos = get_target_backends()
    cls._tar_modules = [
        cls._compile(backend_info) for backend_info in tar_backend_infos
    ]

  def setUp(self):
    # Ran before each unit test.
    super().setUp()
    self._ref_module.create_reinitialized()
    self._tar_modules = [
        module.create_reinitialized() for module in self._tar_modules
    ]

  def compare_backends(self, trace_function):
    """Run the reference and target backends on trace_function and compare them.

    Random seeds for tensorflow, numpy and python are set before each invocation
    of trace_function.

    Args:
      trace_function: a function accepting a TracedModule as its argument.
    """
    # Create Traces for each backend.
    ref_trace = Trace(self._ref_module, trace_function)
    tar_traces = [Trace(module, trace_function) for module in self._tar_modules]

    # Run the traces through trace_function with their associated modules.
    tf_utils.set_random_seed()
    trace_function(TracedModule(self._ref_module, ref_trace))
    if FLAGS.log_all_traces:
      logging.info(ref_trace)
    for module, trace in zip(self._tar_modules, tar_traces):
      tf_utils.set_random_seed()
      trace_function(TracedModule(module, trace))
      if FLAGS.log_all_traces:
        logging.info(trace)

    # Compare each target trace of trace_function with the reference trace.
    failed_backend_indices = []
    for i, tar_trace in enumerate(tar_traces):
      logging.info("Comparing the reference backend '%s' with '%s'",
                   ref_trace.backend, tar_trace.backend)
      traces_match = Trace.compare_traces(ref_trace, tar_trace)
      if not traces_match:
        failed_backend_indices.append(i)

    # Save the results to disk before validating.
    ref_trace.save_plaintext(self._artifacts_dir, FLAGS.summarize)
    for tar_trace in tar_traces:
      tar_trace.save_plaintext(self._artifacts_dir, FLAGS.summarize)

    # Validate results.
    if failed_backend_indices:
      # Extract info for logging.
      failed_backends = [tar_traces[i].backend for i in failed_backend_indices]
      self.fail(
          "Comparision between the reference backend and the following targets "
          f"failed: {failed_backends}. The errors above show the inputs and "
          "outputs of the non-matching calls.")

  @classmethod
  def tearDownClass(cls):
    # Ran after all unit tests are completed.
    super().tearDownClass()
