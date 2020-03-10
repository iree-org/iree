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

# pylint: disable=not-callable
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import collections
import os
import re
import tempfile

from absl import flags
from absl import logging
import numpy as np
from pyiree import rt
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf

flags.DEFINE_string(
    "override_backends", None,
    "Explicit comma-delimited list of target backends. "
    "(Overrides environment variables and auto detection)")
flags.DEFINE_string(
    "debug_dir", None,
    "Specifies a directory to dump debug artifacts to. Defaults to "
    "--test_tmpdir")
FLAGS = flags.FLAGS

ORIGNAL_SAVED_MODEL_PATH_ATTR = "_ORIGINAL_SAVED_MODEL_PATH"

# Per test directory where debug artifacts are dumped.
global_debug_dir = None


def save_and_compile_tf_module(tf_module, exported_names=(),
                               target_backends=()):
  """Saves and compiles a TensorFlow tf.Module.

  Note that if the module has the special _ORIGINAL_SAVED_MODEL_PATH attribute,
  then it will be compiled directly from that path instead of saved and then
  loaded.

  Args:
    tf_module: A tf.Module.
    exported_names: Iterable of dotted function names to consider for
      compilation.
    target_backends: Iterable of string backend names to compile for.

  Returns:
    An _IreeCompiledModule.
  """

  def compile_from_path(sm_path):
    compiler_context = compiler.Context()
    # Break up the compilation so we can save debug artifacts.
    compiler_module = compiler.tf_load_saved_model(
        sm_path,
        exported_names=exported_names,
        compiler_context=compiler_context,
        pass_pipeline=())

    # Save the input MLIR module.
    flattened_target_backends = re.sub("[^0-9a-zA-Z]+", "_",
                                       "__".join(target_backends))
    if global_debug_dir:
      mlir_path = os.path.join(global_debug_dir,
                               "raw_%s.mlir" % flattened_target_backends)
      logging.info("Saving raw TF input MLIR to: %s", mlir_path)
      with open(mlir_path, "w") as f:
        f.write(compiler_module.to_asm())

    # Now run the passes manually that tf_load_saved_model would usually do.
    compiler_module.run_pass_pipeline(compiler.TF_IMPORT_PASS_PIPELINE)

    if global_debug_dir:
      mlir_path = os.path.join(global_debug_dir,
                               "input_%s.mlir" % flattened_target_backends)
      logging.info("Saving IREE input MLIR to: %s", mlir_path)
      with open(mlir_path, "w") as f:
        f.write(compiler_module.to_asm())

    compiled_module = compiler_module.compile(target_backends=target_backends)
    if global_debug_dir:
      compiled_path = os.path.join(
          global_debug_dir, "compiled_%s.vmfb" % flattened_target_backends)
      logging.info("Saving compiled IREE module to: %s", compiled_path)
      with open(compiled_path, "wb") as f:
        f.write(compiled_module)

    return compiled_module

  if hasattr(tf_module, ORIGNAL_SAVED_MODEL_PATH_ATTR):
    # Compile directly from the original path.
    sm_path = getattr(tf_module, ORIGNAL_SAVED_MODEL_PATH_ATTR)
    logging.info(
        "Compiling from original saved_model path (not round-tripping): %s",
        sm_path)
    return compile_from_path(sm_path)
  else:
    # Round-trip through a temporary director.
    with tempfile.TemporaryDirectory() as sm_path:
      options = tf.saved_model.SaveOptions(save_debug_info=True)
      tf.saved_model.save(tf_module, sm_path, options=options)
      return compile_from_path(sm_path)


def load_tf_module(path):
  """Wrapper around tf.saved_model.load which preserves the path.

  Args:
    path: The path to load from.

  Returns:
    The loaded module with an extra property _ORIGINAL_SAVED_MODEL_PATH added.
    This is used on subsequent compiles to load directly from the original
    path, which gives us unmolested access to the original debug information,
    which TensorFlow tends to lose on round-trip.
  """
  tf_module = tf.saved_model.load(path)
  assert not hasattr(tf_module, ORIGNAL_SAVED_MODEL_PATH_ATTR), (
      "Saved model (%s) already has attribute %s" %
      (path, ORIGNAL_SAVED_MODEL_PATH_ATTR))
  setattr(tf_module, ORIGNAL_SAVED_MODEL_PATH_ATTR, path)
  return tf_module


class CompiledModule(object):
  """Base class for per-backend compiled module facade."""

  def __init__(self, ctor, exported_names, backend):
    self._ctor = ctor
    self._exported_names = exported_names
    self._backend = backend

  @staticmethod
  def create(ctor, exported_names, backend):
    compiled_module_class = backend.CompiledModule
    return compiled_module_class(ctor, exported_names, backend)

  @property
  def ctor(self):
    return self._ctor

  def instantiate(self):
    raise NotImplementedError()


class TfCompiledModule(CompiledModule):
  """TensorFlow 'compiled' module.

  This just wraps the constructor.
  """

  def instantiate(self):
    tf_module = self.ctor()
    return _TfModuleInstance(tf_module)


class _TfModuleInstance(object):
  """Instance of a TF module."""

  def __init__(self, tf_module):
    self._tf_module = tf_module

  def __getattr__(self, attr):
    # Try to resolve it as a function.
    if not hasattr(self._tf_module, attr):
      raise AttributeError("The TensorFlow module does not have attr '%s'" %
                           (attr,))
    f = getattr(self._tf_module, attr)
    if not f or not hasattr(f, "__call__"):
      raise AttributeError(
          "The TensorFlow module does not have a callable attr '%s'" % (attr,))
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


class IreeCompiledModule(CompiledModule):
  """Iree compiled module."""

  def __init__(self, ctor, exported_names, backend):
    super().__init__(ctor, exported_names, backend)
    self._iree_module_blob = save_and_compile_tf_module(
        ctor(),
        exported_names=exported_names,
        target_backends=backend.iree_compiler_targets)
    self._iree_module = rt.VmModule.from_flatbuffer(self._iree_module_blob)

  def instantiate(self):
    return _IreeModuleInstance(self._backend, self._iree_module_blob,
                               self._iree_module)


class _IreeModuleInstance(object):
  """An instance of an IREE module."""

  def __init__(self, backend, iree_module_blob, iree_module):
    self._backend = backend
    self._iree_module_blob = iree_module_blob
    self._iree_module = iree_module
    self._iree_module_name = self._iree_module.name

    self._system_config = rt.Config(driver_name=backend.iree_driver)
    self._context = rt.SystemContext(
        modules=[self._iree_module], config=self._system_config)

  def __getattr__(self, attr):
    # Try to resolve it as a function.
    m = self._context.modules[self._iree_module_name]
    f = m[attr]
    return _IreeFunctionWrapper(self._context, f)


class _IreeFunctionWrapper(object):
  """Wraps an IRRE function, making it callable."""

  def __init__(self, context, f):
    self._context = context
    self._f = f

  def __call__(self, *args):
    return self._f(*args)


class _VirtualModuleInstance(object):
  """Wraps a namedtuple of modules and represents a union of them."""

  def __init__(self, named_modules, match_spec):
    self._named_modules = named_modules
    self._match_spec = match_spec

  def __getattr__(self, attr):
    match_modules = {
        k: v
        for k, v in self._named_modules.items()
        if re.search(self._match_spec, k)
    }
    if not match_modules:
      raise AttributeError(
          "Module match spec '%s' did not match anything. (Have %r)" %
          (self._match_spec, self._named_modules.keys()))
    # Resolve functions on each.
    match_functions = {}
    for backend, module in match_modules.items():
      try:
        match_functions[backend] = getattr(module, attr)
      except:
        raise AttributeError(
            "Could not resolve function '%s' on backend module '%s'" %
            (attr, backend))
    return _VirtualFunctionWrapper(match_functions)


class _VirtualFunctionWrapper(object):
  """Wrapper around a virtual dict of functions."""

  def __init__(self, backend_function_dict):
    self._backend_function_dict = backend_function_dict

  def __call__(self, *args, **kwargs):
    all_results = {
        backend: f(*args, **kwargs)
        for backend, f in self._backend_function_dict.items()
    }
    # Turn it into a named tuple so we get nice class-like access to it.
    results_tuple_class = collections.namedtuple("Results", all_results.keys())
    return _make_multi_result_class(results_tuple_class)(*all_results.values())


def _collect_disagreements(mr, predicate):
  """Verifies that result structs.

  Args:
    mr: A MultiResults namedtuple where each entry corresponds to a backend set
      of results.
    predicate: A predicate function which takes (a, b) and returns whether they
      should be considered equivalent.

  Returns:
    An equivalent MultiResults where each entry is an array of result names
    that disagree.
  """
  has_disagreement = False
  disagreement_list = [list() for _ in mr]
  for i in range(len(mr)):
    result_ref = mr[i]
    for j in range(len(mr)):
      if i == j:
        continue  # Don't check self.
      result_tgt = mr[j]
      if not predicate(result_ref, result_tgt):
        has_disagreement = True
        disagreement_list[i].append(mr._fields[j])
  disagreements_tuple = collections.namedtuple("Disagreements", mr._fields)
  return has_disagreement, disagreements_tuple(*disagreement_list)


def _make_multi_result_class(named_tuple_class):
  """Makes a class that wraps a mapping of backend results."""

  class MultiResults(named_tuple_class):
    """Wraps a mapping of results."""

    def assert_all_close(self, rtol=1e-6, atol=1e-6):
      predicate = (lambda a, b: np.allclose(a, b, rtol=rtol, atol=atol))
      has_disagreement, disagreements = _collect_disagreements(self, predicate)
      assert not has_disagreement, ("Multiple backends disagree (%r):\n%r" %
                                    (disagreements, self))
      return self

    def assert_all_equal(self):
      predicate = np.array_equal
      has_disagreement, disagreements = _collect_disagreements(self, predicate)
      assert not has_disagreement, ("Multiple backends disagree (%r):\n%r" %
                                    (disagreements, self))
      return self

    def print(self):
      print(self)
      return self

  return MultiResults


def _instantiate_modules(compiled_modules_dict):
  """Given a dict of modules, instantiates them.

  Args:
    compiled_modules_dict: Dictionary of
        {module_name:{backend_name:CompiledModule}} that should be instantiated.

  Returns:
    namedtuple mapping module_key:VirtualBackendsClass for every module
    in compiled_modules_dict. The VirtualBackendsClass is a dynamically
    generated namedtuple mapping backend_name:ModuleInstance, where the
    ModuleInstance allows attribute resolution of public functions on the
    module. The VirtualBackendsClass also contributes some convenience
    methods for selecting all or a subset of matching backend modules.
  """

  def instantiate_backends(module_dict):
    """Creates a VirtualBackend namedtuple class for a dict.

    Args:
      module_dict: Dictionary of backend_name:ModuleInstance.

    Returns:
      namedtuple subclass with a field for every backend and special
      all and multi() helpers.
    """
    tuple_class = collections.namedtuple("VirtualBackendsTuple",
                                         module_dict.keys())

    class VirtualBackendsClass(tuple_class):
      """Adds a __call__ method that creates a virtual module."""

      def multi(self, match_spec="."):
        """Selects multiple backends that match a regular expression."""
        return _VirtualModuleInstance(self._asdict(), match_spec)

      @property
      def all(self):
        """Shorthand for multi() which selects all backends."""
        return self.multi()

    return VirtualBackendsClass(
        *[m.instantiate() for m in module_dict.values()])

  module_keys = [k for (k, _) in compiled_modules_dict.items()]
  module_insts = [
      instantiate_backends(module_dict)
      for (_, module_dict) in compiled_modules_dict.items()
  ]
  tuple_class = collections.namedtuple("Modules", module_keys)
  return tuple_class(*module_insts)


def compile_modules(backends=None, **kwargs):
  """Decorator applied to a SavedModelTestCase subclass to compile modules.

  Args:
    backends: an iterable of backend names to include (or None to use
      environment defaults).
    **kwargs: name/Module constructor mappings. Each such arg will be added to
      the classes 'compiled_modules' field.

  Returns:
    Class decorator function.
  """

  def decorator(cls):
    """Decorator function."""
    assert issubclass(cls, SavedModelTestCase), (
        "The 'compile_modules' decorator must be applied to a "
        "SavedModelTestCase derived class.")
    if not cls._modules_to_compile:
      cls._modules_to_compile = {}
    for name, ctor in kwargs.items():
      assert name not in cls._modules_to_compile, (
          "@compile_modules called with duplicate module names '%s'" % (name,))
      exported_names = ()
      if isinstance(ctor, tuple):
        ctor, exported_names = ctor
      cls._modules_to_compile[name] = (ctor, exported_names, backends)

    return cls

  return decorator


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


def _backend_spec_string_to_backends(backend_spec):
  """Decodes a comma-delimited string of backends into BackendInfo objects."""
  backends = []
  for backend_name in backend_spec.split(","):
    if backend_name not in BackendInfo.ALL.keys():
      raise ValueError(
          "Invalid backend specification string '{}', unexpected name '{}';"
          " valid names are '{}'".format(backend_spec, backend_name,
                                         BackendInfo.ALL.keys()))
    backends.append(BackendInfo.ALL[backend_name])
  return backends


def get_override_backends():
  """Gets the BackendInfo instances to test, as overridden by the user.

  Returns:
    Sequence of BackendInfo that should be used, or None if there is no
    override.
  """

  if FLAGS.override_backends is not None:
    backends_spec = FLAGS.override_backends
    logging.info("Using backends from command line: %s", backends_spec)
  else:
    backends_spec = os.environ.get("IREE_OVERRIDE_BACKENDS")
    if backends_spec is not None:
      logging.info("Using backends from environment IREE_OVERRIDE_BACKENDS: %s",
                   backends_spec)

  if backends_spec:
    return _backend_spec_string_to_backends(backends_spec)
  else:
    logging.info("No backend overrides.")
    return None


def get_default_backends():
  """Gets the BackendInfo instances to use by default."""
  backend_spec = os.environ.get("IREE_DEFAULT_BACKENDS")
  if backend_spec is None:
    return BackendInfo.ALL.values()
  return _backend_spec_string_to_backends(backend_spec)


class SavedModelTestCase(tf.test.TestCase):
  """Tests against a SavedModel."""

  # Will be initialized to a dict by the @compile_modules decorator.
  # The dict maps module name to (ctor, exported_names, backend_names).
  _modules_to_compile = None

  # Will be initialized in setUpClass to a dict of (name, CompiledModule)
  # instances mirroring _modules_to_compile.
  compiled_modules = None

  TRACE_FILE_NAME = None

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.modules = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.compiled_modules = {}
    if cls._modules_to_compile:
      for name, (ctor, exported_names,
                 backends) in cls._modules_to_compile.items():

        # Setup the debug directory.
        debug_parent_dir = FLAGS.debug_dir
        if not debug_parent_dir:
          debug_parent_dir = FLAGS.test_tmpdir
        debug_parent_dir = os.path.join(debug_parent_dir, cls.__name__)

        try:
          os.makedirs(debug_parent_dir)
        except IOError:
          logging.exception("Error creating crash reproducer dir for: %s",
                            debug_parent_dir)

        # Setup crash reproducer and global debug dir.
        crash_reproducer_path = os.path.join(debug_parent_dir,
                                             name + "_reproducer.mlir")
        compiler.Context.default_crash_reproducer_path = crash_reproducer_path
        global global_debug_dir
        global_debug_dir = debug_parent_dir

        try:
          # Compile.
          # Expand backend names to BackendInfo objects.
          def _resolve(backend_spec):
            if isinstance(backend_spec, BackendInfo):
              return backend_spec
            # Handle the string form.
            return BackendInfo.ALL[backend_spec]

          override_backends = get_override_backends()
          if override_backends is not None:
            backends = override_backends
          elif backends is None:
            backends = get_default_backends()
          backends = [_resolve(backend) for backend in backends]
          cls.compiled_modules[name] = dict([
              (backend.name, CompiledModule.create(ctor, exported_names,
                                                   backend))
              for backend in backends
          ])
        finally:
          # Disable crash reproducer (to avoid inadvertently overwriting this
          # path on a subsequent interaction).
          compiler.Context.default_crash_reproducer_path = None
          global_debug_dir = None

  @classmethod
  def tearDownClass(cls):
    trace_file_name = cls.TRACE_FILE_NAME
    if not trace_file_name:
      trace_file_name = cls.__name__ + ".wtf-trace"
    trace_file = os.path.join(tempfile.gettempdir(), trace_file_name)
    print("Flushing trace file to:", trace_file)
    rt.binding.tracing.flush(trace_file)
    print("Flush complete")
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.modules = _instantiate_modules(self.compiled_modules)
