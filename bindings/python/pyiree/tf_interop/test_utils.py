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

import os
import tempfile
import timeit

from .. import binding
from .. import compiler
import numpy as np
import tensorflow.compat.v2 as tf


def save_and_compile_tf_module(tf_module):
  with tempfile.TemporaryDirectory() as sm_path:
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(tf_module, sm_path, options=options)
    return compiler.tf_compile_saved_model(sm_path)


def dump_iree_module(m):
  print("Loaded module:", m.name)
  i = 0
  while True:
    f = m.lookup_function_by_ordinal(i)
    if not f:
      break
    print("  Export:", f.name, "-> args(", f.signature.argument_count,
          "), results(", f.signature.result_count, ")")
    i += 1


def get_default_test_backends():
  backends_env = os.environ.get("IREE_TEST_BACKENDS")
  if backends_env:
    return backends_env.split(",")
  else:
    return ("tf", "iree.interpreter")


class _TfBackend(object):
  """Backend for running directly on the TF module."""

  def __init__(self, test_case, backend_name, fn_name):
    self.backend_name = backend_name
    self.module_f = getattr(test_case.tf_module, fn_name)

  def __call__(self, *args):
    return self.module_f(*args)

  def postprocess(self, results):
    # Handle single result (technically ambiguous with return of a tuple).
    if not isinstance(results, tuple):
      results = (results,)
    # TODO(laurenzo): Handle structure mapping, etc.
    return [r.numpy() for r in results]


class _IreeBackend(object):
  """Backend for running on an IREE driver."""

  def __init__(self, test_case, backend_name, fn_name):
    self.backend_name = backend_name
    driver_name = backend_name.split(".")[-1]
    self.policy = binding.rt.Policy()
    instance = binding.rt.Instance(driver_name=driver_name)
    self.context = binding.rt.Context(instance=instance, policy=self.policy)
    self.context.register_module(test_case.iree_vm_module)
    self.f = self.context.resolve_function("module." + fn_name)

  def __call__(self, *args):
    args = [self.context.wrap_for_input(arg) for arg in args]
    # Invoke the function and wait for completion.
    inv = self.context.invoke(self.f, self.policy, args)
    inv.await_ready()
    # Get results as a numpy array.
    results = [np.array(r.map(), copy=False) for r in inv.results]
    return results

  def postprocess(self, results):
    return results


_ALL_BACKENDS = {
    "tf": _TfBackend,
    "iree.interpreter": _IreeBackend,
    "iree.vulkan": _IreeBackend,
}


def _wrap_per_backend_fn(saved_model_test_case, fn_name, iterations=100):
  """Generates a wrapper function for a backend fn name."""

  def invoke_fn(*args):
    """Lambda that invokes the function on all backends."""

    backend_names = saved_model_test_case.BACKENDS
    if not backend_names:
      backend_names = get_default_test_backends()

    backends = [
        _ALL_BACKENDS[b](saved_model_test_case, b, fn_name)
        for b in backend_names
    ]
    test_id = saved_model_test_case.id().split(".")[-1]

    per_backend_results = []
    binding.tracing.enable_thread()
    for backend in backends:
      # pylint: disable=cell-var-from-loop
      print(":INVOKE %s:%s on %s" % (test_id, fn_name, backend.backend_name))
      event = binding.tracing.ScopedEvent(
          "%s_%s#%s" % (test_id, fn_name, backend.backend_name))

      def run_iteration():
        with event:
          return backend(*args)

      # Run one for correctness.
      results = backend.postprocess(run_iteration())
      per_backend_results.append((backend.backend_name, results))
      # Then time it.
      backend_time_ms = timeit.timeit(run_iteration, number=iterations) * 1000
      iteration_time_ms = backend_time_ms / iterations
      print(":BENCHMARK %s:%s on %s: time=%rms" %
            (test_id, fn_name, backend.backend_name, iteration_time_ms))
      # pylint: enable=cell-var-from-loop

    # Verify results.
    ref_backend_name, ref_results = per_backend_results[0]
    print(":REF RESULTS %s:%s %s:" % (test_id, fn_name, ref_backend_name),
          ref_results)
    for backend_name, results in per_backend_results[1:]:
      print(":COMPARE %s:%s %s vs %s" %
            (test_id, fn_name, ref_backend_name, backend_name))
      print("  :", results)
      for ref_result, result in zip(ref_results, results):
        saved_model_test_case.assertAllClose(
            ref_result,
            result,
            msg="Result mismatch %s vs %s" % (ref_backend_name, backend_name))

    return ref_results

  return invoke_fn


def per_backend_test(*fn_names):
  """Wraps a SavedModelTestCase test method to run per backend tests.

  Args:
    *fn_names: Names of functions to run tests against. These will be converted
      to python functions that invoke all of the backends and passed to the test
      case method.

  Returns:
    A decorated function.
  """

  def decorator(f):

    def replacement(self):
      fns = [_wrap_per_backend_fn(self, fn_name) for fn_name in fn_names]
      f(self, *fns)

    replacement.__name__ = f.__name__
    return replacement

  return decorator


class SavedModelTestCase(tf.test.TestCase):
  """Tests against a SavedModel.

  Use this by subclassing and then defining a TF_MODULE_CONSTRUCTOR member.
  """

  TF_MODULE_CONSTRUCTOR = None
  TRACE_FILE_NAME = None
  BACKENDS = None

  @classmethod
  def tearDownClass(cls):
    trace_file_name = cls.TRACE_FILE_NAME
    if not trace_file_name:
      trace_file_name = cls.__name__ + ".wtf-trace"
    trace_file = os.path.join(tempfile.gettempdir(), trace_file_name)
    print("Flushing trace file to:", trace_file)
    binding.tracing.flush(trace_file)
    print("Flush complete")
    super().tearDownClass()

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if cls.TF_MODULE_CONSTRUCTOR is None:
      raise ValueError("Expected a class level TF_MODULE_CONSTRUCTOR")
    # Compile the module. We do this once.
    cls.tf_module = cls.TF_MODULE_CONSTRUCTOR()  # pylint: disable=not-callable
    cls.iree_blob = save_and_compile_tf_module(cls.tf_module)
    cls.iree_vm_module = binding.vm.create_module_from_blob(cls.iree_blob)
    dump_iree_module(cls.iree_vm_module)
