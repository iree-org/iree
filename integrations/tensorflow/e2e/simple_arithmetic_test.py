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
"""Several baseline e2e simple arithmetic tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import timeit

import numpy as np
import tensorflow.compat.v2 as tf
import pyiree


class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

  @tf.function(input_signature=[
      tf.TensorSpec([128, 3072], tf.float32),
      tf.TensorSpec([3072, 256], tf.float32),
  ])
  def simple_matmul(self, a, b):
    return tf.matmul(a, b)


def save_and_load_tf_module(tf_module):
  with tempfile.TemporaryDirectory() as sm_path:
    options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(tf_module, sm_path, options=options)
    ctx = pyiree.CompilerContext()
    input_module = pyiree.tf_load_saved_model(ctx, sm_path)
  return input_module


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


class SimpleArithmeticTest(tf.test.TestCase):

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    print("Flushing trace file...")
    pyiree.tracing.flush("/tmp/simple_arithmetic_test.wtf-trace")

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Compile the module. We do this once.
    cls.tf_module = SimpleArithmeticModule()
    cls.mlir_input_module = save_and_load_tf_module(cls.tf_module)
    print("LOADED ASM:",
          cls.mlir_input_module.to_asm(debug_info=True, pretty=True))

    # Canonicalize the TF import.
    cls.mlir_input_module.run_pass_pipeline([
        "tf-executor-graph-pruning",
        "tf-standard-pipeline",
        "canonicalize",
    ])
    print("CANONICAL TF ASM:",
          cls.mlir_input_module.to_asm(debug_info=True, pretty=True))

    # Legalize to XLA (high-level).
    cls.mlir_input_module.run_pass_pipeline([
        "xla-legalize-tf",
    ])
    print("XLA ASM:",
          cls.mlir_input_module.to_asm(debug_info=True, pretty=True))

    # Compile the module with IREE.
    cls.iree_blob = cls.mlir_input_module.compile_to_sequencer_blob(
        print_mlir=True)
    cls.iree_vm_module = pyiree.binding.vm.create_module_from_blob(
        cls.iree_blob)
    dump_iree_module(cls.iree_vm_module)

  def test_simple_matmul(self):
    pyiree.tracing.enable_thread()
    # Initialize the runtime and register the module.
    # Use the CPU interpreter driver (which has the most implementation done):
    driver_name = "interpreter"

    # Live on the edge and give the vulkan driver a try:
    # driver_name = "vulkan"

    policy = pyiree.binding.rt.Policy()
    instance = pyiree.binding.rt.Instance(driver_name=driver_name)
    context = pyiree.binding.rt.Context(instance=instance, policy=policy)
    context.register_module(self.iree_vm_module)

    f = context.resolve_function("module.simple_matmul")
    tf_f = self.tf_module.simple_matmul
    a = np.zeros((128, 3072), dtype=np.float32) + 1
    b = np.ones((3072, 256), dtype=np.float32) + 2

    iree_event = pyiree.tracing.ScopedEvent(
        "SimpleArithmeticTest#simple_matmul")

    def invoke_iree():
      with iree_event:
        arg0 = context.wrap_for_input(a)
        arg1 = context.wrap_for_input(b)

        # Invoke the function and wait for completion.
        inv = context.invoke(f, policy, [arg0, arg1])
        inv.await_ready()

        # Get the result as a numpy array and print.
        results = inv.results
        result = results[0].map()
        result_ary = np.array(result, copy=False)
        return result_ary

    def invoke_tf():
      arg0 = a
      arg1 = b
      result = tf_f(arg0, arg1)
      return result.numpy()

    # Check that results are equal.
    self.assertAllEqual(invoke_iree(), invoke_tf())
    # Quick benchmark.
    iterations = 5  # TODO(laurenzo): Increase when AVX bug fixed.
    print("+++BM simple_matmul:")
    iree_time = timeit.timeit(invoke_iree, number=iterations)
    print("IREE -> TIME/ITERATION =", (iree_time / iterations) * 1000, "ms")
    tf_time = timeit.timeit(invoke_tf, number=iterations)
    print("TF   -> TIME/ITERATION =", (tf_time / iterations) * 1000, "ms")
    tf_vs_iree_factor = tf_time / iree_time
    print("IREE VS TF SPEEDUP FACTOR =", tf_vs_iree_factor)

  def test_simple_scalar_mul(self):
    pyiree.tracing.enable_thread()
    # Initialize the runtime and register the module.
    # Use the CPU interpreter driver (which has the most implementation done):
    driver_name = "interpreter"

    # Live on the edge and give the vulkan driver a try:
    # driver_name = "vulkan"

    policy = pyiree.binding.rt.Policy()
    instance = pyiree.binding.rt.Instance(driver_name=driver_name)
    context = pyiree.binding.rt.Context(instance=instance, policy=policy)
    context.register_module(self.iree_vm_module)

    f = context.resolve_function("module.simple_mul")
    tf_f = self.tf_module.simple_mul
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    iree_event = pyiree.tracing.ScopedEvent("SimpleArithmeticTest#simple_mul")

    def invoke_iree():
      with iree_event:
        arg0 = context.wrap_for_input(a)
        arg1 = context.wrap_for_input(b)

        # Invoke the function and wait for completion.
        inv = context.invoke(f, policy, [arg0, arg1])
        inv.await_ready()

        # Get the result as a numpy array and print.
        results = inv.results
        result = results[0].map()
        result_ary = np.array(result, copy=False)
        return result_ary

    def invoke_tf():
      arg0 = a
      arg1 = b
      result = tf_f(arg0, arg1)
      return result.numpy()

    # Check that results are equal.
    self.assertAllEqual(invoke_iree(), invoke_tf())
    # Quick benchmark.
    iterations = 1000
    print("+++BM simple_mul:")
    iree_time = timeit.timeit(invoke_iree, number=iterations)
    print("IREE -> TIME/ITERATION =", (iree_time / iterations) * 1000, "ms")
    tf_time = timeit.timeit(invoke_tf, number=iterations)
    print("TF   -> TIME/ITERATION =", (tf_time / iterations) * 1000, "ms")
    tf_vs_iree_factor = tf_time / iree_time
    print("IREE VS TF SPEEDUP FACTOR =", tf_vs_iree_factor)

  def test_simple_scalar_mul_streamed(self):
    pyiree.tracing.enable_thread()
    # Initialize the runtime and register the module.
    # Use the CPU interpreter driver (which has the most implementation done):
    driver_name = "interpreter"

    # Live on the edge and give the vulkan driver a try:
    # driver_name = "vulkan"

    policy = pyiree.binding.rt.Policy()
    instance = pyiree.binding.rt.Instance(driver_name=driver_name)
    context = pyiree.binding.rt.Context(instance=instance, policy=policy)
    context.register_module(self.iree_vm_module)

    f = context.resolve_function("module.simple_mul")
    tf_f = self.tf_module.simple_mul
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    b = np.array([400., 5., 6., 7.], dtype=np.float32)

    iree_dispatch_event = pyiree.tracing.ScopedEvent(
        "SimpleArithmeticTest#simple_mul_dispatch")
    iree_await_event = pyiree.tracing.ScopedEvent(
        "SimpleArithmeticTest#simple_mul_await")

    invocations = []

    def invoke_iree():
      with iree_dispatch_event:
        arg0 = context.wrap_for_input(a)
        arg1 = context.wrap_for_input(b)

        # Invoke the function and wait for completion.
        inv = context.invoke(f, policy, [arg0, arg1])
        invocations.append(inv)

    def await_all():
      with iree_await_event:
        invocations[-1].await_ready()

    def invoke_tf():
      arg0 = a
      arg1 = b
      result = tf_f(arg0, arg1)
      return result.numpy()

    # Quick benchmark.
    iterations = 1000
    print("+++BM simple_mul_streamed:")
    iree_time = timeit.timeit(invoke_iree, number=iterations)
    iree_time += timeit.timeit(await_all, number=1)
    print("IREE -> TIME/ITERATION =", (iree_time / iterations) * 1000, "ms")
    tf_time = timeit.timeit(invoke_tf, number=iterations)
    print("TF   -> TIME/ITERATION =", (tf_time / iterations) * 1000, "ms")
    tf_vs_iree_factor = tf_time / iree_time
    print("IREE VS TF SPEEDUP FACTOR =", tf_vs_iree_factor)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
