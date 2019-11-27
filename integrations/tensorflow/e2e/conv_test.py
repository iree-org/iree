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

import os

from absl.testing import absltest
import numpy as np
from pyiree import binding
import tensorflow as tf

SAVE_PATH = "/tmp"


def create_conv(img_shape, kernel_shape, padding):

  class ConvModule(tf.Module):

    def __init__(self):
      pass

    @tf.function(input_signature=[
        tf.TensorSpec(img_shape, tf.float32),
        tf.TensorSpec(kernel_shape, tf.float32)
    ])
    def function(self, img, kernel):
      return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], padding, name="result")

  return ConvModule()


def baseline_conv(img, kernel, padding):
  return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], padding, name="result").numpy()


def run_conv(img, kernel, padding):
  save_model_path = os.path.join(SAVE_PATH, "conv.sm")
  conv = create_conv(list(img.shape), list(kernel.shape), padding)
  options = tf.saved_model.SaveOptions(save_debug_info=True)
  tf.saved_model.save(conv, save_model_path, options=options)

  ctx = binding.compiler.CompilerContext()
  input_module = binding.tf_interop.load_saved_model(ctx, save_model_path)
  input_module.run_pass_pipeline([
      "tf-executor-graph-pruning",
      "tf-standard-pipeline",
      "canonicalize",
      "xla-legalize-tf",
      "xla-legalize-tf-control-flow",
      "xla-legalize-control-flow",
      "convert-from-tuple-calling-convention",
      "canonicalize",
  ])
  blob = input_module.compile_to_sequencer_blob(
      target_backends=["vulkan-spirv"])
  m = binding.vm.create_module_from_blob(blob)
  policy = binding.rt.Policy()
  instance = binding.rt.Instance(driver_name="vulkan")
  context = binding.rt.Context(instance=instance, policy=policy)
  context.register_module(m)

  f = context.resolve_function("module.function")

  input_tensor = context.wrap_for_input(img)
  input_kernel = context.wrap_for_input(kernel)

  invocation = context.invoke(f, policy, [input_tensor, input_kernel])
  invocation.await_ready()

  return np.array(invocation.results[0].map(), copy=True)


class ConvTest(absltest.TestCase):

  def test_id_batch_size_1(self):
    i = np.arange(20, dtype=np.float32).reshape([1, 4, 5, 1])
    k = np.ones([1, 1, 1, 1], dtype=np.float32)
    r = run_conv(i, k, "VALID")
    g = baseline_conv(i, k, "VALID")
    np.testing.assert_array_equal(r, g)

  def test_id_batch_size_2(self):
    i = np.arange(40, dtype=np.float32).reshape([2, 4, 5, 1])
    k = np.ones([1, 1, 1, 1], dtype=np.float32)
    r = run_conv(i, k, "VALID")
    g = baseline_conv(i, k, "VALID")
    np.testing.assert_array_equal(r, g)

  def test_asym_kernel(self):
    i = np.arange(20, dtype=np.float32).reshape([1, 4, 5, 1])
    k = np.array([[1, 4, 2], [-2, 0, 1]], dtype=np.float32).reshape(2, 3, 1, 1)
    r = run_conv(i, k, "VALID")
    g = baseline_conv(i, k, "VALID")
    np.testing.assert_array_equal(r, g)

  def test_padding(self):
    i = np.arange(20, dtype=np.float32).reshape([1, 4, 5, 1])
    k = np.array([[1, 4, 2], [-2, 0, 1]], dtype=np.float32).reshape(2, 3, 1, 1)
    r = run_conv(i, k, "SAME")
    g = baseline_conv(i, k, "SAME")
    np.testing.assert_array_equal(r, g)

  def test_batched_padding(self):
    i = np.arange(40, dtype=np.float32).reshape([2, 4, 5, 1])
    k = np.array([[1, 4, 2], [-2, 0, 1]], dtype=np.float32).reshape(2, 3, 1, 1)
    r = run_conv(i, k, "SAME")
    g = baseline_conv(i, k, "SAME")
    np.testing.assert_array_equal(r, g)

  def test_feature_reduce(self):
    i = np.arange(40, dtype=np.float32).reshape([1, 4, 5, 2])
    k = np.ones([3, 2, 2, 1], dtype=np.float32)
    r = run_conv(i, k, "SAME")
    g = baseline_conv(i, k, "SAME")
    np.testing.assert_array_equal(r, g)


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  absltest.main()
