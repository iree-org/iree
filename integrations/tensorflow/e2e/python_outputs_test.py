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
"""Tests of tf.functions returning common python outputs types."""

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class PythonOutputsModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def tensor_output(self, a):
    return a

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def tuple_output_len_one(self, a):
    return (a,)

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def tuple_output_len_two(self, a):
    return (a, a * a)

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def tuple_output_len_three(self, a):
    return (a, a * a, a * a * a)

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def list_output_len_one(self, a):
    return [a]

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def list_output_len_two(self, a):
    return [a, a * a]

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def list_output_len_three(self, a):
    return [a, a * a, a * a * a]

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def dict_output_len_one(self, a):
    return {"a": a}

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def dict_output_len_two(self, a):
    return {"a": a, "a * a": a * a}

  @tf.function(input_signature=[tf.TensorSpec([2], tf.float32)])
  def dict_output_len_three(self, a):
    return {"a": a, "a * a": a * a, "a * a * a": a * a * a}


class PythonOutputsTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(PythonOutputsModule)

  # yapf: disable
  def test_tensor_output(self):
    def tensor_output(module):
      module.tensor_output(np.array([1, 2], dtype=np.float32))
    self.compare_backends(tensor_output, self._modules)

  def test_tuple_output_len_one(self):
    def tuple_output_len_one(module):
      module.tuple_output_len_one(np.array([1, 2], dtype=np.float32))
    self.compare_backends(tuple_output_len_one, self._modules)

  def test_tuple_output_len_two(self):
    def tuple_output_len_two(module):
      module.tuple_output_len_two(np.array([1, 2], dtype=np.float32))
    self.compare_backends(tuple_output_len_two, self._modules)

  def test_tuple_output_len_three(self):
    def tuple_output_len_three(module):
      module.tuple_output_len_three(np.array([1, 2], dtype=np.float32))
    self.compare_backends(tuple_output_len_three, self._modules)

  def test_list_output_len_one(self):
    def list_output_len_one(module):
      module.list_output_len_one(np.array([1, 2], dtype=np.float32))
    self.compare_backends(list_output_len_one, self._modules)

  def test_list_output_len_two(self):
    def list_output_len_two(module):
      module.list_output_len_two(np.array([1, 2], dtype=np.float32))
    self.compare_backends(list_output_len_two, self._modules)

  def test_list_output_len_three(self):
    def list_output_len_three(module):
      module.list_output_len_three(np.array([1, 2], dtype=np.float32))
    self.compare_backends(list_output_len_three, self._modules)

  def test_dict_output_len_one(self):
    def dict_output_len_one(module):
      module.dict_output_len_one(np.array([1, 2], dtype=np.float32))
    self.compare_backends(dict_output_len_one, self._modules)

  def test_dict_output_len_two(self):
    def dict_output_len_two(module):
      module.dict_output_len_two(np.array([1, 2], dtype=np.float32))
    self.compare_backends(dict_output_len_two, self._modules)

  def test_dict_output_len_three(self):
    def dict_output_len_three(module):
      module.dict_output_len_three(np.array([1, 2], dtype=np.float32))
    self.compare_backends(dict_output_len_three, self._modules)
  # yapf: enable


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
