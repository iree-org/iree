# Lint as: python3
# Copyright 2020 Google LLC
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
"""Tests for pyiree.tf.support.tf_utils."""

import os
import tempfile

from absl import logging
from absl.testing import parameterized
import numpy as np
from pyiree.tf.support import tf_utils
import tensorflow as tf


class ConstantModule(tf.Module):

  @tf.function(input_signature=[])
  def meaning(self):
    return tf.constant([42.])


class StatefulCountingModule(tf.Module):

  def __init__(self):
    self.count = tf.Variable([0.])

  @tf.function(input_signature=[])
  def increment(self):
    self.count.assign_add(tf.constant([1.]))

  @tf.function(input_signature=[])
  def get_count(self):
    return self.count


class UtilsTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': 'single_backend',
          'backend_infos': [tf_utils.BackendInfo('iree_vmla')],
      },
      {
          'testcase_name':
              'multiple_backends',
          'backend_infos': [
              tf_utils.BackendInfo('iree_vmla'),
              tf_utils.BackendInfo('iree_llvmjit')
          ],
      },
  ])
  def test_artifact_saving(self, backend_infos):
    with tempfile.TemporaryDirectory() as artifacts_dir:
      tf_module = ConstantModule()
      iree_compiled_module = tf_utils.compile_tf_module(
          tf_module, backend_infos=backend_infos, artifacts_dir=artifacts_dir)

      compiled_path = tf_utils._get_backends_path('compiled', backend_infos,
                                                  artifacts_dir)
      compiled_path = f'{compiled_path}.vmfb'
      artifacts_to_check = [
          'tf_input.mlir',
          'iree_input.mlir',
          compiled_path,
      ]
      for artifact in artifacts_to_check:
        artifact_path = os.path.join(artifacts_dir, artifact)
        logging.info('Checking path: %s', artifact_path)
        self.assertTrue(os.path.exists(artifact_path))

  @parameterized.named_parameters([
      {
          'testcase_name': 'tensorflow',
          'backend_name': 'tf',
      },
      {
          'testcase_name': 'vmla',
          'backend_name': 'iree_vmla',
      },
  ])
  def test_unaltered_state(self, backend_name):
    backend_info = tf_utils.BackendInfo(backend_name)
    module = backend_info.compile(StatefulCountingModule)

    # Test that incrementing works properly.
    self.assertEqual([0.], module.get_count())
    module.increment()
    self.assertEqual([1.], module.get_count())

    reinitialized_module = module.create_reinitialized()
    # Test reinitialization.
    self.assertEqual([0.], reinitialized_module.get_count())
    # Test independent state.
    self.assertEqual([1.], module.get_count())

  def test_to_mlir_type(self):
    self.assertEqual('i8', tf_utils.to_mlir_type(np.dtype('int8')))
    self.assertEqual('i32', tf_utils.to_mlir_type(np.dtype('int32')))
    self.assertEqual('f32', tf_utils.to_mlir_type(np.dtype('float32')))
    self.assertEqual('f64', tf_utils.to_mlir_type(np.dtype('float64')))

  def test_save_input_values(self):
    inputs = [np.array([1, 2], dtype=np.int32)]
    self.assertEqual('2xi32=1 2', tf_utils.save_input_values(inputs))
    inputs = [np.array([1, 2], dtype=np.float32)]
    self.assertEqual('2xf32=1.0 2.0', tf_utils.save_input_values(inputs))


if __name__ == '__main__':
  tf.test.main()
