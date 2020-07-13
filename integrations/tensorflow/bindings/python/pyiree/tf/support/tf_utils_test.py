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

from absl.testing import parameterized
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
          'target_backends': ['vmla'],
      },
      {
          'testcase_name': 'multiple_backends',
          'target_backends': ['vmla', 'llvm'],
      },
  ])
  def test_artifact_saving(self, target_backends):
    with tempfile.TemporaryDirectory() as artifacts_dir:
      tf_module = ConstantModule()
      iree_compiled_module = tf_utils.compile_tf_module(
          tf_module,
          target_backends=target_backends,
          artifacts_dir=artifacts_dir)

      artifacts_to_check = [
          'saved_model',
          f'tf_input__{"__".join(target_backends)}.mlir',
          f'iree_input__{"__".join(target_backends)}.mlir',
          f'compiled__{"__".join(target_backends)}.vmfb',
      ]
      for artifact in artifacts_to_check:
        self.assertTrue(os.path.exists(os.path.join(artifacts_dir, artifact)))

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
    info = tf_utils.BackendInfo.ALL[backend_name]
    module = tf_utils.CompiledModule.compile(StatefulCountingModule, info)

    # Test that incrementing works properly.
    self.assertEqual([0.], module.get_count())
    module.increment()
    self.assertEqual([1.], module.get_count())

    reinitialized_module = tf_utils.CompiledModule.from_existing(module)
    # Test reinitialization.
    self.assertEqual([0.], reinitialized_module.get_count())
    # Test independent state.
    self.assertEqual([1.], module.get_count())


if __name__ == '__main__':
  tf.test.main()
