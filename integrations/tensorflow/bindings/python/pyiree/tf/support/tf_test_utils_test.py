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
"""Tests for pyiree.tf.support.tf_test_utils."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow as tf


class StatefulCountingModule(tf.Module):

  def __init__(self):
    self.count = tf.Variable([0.])

  @tf.function(input_signature=[])
  def increment(self):
    self.count.assign_add(tf.constant([1.]))

  @tf.function(input_signature=[])
  def get_count(self):
    return self.count

  @tf.function(input_signature=[tf.TensorSpec([1])])
  def increment_by(self, value):
    self.count.assign_add(value)

  @tf.function(input_signature=[tf.TensorSpec([1]), tf.TensorSpec([1])])
  def increment_by_max(self, a, b):
    result = tf.maximum(a, b)
    self.count.assign_add(result)
    return result

  @tf.function(input_signature=[])
  def decrement(self):
    self.count.assign_sub(tf.constant([1.]))


class TfFunctionUnittestModule(tf_test_utils.TestModule):

  @tf_test_utils.tf_function_unittest(input_signature=[])
  def no_args(self):
    return np.array([True], dtype=np.bool)

  @tf_test_utils.tf_function_unittest(input_signature=[
      tf.TensorSpec([4]),
      tf.TensorSpec([4]),
  ])
  def default_uniform_inputs(self, a, b):
    return a + b

  @tf_test_utils.tf_function_unittest(
      input_signature=[
          tf.TensorSpec([4]),
          tf.TensorSpec([4]),
      ],
      input_generator=tf_utils.ndarange,
  )
  def custom_input_generator(self, a, b):
    return a + b

  @tf_test_utils.tf_function_unittest(
      input_signature=[
          tf.TensorSpec([4]),
          tf.TensorSpec([4]),
      ],
      input_args=[
          np.array([0, 1, 2, 3], np.float32),
          -np.array([0, 1, 2, 3], np.float32),
      ],
  )
  def custom_input_args(self, a, b):
    return a + b

  # This test will fail if atol is not successfully set.
  @tf_test_utils.tf_function_unittest(
      input_signature=[
          tf.TensorSpec([128, 3072], tf.float32),
          tf.TensorSpec([3072, 256], tf.float32),
      ],
      atol=1e-2,
  )
  def high_tolerance(self, a, b):
    return tf.matmul(a, b)


class TestUtilsTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': 'all the same',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': True,
      },
      {
          'testcase_name': 'wrong int',
          'array_c': np.array([1, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': False,
      },
      {
          'testcase_name': 'wrong string',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['a', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tar_same': False,
      },
      {
          'testcase_name': 'wrong float',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([1.0, 0.1, 0.2]),
          'tar_same': False,
      },
  ])
  def test_recursive_check_same(self, array_c, array_d, array_e, tar_same):

    # yapf: disable
    ref = {
        'a': 1,
        'b': [
            {'c': np.array([0, 1, 2])},
            {'d': np.array(['0', '1', '2'])},
            {'e': np.array([0.0, 0.1, 0.2])}
        ],
    }
    tar = {
        'a': 1,
        'b': [
            {'c': array_c},
            {'d': array_d},
            {'e': array_e}
        ],
    }
    # yapf: enable
    same, _ = tf_test_utils.Trace._check_same(ref, tar, rtol=1e-6, atol=1e-6)
    self.assertEqual(tar_same, same)

  def test_trace_inputs_and_outputs(self):

    def trace_function(module):
      # No inputs or outputs
      module.increment()
      # Only inputs
      module.increment_by(np.array([81.], dtype=np.float32))
      # Only outputs
      module.get_count()

    module = tf_utils.TfCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('tf'))
    trace = tf_test_utils.Trace(module, trace_function)
    trace_function(tf_test_utils.TracedModule(module, trace))

    self.assertIsInstance(trace.calls[0].inputs, tuple)
    self.assertEmpty(trace.calls[0].inputs)
    self.assertIsInstance(trace.calls[0].outputs, tuple)
    self.assertEmpty(trace.calls[0].outputs)

    self.assertAllClose(trace.calls[1].inputs[0], [81.])
    self.assertAllClose(trace.calls[2].outputs[0], [82.])

  def test_nonmatching_methods(self):

    def tf_function(module):
      module.increment()
      module.increment()

    def vmla_function(module):
      module.increment()
      module.decrement()

    tf_module = tf_utils.TfCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('tf'))
    tf_trace = tf_test_utils.Trace(tf_module, tf_function)
    tf_function(tf_test_utils.TracedModule(tf_module, tf_trace))

    vmla_module = tf_utils.IreeCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('iree_vmla'))
    vmla_trace = tf_test_utils.Trace(vmla_module, vmla_function)
    vmla_function(tf_test_utils.TracedModule(vmla_module, vmla_trace))

    with self.assertRaises(ValueError):
      tf_test_utils.Trace.compare_traces(tf_trace, vmla_trace)

  def test_nonmatching_inputs(self):

    def tf_function(module):
      module.increment_by(np.array([42.], dtype=np.float32))

    def vmla_function(module):
      module.increment_by(np.array([22.], dtype=np.float32))

    tf_module = tf_utils.TfCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('tf'))
    tf_trace = tf_test_utils.Trace(tf_module, tf_function)
    tf_function(tf_test_utils.TracedModule(tf_module, tf_trace))

    vmla_module = tf_utils.IreeCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('iree_vmla'))
    vmla_trace = tf_test_utils.Trace(vmla_module, vmla_function)
    vmla_function(tf_test_utils.TracedModule(vmla_module, vmla_trace))

    same, error_messages = tf_test_utils.Trace.compare_traces(
        tf_trace, vmla_trace)
    self.assertFalse(same)

  def test_trace_serialize_and_load(self):

    def trace_function(module):
      module.increment()
      module.increment_by(np.array([81.], dtype=np.float32))
      module.increment_by_max(np.array([81], dtype=np.float32),
                              np.array([92], dtype=np.float32))
      module.get_count()

    module = tf_utils.IreeCompiledModule.create_from_class(
        StatefulCountingModule, tf_utils.BackendInfo('iree_vmla'))
    trace = tf_test_utils.Trace(module, trace_function)
    trace_function(tf_test_utils.TracedModule(module, trace))

    with tempfile.TemporaryDirectory() as artifacts_dir:
      trace_function_dir = tf_test_utils._get_trace_dir(artifacts_dir, trace)
      trace.serialize(trace_function_dir)
      self.assertTrue(
          os.path.exists(os.path.join(trace_function_dir, 'metadata.pkl')))
      loaded_trace = tf_test_utils.Trace.load(trace_function_dir)

      # Check all calls match.
      self.assertTrue(tf_test_utils.Trace.compare_traces(trace, loaded_trace))

      # Check all other metadata match.
      self.assertAllEqual(trace.__dict__.keys(), loaded_trace.__dict__.keys())
      for key in trace.__dict__.keys():
        if key != 'calls':
          self.assertEqual(trace.__dict__[key], loaded_trace.__dict__[key])

  def test_tf_function_unittet(self):

    class TfFunctionUnittestTest(tf_test_utils.TracedModuleTestCase):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._modules = tf_test_utils.compile_tf_module(
            TfFunctionUnittestModule)

    TfFunctionUnittestTest.generate_unittests(TfFunctionUnittestModule)
    test_case = TfFunctionUnittestTest()
    self.assertTrue(hasattr(test_case, 'test_no_args'))
    self.assertTrue(hasattr(test_case, 'test_default_uniform_inputs'))
    self.assertTrue(hasattr(test_case, 'test_custom_input_generator'))
    self.assertTrue(hasattr(test_case, 'test_custom_input_args'))
    self.assertTrue(hasattr(test_case, 'test_high_tolerance'))

    # Will throw an error if 'atol' and 'rtol' are not set.
    test_case = TfFunctionUnittestTest()
    test_case.test_high_tolerance()


if __name__ == '__main__':
  tf.test.main()
