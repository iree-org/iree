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


class UtilsTests(tf.test.TestCase, parameterized.TestCase):

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
    same = tf_test_utils.Trace._check_same(ref, tar, rtol=1e-6, atol=1e-6)
    self.assertEqual(tar_same, same)

  def test_trace_inputs_and_outputs(self):

    def trace_function(module):
      # No inputs or outputs
      module.increment()
      # Only inputs
      module.increment_by(np.array([81.], dtype=np.float32))
      # Only outputs
      module.get_count()

    module = tf_utils.TfCompiledModule(StatefulCountingModule,
                                       tf_utils.BackendInfo('tf'))
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

    tf_module = tf_utils.TfCompiledModule(StatefulCountingModule,
                                          tf_utils.BackendInfo('tf'))
    tf_trace = tf_test_utils.Trace(tf_module, tf_function)
    tf_function(tf_test_utils.TracedModule(tf_module, tf_trace))

    vmla_module = tf_utils.IreeCompiledModule(StatefulCountingModule,
                                              tf_utils.BackendInfo('iree_vmla'))
    vmla_trace = tf_test_utils.Trace(vmla_module, vmla_function)
    vmla_function(tf_test_utils.TracedModule(vmla_module, vmla_trace))

    with self.assertRaises(ValueError):
      tf_test_utils.Trace.compare_traces(tf_trace, vmla_trace)

  def test_nonmatching_inputs(self):

    def tf_function(module):
      module.increment_by(np.array([42.], dtype=np.float32))

    def vmla_function(module):
      module.increment_by(np.array([22.], dtype=np.float32))

    tf_module = tf_utils.TfCompiledModule(StatefulCountingModule,
                                          tf_utils.BackendInfo('tf'))
    tf_trace = tf_test_utils.Trace(tf_module, tf_function)
    tf_function(tf_test_utils.TracedModule(tf_module, tf_trace))

    vmla_module = tf_utils.IreeCompiledModule(StatefulCountingModule,
                                              tf_utils.BackendInfo('iree_vmla'))
    vmla_trace = tf_test_utils.Trace(vmla_module, vmla_function)
    vmla_function(tf_test_utils.TracedModule(vmla_module, vmla_trace))

    self.assertFalse(tf_test_utils.Trace.compare_traces(tf_trace, vmla_trace))

  def test_trace_serialize_and_load(self):

    def trace_function(module):
      module.increment()
      module.increment_by(np.array([81.], dtype=np.float32))
      module.increment_by_max(
          np.array([81], dtype=np.float32), np.array([92], dtype=np.float32))
      module.get_count()

    module = tf_utils.TfCompiledModule(StatefulCountingModule,
                                       tf_utils.BackendInfo('tf'))
    trace = tf_test_utils.Trace(module, trace_function)
    trace_function(tf_test_utils.TracedModule(module, trace))

    with tempfile.TemporaryDirectory() as artifacts_dir:
      trace_function_dir = tf_test_utils._get_trace_dir(artifacts_dir, trace)
      trace.serialize(trace_function_dir)
      loaded_trace = tf_test_utils.Trace.load(trace_function_dir)

      # Check all calls match.
      self.assertTrue(tf_test_utils.Trace.compare_traces(trace, loaded_trace))

      # Check all other metadata match.
      self.assertAllEqual(trace.__dict__.keys(), loaded_trace.__dict__.keys())
      for key in trace.__dict__.keys():
        if key != 'calls':
          self.assertEqual(trace.__dict__[key], loaded_trace.__dict__[key])


if __name__ == '__main__':
  tf.test.main()
