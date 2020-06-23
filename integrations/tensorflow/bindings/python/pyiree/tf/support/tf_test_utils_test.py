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

from absl.testing import parameterized
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow as tf


class UtilsTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      {
          'testcase_name': 'all the same',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tgt_same': True,
      },
      {
          'testcase_name': 'wrong int',
          'array_c': np.array([1, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tgt_same': False,
      },
      {
          'testcase_name': 'wrong string',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['a', '1', '2']),
          'array_e': np.array([0.0, 0.1, 0.2]),
          'tgt_same': False,
      },
      {
          'testcase_name': 'wrong float',
          'array_c': np.array([0, 1, 2]),
          'array_d': np.array(['0', '1', '2']),
          'array_e': np.array([1.0, 0.1, 0.2]),
          'tgt_same': False,
      },
  ])
  def test_recursive_check_same(self, array_c, array_d, array_e, tgt_same):

    ref = {
        'a':
            1,
        'b': [{
            'c': np.array([0, 1, 2])
        }, {
            'd': np.array(['0', '1', '2'])
        }, {
            'e': np.array([0.0, 0.1, 0.2])
        }],
    }
    tgt = {
        'a': 1,
        'b': [{
            'c': array_c
        }, {
            'd': array_d
        }, {
            'e': array_e
        }],
    }
    same = tf_test_utils._recursive_check_same(ref, tgt)
    self.assertEqual(tgt_same, same)


if __name__ == '__main__':
  tf.test.main()
