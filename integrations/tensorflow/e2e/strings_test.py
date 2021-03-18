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

import string

from absl import app
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf


class StringsModule(tf.Module):
  """A Module for converting a set of ids to the concatenated string."""

  def __init__(self):
    wordparts = [str(c) for c in string.printable]
    self.wordparts = tf.constant(wordparts, tf.string)

  @tf.function(input_signature=[
      tf.TensorSpec((None, None), dtype=tf.int32),
  ])
  def print_ids(self, ids):
    string_tensor = tf.strings.as_string(ids)
    tf.print(string_tensor)

  @tf.function(input_signature=[
      tf.TensorSpec((None,), dtype=tf.int32),
      tf.TensorSpec((None,), dtype=tf.int32),
  ])
  def gather(self, string_values, indices):
    tf.print(tf.gather(tf.as_string(string_values), indices))

#  @tf.function(input_signature=[
#      tf.TensorSpec((None, None), dtype=tf.int32),
#  ])
#  def strings_to_ids(self, ids):
#    wps = tf.gather(self.wordparts, ids)
#    return tf.strings.reduce_join(wps, 1)


class StringsTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(StringsModule)

  def test_print_ids(self):

    def print_ids(module):
      input_ids = np.asarray([[1, 2, 3, 4, 5, 6], [10, 11, 12, 14, 15, 16]],
                             dtype=np.int32)
      module.print_ids(input_ids)

    self.compare_backends(print_ids, self._modules)

  def test_gather(self):

    def gather(module):
      string_values = np.asarray([ord(c) for c in string.printable],
                                 dtype=np.int32)
      input_indices = np.asarray([12, 10, 29, 21, 10, 34], dtype=np.int32)
      module.gather(string_values, input_indices)

    self.compare_backends(gather, self._modules)


#  def test_strings_to_ids(self):
#
#    def strings_to_ids(module):
#      input_ids = np.asarray(
#          [[12, 10, 29, 28, 94, 15, 24, 27, 94, 25, 21, 10, 34],
#           [13, 24, 16, 28, 94, 15, 24, 27, 94, 28, 29, 10, 34]])
#      module.strings_to_ids(input_ids)
#
#    self.compare_backends(strings_to_ids, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
