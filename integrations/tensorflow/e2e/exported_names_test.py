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

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class DontExportEverythingModule(tf.Module):

  @tf.function(input_signature=[])
  def valid_fn(self):
    return tf.constant([42.])

  # Here an input_signature is elided so that the SavedModel importer will raise
  # an error if it attempts to compile this function.
  def invalid_fn(self):
    return tf.constant([24.])


@tf_test_utils.compile_module(
    DontExportEverythingModule, exported_names=["exported_fn"])
class DontExportEverythingTest(tf_test_utils.SavedModelTestCase):

  def test_exported_name(self):
    # This implicitly tests that invalid_fn is not exported because
    # DontExportEverythingModule is compiled to run this test.
    self.get_module().valid_fn().assert_all_close()

  def test_unreachable_name(self):
    # Additionally check that the function name is unreachable.
    with self.assertRaises(AttributeError):
      self.get_module().invalid_fn()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
