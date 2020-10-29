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
"""Test FaceSSD v4."""

import os
import posixpath

from absl import app
from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
from pyiree.tf import compiler
import tensorflow.compat.v2 as tf


class FaceSsdP20Test(tf_test_utils.TracedModuleTestCase):
  """Tests of FaceSSd."""

  def __init__(self, *args, **kwargs):
    super(FaceSsdP20Test, self).__init__(*args, **kwargs)

    self._modules = tf_test_utils.compile_tf_signature_def_saved_model(
        saved_model_dir='third_party/mlir_edge/model_curriculum/saved_models/facessd_p20/facessd_p20.sm',
        saved_model_tags=set(['serve']),
        module_name='FaceSsdP20',
        exported_name='serving_default',
        input_names=['normalized_input_image_tensor'],
        output_names=[
            'raw_outputs/box_encodings', 'raw_outputs/class_predictions'
        ])

  def test_serving_default(self):

    def serving_default(module):
      input_tensor = tf_utils.uniform((1, 240, 320, 1))
      module.serving_default(
          normalized_input_image_tensor=input_tensor, atol=5e-5, rtol=1e-5)

    self.compare_backends(serving_default, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
