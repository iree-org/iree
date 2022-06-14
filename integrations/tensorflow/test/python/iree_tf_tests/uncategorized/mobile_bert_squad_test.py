# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test MobileBERT.

Model topology and weights are from
https://github.com/google-research/google-research/tree/master/mobilebert
"""

import os
import posixpath

from absl import app
from absl import flags
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_quantized_weights', False,
                     'Whether to use quantized or floating point weights.')

MAX_SEQ_LENGTH = 384  # Max input sequence length used in mobilebert_squad.

FILE_NAME = 'mobilebert_squad_savedmodels'
MODEL_URL = posixpath.join(
    f'https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/{FILE_NAME}.tar.gz'
)


class MobileBertSquadTest(tf_test_utils.TracedModuleTestCase):
  """Tests of MobileBertSquad."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    model_type = 'quant_saved_model' if FLAGS.use_quantized_weights else 'float'

    # Get_file will download the model weights from a publicly available folder,
    # save them to cache_dir=~/.keras/datasets/ and return a path to them.
    model_path = tf.keras.utils.get_file(
        FILE_NAME,
        MODEL_URL,
        untar=True,
        cache_dir=tf_test_utils._setup_artifacts_dir("download"))
    model_dir = os.path.dirname(model_path)
    extracted_name = FILE_NAME.split('.')[0]
    model_path = os.path.join(model_dir, extracted_name, model_type)

    self._modules = tf_test_utils.compile_tf_signature_def_saved_model(
        saved_model_dir=model_path,
        saved_model_tags=set(['serve']),
        module_name='MobileBertSquad',
        exported_name='serving_default',
        input_names=['input_ids', 'input_mask', 'segment_ids'],
        output_names=['start_logits', 'end_logits'])

  def test_serving_default(self):

    def serving_default(module):
      input_ids = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
      input_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
      segment_ids = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)

      module.serving_default(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             atol=1e0)

    self.compare_backends(serving_default, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
