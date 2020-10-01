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
"""Test MobileBERT.

Model topology and weights are from
https://github.com/google-research/google-research/tree/master/mobilebert
"""

import os
import posixpath

from absl import app
from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_quantized_weights", False,
                     "Whether to use quantized or floating point weights.")

MAX_SEQ_LENGTH = 384  # Max input sequence length used in mobilebert_squad.

FILE_NAME = "mobilebert_squad_savedmodels.tar.gz"
MODEL_URL = posixpath.join(
    "https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/",
    FILE_NAME)


class MobileBertSquad(tf.Module):
  """Wrapper of MobileBertSquad saved model v1."""

  def __init__(self):
    self.model_path = self.get_model_path()
    self.saved_model = tf.saved_model.load(self.model_path, tags=["serve"])
    self.inference_func = self.saved_model.signatures["serving_default"]

  @staticmethod
  def get_model_path():
    model_type = "quant_saved_model" if FLAGS.use_quantized_weights else "float"

    # Get_file will download the model weights from a publicly available folder,
    # save them to cache_dir=~/.keras/datasets/ and return a path to them.
    model_path = tf.keras.utils.get_file(FILE_NAME, MODEL_URL, untar=True)
    model_dir = os.path.dirname(model_path)
    extracted_name = FILE_NAME.split(".")[0]
    model_path = os.path.join(model_dir, extracted_name, model_type)
    return model_path

  @staticmethod
  def get_legacy_tflite_saved_model_converter_kwargs():
    return dict([("input_arrays", ["input_ids", "input_mask", "segment_ids"]),
                 ("output_arrays", ["start_logits", "end_logits"]),
                 ("exported_name", "predict"),
                 ("model_path", MobileBertSquad.get_model_path())])

  @tf.function(input_signature=[
      tf.TensorSpec((1, MAX_SEQ_LENGTH), tf.int32),
      tf.TensorSpec((1, MAX_SEQ_LENGTH), tf.int32),
      tf.TensorSpec((1, MAX_SEQ_LENGTH), tf.int32),
  ])
  def predict(self, input_ids, input_mask, segment_ids):
    inputs = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
    }
    return self.inference_func(**inputs)


class MobileBertSquadTest(tf_test_utils.TracedModuleTestCase):
  """Tests of MobileBertSquad."""

  def __init__(self, methodName="runTest"):
    super(MobileBertSquadTest, self).__init__(methodName)
    self._modules = tf_test_utils.compile_tf_module(MobileBertSquad,
                                                    exported_names=["predict"])

  def test_predict(self):

    def predict(module):
      input_ids = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
      input_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
      segment_ids = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)

      module.predict(input_ids, input_mask, segment_ids, atol=1e0)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
