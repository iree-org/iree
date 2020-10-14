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
"""Test Keras hotword models."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


import os
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve())
# Remove `integrations/tensorflow/e2e/keras/test_name.py`.
base_path = os.sep.join(base_path.split(os.sep)[:-5])
research_path = os.path.join(base_path, 'third_party/google-research')
logging.warning(research_path)
sys.path.append(research_path)

from kws_streaming.models import model_params
from kws_streaming.models import models
from kws_streaming.train import model_flags

FLAGS = flags.FLAGS

# Testing all models automatically can take time
# so we test it one by one, with argument --model=svdf
flags.DEFINE_string(
    'model', 'svdf', 'model name, it supports: '
    'svdf, ds_cnn_stride, gru, lstm, cnn_stride'
    'cnn, crnn, dnn, att_rnn, att_mh_rnn')

NON_STREAM_INPUT_SHAPE = [1, 16000]


def initialize_model():
  tf_utils.set_random_seed()
  tf.keras.backend.set_learning_phase(False)  # TODO(meadowlark)

  p = model_params.HOTWORD_MODEL_PARAMS[FLAGS.model]
  p = model_flags.update_flags(p)
  return models.MODELS[p.model_name](p)


class HotwordModule(tf.Module):

  def __init__(self):
    super(HotwordModule, self).__init__()
    self.m = initialize_model()
    self.predict = tf.function(
        input_signature=[tf.TensorSpec(NON_STREAM_INPUT_SHAPE, np.float32)])(
            self.m.call)


class AppTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    super(AppTest, self).__init__(methodName)
    self._modules = tf_test_utils.compile_tf_module(
        HotwordModule, exported_names=['predict'])

  def test_predict(self):

    def predict(module):
      module.predict(tf_utils.uniform(NON_STREAM_INPUT_SHAPE), atol=1e-5)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.model not in model_params.HOTWORD_MODEL_PARAMS:
    raise ValueError(f'Unsupported model: {FLAGS.model}')
  HotwordModule.__name__ = FLAGS.model

  tf.test.main()


if __name__ == '__main__':
  app.run(main)
