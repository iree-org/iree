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
"""Tests of streamable Keyword Spotting models implemented in Keras."""

import os
import sys
import pathlib

from absl import app
from absl import flags
from absl import logging
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

# We need to make python aware of the kws_streaming module in third_party, which
# does not come with bazel BUILD files.
source_path = str(pathlib.Path(__file__).resolve())
iree_dir = source_path.replace(
    "integrations/tensorflow/e2e/keras/keyword_spotting_streaming_test.py", "")
google_research_dir = os.path.join(iree_dir, 'third_party/google-research')
sys.path.append(google_research_dir)

from kws_streaming.layers import modes
from kws_streaming.models import model_params
from kws_streaming.models import models
from kws_streaming.models import utils
from kws_streaming.train import model_flags
FLAGS = flags.FLAGS

ALL_MODELS = list(model_params.HOTWORD_MODEL_PARAMS.keys())
MODELS_HELP = [f"'{name}'" for name in ALL_MODELS]
MODELS_HELP = f'{", ".join(MODELS_HELP[:-1])}, or {MODELS_HELP[-1]}'

flags.DEFINE_string(
    'model', 'svdf', f'Name of the model to compile. Either {MODELS_HELP}.\n'
    'See https://github.com/google-research/google-research/blob/master/kws_streaming/models/models.py#L38-L58'
)
flags.DEFINE_enum('mode', 'non_streaming',
                  ['non_streaming', 'internal_streaming'],
                  'Mode to execute the model in.')

MODE_ENUM_TO_MODE = {
    'non_streaming': modes.Modes.NON_STREAM_INFERENCE,
    'internal_streaming': modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
}
MODE_TO_INPUT_SHAPE = {
    'non_streaming': (1, 16000),
    'internal_streaming': (1, 320),
}


def get_input_shape():
  return MODE_TO_INPUT_SHAPE[FLAGS.mode]


def initialize_model():
  params = model_params.HOTWORD_MODEL_PARAMS[FLAGS.model]
  params = model_flags.update_flags(params)
  model = models.MODELS[params.model_name](params)

  if FLAGS.mode == 'internal_streaming':
    mode = MODE_ENUM_TO_MODE[FLAGS.mode]
    input_shape = get_input_shape()
    params.batch_size = input_shape[0]
    params.desired_samples = input_shape[1]
    model = utils.to_streaming_inference(model, flags=params, mode=mode)

  return model


class KeywordSpottingModule(tf.Module):

  def __init__(self):
    super().__init__()
    self.m = initialize_model()
    self.m.predict = lambda x: self.m.call(x, training=False)
    self.predict = tf.function(
        input_signature=[tf.TensorSpec(get_input_shape())])(self.m.predict)


class KeywordSpottingTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(KeywordSpottingModule,
                                                    exported_names=['predict'])

  def test_predict(self):

    def predict(module):
      module.predict(tf_utils.uniform(get_input_shape()), atol=1e-5)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.model not in ALL_MODELS:
    raise ValueError(f'Unsupported model: {FLAGS.model}.\n'
                     f'Expected one of {MODELS_HELP}.')
  KeywordSpottingModule.__name__ = f'kws_{FLAGS.model}'

  tf.test.main()


if __name__ == '__main__':
  app.run(main)
