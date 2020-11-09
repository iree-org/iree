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

from absl import app
from absl import flags
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

from kws_streaming.layers import modes
from kws_streaming.models import model_flags
from kws_streaming.models import model_params
from kws_streaming.models import models
from kws_streaming.models import utils

FLAGS = flags.FLAGS

ALL_MODELS = list(model_params.HOTWORD_MODEL_PARAMS.keys())
MODELS_HELP = [f"'{name}'" for name in ALL_MODELS]
MODELS_HELP = f'{", ".join(MODELS_HELP[:-1])}, or {MODELS_HELP[-1]}'

flags.DEFINE_string(
    'model', 'svdf', f'Name of the model to compile. Either {MODELS_HELP}.\n'
    'See https://github.com/google-research/google-research/blob/master/kws_streaming/models/models.py#L38-L58'
)
flags.DEFINE_enum('mode', 'non_streaming',
                  ['non_streaming', 'internal_streaming', 'external_streaming'],
                  'Mode to execute the model in.')

MODE_ENUM_TO_MODE = {
    'non_streaming': modes.Modes.NON_STREAM_INFERENCE,
    'internal_streaming': modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
    'external_streaming': modes.Modes.STREAM_EXTERNAL_STATE_INFERENCE,
}


class KeywordSpottingModule(tf.Module):

  def __init__(self):
    super().__init__()
    self.m = utils.get_model_with_default_params(FLAGS.model,
                                                 MODE_ENUM_TO_MODE[FLAGS.mode])
    self.write_input_shapes_to_cls(self.m)
    self.m.predict = lambda x: self.m.call(x, training=False)
    input_signature = [tf.TensorSpec(shape) for shape in self.input_shapes]
    self.predict = tf.function(input_signature=input_signature)(self.m.predict)

  @classmethod
  def write_input_shapes_to_cls(cls, model):
    # We store the input shapes on the cls because we need access to them to
    # generate the random inputs. Lists are not valid exported names, so we
    # cannot access them from the module instance itself, and instead we store
    # the input shapes on the test case below.
    cls.input_shapes = [tensor.shape for tensor in model.inputs]


class KeywordSpottingTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(KeywordSpottingModule,
                                                    exported_names=['predict'])
    self._input_shapes = KeywordSpottingModule.input_shapes

  def test_predict(self):

    def predict(module):
      inputs = [tf_utils.uniform(shape) for shape in self._input_shapes]
      inputs = inputs[0] if len(inputs) == 1 else inputs
      module.predict(inputs, atol=1e-5)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.model not in ALL_MODELS:
    raise ValueError(f'Unsupported model: {FLAGS.model}.\n'
                     f'Expected one of {MODELS_HELP}.')
  KeywordSpottingModule.__name__ = os.path.join('keyword_spotting', FLAGS.model,
                                                FLAGS.mode)

  tf.test.main()


if __name__ == '__main__':
  app.run(main)
