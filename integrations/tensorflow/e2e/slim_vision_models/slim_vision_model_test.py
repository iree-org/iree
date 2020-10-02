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
"""Test all vision models from slim lib."""

import posixpath

from absl import app
from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow as tf
import tensorflow_hub as hub

FLAGS = flags.FLAGS

# Testing vision models from
# https://github.com/tensorflow/models/tree/master/research/slim
# slim models were designed with tf v1 and then coverted to SavedModel
# they are stored at tensorflow_hub.
flags.DEFINE_string(
    'model', 'mobilenet_v1_100_224', 'example model names: '
    '[resnet_v1_50, resnet_v1_101, resnet_v2_50, resnet_v2_101, '
    'mobilenet_v1_100_224, mobilenet_v1_025_224, mobilenet_v2_100_224, '
    'mobilenet_v2_035_224]\nAt least a subset can be viewed here:\n'
    'https://tfhub.dev/s?dataset=imagenet&module-type=image-classification,image-classifier'
)
flags.DEFINE_string('tf_hub_url', 'https://tfhub.dev/google/imagenet/',
                    'Base URL for the models to test')

# Classification mode; 4 - is a format of the model (SavedModel TF v2).
MODE = 'classification/4'
INPUT_SHAPE = (1, 224, 224, 3)


class SlimVisionModule(tf.Module):

  def __init__(self):
    super(SlimVisionModule, self).__init__()
    tf_utils.set_random_seed()
    model_path = posixpath.join(FLAGS.tf_hub_url, FLAGS.model, MODE)
    hub_layer = hub.KerasLayer(model_path)
    self.m = tf.keras.Sequential([hub_layer])
    self.m.build(INPUT_SHAPE)
    self.predict = tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE)])(
        self.m.call)


class SlimVisionTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, methodName="runTest"):
    super(SlimVisionTest, self).__init__(methodName)
    self._modules = tf_test_utils.compile_tf_module(SlimVisionModule,
                                                    exported_names=['predict'])

  def test_predict(self):

    def predict(module):
      input_data = np.random.rand(*INPUT_SHAPE).astype(np.float32)
      module.predict(input_data, atol=2e-5)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  SlimVisionModule.__name__ = FLAGS.model
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
