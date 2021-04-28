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
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
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
flags.DEFINE_string(
    'tf_hub_url', None, 'Base URL for the models to test. URL at the time of '
    'writing:\nhttps://tfhub.dev/google/imagenet/')

LARGE_MODELS = ['amoebanet_a_n18_f448', "nasnet_large", "pnasnet_large"]


def get_input_shape():
  if FLAGS.model in LARGE_MODELS:
    return (1, 331, 331, 3)
  elif FLAGS.model.startswith('mobilenet_v2'):
    # The MobileNetV2 models have variable size that seems to be only inferrible
    # from their TFHub name.
    size = int(FLAGS.model.split('_')[-1])
    return (1, size, size, 3)
  elif FLAGS.model.startswith('mobilenet_v3_large'):
    size = int(FLAGS.model.split('_')[-1])
    return (1, size, size, 3)
  else:
    # Default input shape.
    return (1, 224, 224, 3)


def get_mode(model_name):
  if model_name.startswith('mobilenet_v3'):
    return 'classification/5'
  # Classification mode; 4 - is a format of the model (SavedModel TF v2).
  return 'classification/4'


class SlimVisionModule(tf.Module):

  def __init__(self):
    super().__init__()
    tf_utils.set_random_seed()
    model_path = posixpath.join(FLAGS.tf_hub_url, FLAGS.model,
                                get_mode(FLAGS.model))
    hub_layer = hub.KerasLayer(model_path)
    self.m = tf.keras.Sequential([hub_layer])
    input_shape = get_input_shape()
    self.m.build(input_shape)
    self.predict = tf.function(input_signature=[tf.TensorSpec(input_shape)])(
        self.m.call)


class SlimVisionTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        SlimVisionModule,
        exported_names=['predict'],
        relative_artifacts_dir=FLAGS.model)

  def test_predict(self):

    def predict(module):
      input_data = np.random.rand(*get_input_shape()).astype(np.float32)
      # Only TF vs. TF passes at the default atol.
      module.predict(input_data, atol=5e-5)

    self.compare_backends(predict, self._modules)


def main(argv):
  del argv  # Unused.
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
