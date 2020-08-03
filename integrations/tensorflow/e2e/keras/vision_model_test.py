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
"""Test all applications models in Keras."""

import os

from absl import app
from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# Testing all applications models automatically can take time
# so we test it one by one, with argument --model=MobileNet
flags.DEFINE_string('model', 'ResNet50', 'model name')
flags.DEFINE_string(
    'url', '', 'url with model weights '
    'for example https://storage.googleapis.com/iree_models/')
flags.DEFINE_enum('data', 'cifar10', ['cifar10', 'imagenet'],
                  'data sets on which model was trained: imagenet, cifar10')
flags.DEFINE_integer('include_top', 0, 'if 1 top level is appended')

APP_MODELS = {
    'ResNet50':
        tf.keras.applications.resnet.ResNet50,
    'ResNet101':
        tf.keras.applications.resnet.ResNet101,
    'ResNet152':
        tf.keras.applications.resnet.ResNet152,
    'ResNet50V2':
        tf.keras.applications.resnet_v2.ResNet50V2,
    'ResNet101V2':
        tf.keras.applications.resnet_v2.ResNet101V2,
    'ResNet152V2':
        tf.keras.applications.resnet_v2.ResNet152V2,
    'VGG16':
        tf.keras.applications.vgg16.VGG16,
    'VGG19':
        tf.keras.applications.vgg19.VGG19,
    'Xception':
        tf.keras.applications.xception.Xception,
    'InceptionV3':
        tf.keras.applications.inception_v3.InceptionV3,
    'InceptionResNetV2':
        tf.keras.applications.inception_resnet_v2.InceptionResNetV2,
    'MobileNet':
        tf.keras.applications.mobilenet.MobileNet,
    'MobileNetV2':
        tf.keras.applications.mobilenet_v2.MobileNetV2,
    'DenseNet121':
        tf.keras.applications.densenet.DenseNet121,
    'DenseNet169':
        tf.keras.applications.densenet.DenseNet169,
    'DenseNet201':
        tf.keras.applications.densenet.DenseNet201,
    'NASNetMobile':
        tf.keras.applications.nasnet.NASNetMobile,
    'NASNetLarge':
        tf.keras.applications.nasnet.NASNetLarge,
}


def get_input_shape():
  if FLAGS.data == 'imagenet':
    if FLAGS.model in ['InceptionV3', 'Xception', 'InceptionResNetV2']:
      return (1, 299, 299, 3)
    elif FLAGS.model == 'NASNetLarge':
      return (1, 331, 331, 3)
    else:
      return (1, 224, 224, 3)
  elif FLAGS.data == 'cifar10':
    return (1, 32, 32, 3)
  else:
    raise ValueError(f'Data not supported: {FLAGS.data}')


def load_cifar10_weights(model):
  file_name = 'cifar10' + FLAGS.model
  # get_file will download the model weights from a publicly available folder,
  # save them to cache_dir=~/.keras/models/ and return a path to them.
  url = os.path.join(
      FLAGS.url, f'cifar10_include_top_{FLAGS.include_top}_{FLAGS.model}.h5')
  weights_path = tf.keras.utils.get_file(file_name, url)
  model.load_weights(weights_path)
  return model


def initialize_model():
  tf_utils.set_random_seed()
  tf.keras.backend.set_learning_phase(False)

  # Keras applications models receive input shapes without a batch dimension, as
  # the batch size is dynamic by default. This selects just the image size.
  input_shape = get_input_shape()[1:]

  # If weights == 'imagenet', the model will load the appropriate weights from
  # an external tf.keras URL.
  weights = 'imagenet' if FLAGS.data == 'imagenet' else None

  model = APP_MODELS[FLAGS.model](
      weights=weights, include_top=FLAGS.include_top, input_shape=input_shape)

  if FLAGS.data == 'cifar10' and FLAGS.url:
    model = load_cifar10_weights(model)
  return model


class VisionModule(tf.Module):

  def __init__(self):
    super(VisionModule, self).__init__()
    self.m = initialize_model()
    # Specify input shape with a static batch size.
    # TODO(b/142948097): Add support for dynamic shapes in SPIR-V lowering.
    # Replace input_shape with m.input_shape to make the batch size dynamic.
    self.predict = tf.function(
        input_signature=[tf.TensorSpec(get_input_shape())])(
            self.m.call)


@tf_test_utils.compile_module(VisionModule, exported_names=['predict'])
class AppTest(tf_test_utils.TracedModuleTestCase):

  def test_application(self):

    def predict(module):
      module.predict(tf_utils.uniform(get_input_shape()))

    self.compare_backends(predict)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

  if FLAGS.model not in APP_MODELS:
    raise ValueError(f'Unsupported model: {FLAGS.model}')
  # Override VisionModule's __name__ to be more specific.
  VisionModule.__name__ = os.path.join(FLAGS.model, FLAGS.data)

  tf.test.main()


if __name__ == '__main__':
  app.run(main)
