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

from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# Testing all applications models automatically can take time
# so we test it one by one, with argument --model=MobileNet
flags.DEFINE_string('model', 'ResNet50', 'model name')

# 32x32 is the minimum image size (for test speed)
# for imagenet case it has to be [1, 224, 224, 3]
INPUT_SHAPE = [1, 32, 32, 3]

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


def models():
  tf.keras.backend.set_learning_phase(False)
  # TODO(ataei): This should move somewhere in SavedModelTestCase, it should
  # guarantee test is deterministic.
  tf.random.set_seed(0)

  # keras model receives images size as input,
  # where batch size is not specified - by default it is dynamic
  if FLAGS.model in APP_MODELS:
    model = APP_MODELS[FLAGS.model](
        weights=None, include_top=False, input_shape=INPUT_SHAPE[1:])
  else:
    raise ValueError('unsupported model', FLAGS.model)

  module = tf.Module()
  module.m = model
  # specify input size with static batch size
  # TODO(b/142948097): with support of dynamic shape
  # replace INPUT_SHAPE by model.input_shape, so batch size will be dynamic (-1)
  module.predict = tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE)])(
      model.call)
  return module


@tf_test_utils.compile_modules(applications=(models, ['predict']))
class AppTest(tf_test_utils.SavedModelTestCase):

  def test_application(self):
    input_data = np.random.rand(np.prod(np.array(INPUT_SHAPE))).astype(
        np.float32)
    input_data = input_data.reshape(INPUT_SHAPE)
    self.modules.applications.all.predict(input_data).print().assert_all_close(
        atol=1e-6)


if __name__ == '__main__':
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()
