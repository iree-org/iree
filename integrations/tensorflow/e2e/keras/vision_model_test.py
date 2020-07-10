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


def get_input_shape(data, model):
  if data == 'imagenet':
    if (model == 'InceptionV3' or model == 'Xception' or
        model == 'InceptionResNetV2'):
      return (1, 299, 299, 3)
    elif model == 'NASNetLarge':
      return (1, 331, 331, 3)
    else:
      return (1, 224, 224, 3)
  elif data == 'cifar10':
    return (1, 32, 32, 3)
  else:
    raise ValueError('Not supported data ', data)


def models():
  tf.keras.backend.set_learning_phase(False)
  tf_utils.set_random_seed()

  input_shape = get_input_shape(FLAGS.data, FLAGS.model)
  # keras model receives images size as input,
  # where batch size is not specified - by default it is dynamic
  if FLAGS.model in APP_MODELS:
    weights = 'imagenet' if FLAGS.data == 'imagenet' else None

    # if weights == 'imagenet' it will load weights from external tf.keras URL
    model = APP_MODELS[FLAGS.model](
        weights=weights,
        include_top=FLAGS.include_top,
        input_shape=input_shape[1:])

    if FLAGS.data == 'cifar10' and FLAGS.url:
      file_name = 'cifar10' + FLAGS.model
      # it will download model weights from publically available folder: PATH
      # and save it to cache_dir=~/.keras and return path to it
      weights_path = tf.keras.utils.get_file(
          file_name,
          os.path.join(
              FLAGS.url,
              'cifar10_include_top_{}_{}'.format(FLAGS.include_top,
                                                 FLAGS.model + '.h5')))

      model.load_weights(weights_path)
  else:
    raise ValueError('Unsupported model', FLAGS.model)

  module = tf.Module()
  module.m = model
  # specify input size with static batch size
  # TODO(b/142948097): with support of dynamic shape
  # replace input_shape by model.input_shape, so batch size will be dynamic (-1)
  module.predict = tf.function(input_signature=[tf.TensorSpec(input_shape)])(
      model.call)
  return module


@tf_test_utils.compile_module(models, exported_names=['predict'])
class AppTest(tf_test_utils.SavedModelTestCase):

  def test_application(self):
    input_shape = get_input_shape(FLAGS.data, FLAGS.model)
    input_data = np.random.rand(np.prod(np.array(input_shape))).astype(
        np.float32)
    input_data = input_data.reshape(input_shape)
    self.get_module().predict(input_data).print().assert_all_close(atol=1e-6)


if __name__ == '__main__':
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()
