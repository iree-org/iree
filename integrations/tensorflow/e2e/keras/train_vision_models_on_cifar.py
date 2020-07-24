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
"""Train vision models on CIFAR10."""

import os
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'MobileNet', 'keras vision model name')
flags.DEFINE_string('model_path', '',
                    'Path to a location where model will be saved.')
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

# minimum size for keras vision models
INPUT_SHAPE = [1, 32, 32, 3]


def main(_):

  # prepare training and testing data
  (train_images,
   train_labels), (test_images,
                   test_labels) = tf.keras.datasets.cifar10.load_data()
  train_labels = np.array([x[0] for x in train_labels])
  test_labels = np.array([x[0] for x in test_labels])

  # Normalize image values to be between 0 and 1
  train_images, test_images = train_images / 255.0, test_images / 255.0

  # reduce training samples for quick training
  # we do not need to use all data for getting non zero output scores
  train_images = train_images[:4000]
  train_labels = train_labels[:4000]

  # It is a toy model for debugging (not optimized for accuracy or speed).
  model = APP_MODELS[FLAGS.model_name](
      weights=None, include_top=FLAGS.include_top, input_shape=INPUT_SHAPE[1:])
  model.summary()
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  # train model
  model.fit(
      train_images,
      train_labels,
      epochs=1,
      validation_data=(test_images, test_labels))

  file_name = os.path.join(
      FLAGS.model_path,
      'cifar10_include_top_{}_{}'.format(FLAGS.include_top,
                                         FLAGS.model_name + '.h5'))
  try:
    model.save_weights(file_name)
  except IOError as e:
    raise IOError('Failed to save model at: %s, error: %s' % (file_name, e))

  # test model
  _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  logging.info('Test accuracy: %f', test_acc)


if __name__ == '__main__':
  tf.app.run(main=main)
