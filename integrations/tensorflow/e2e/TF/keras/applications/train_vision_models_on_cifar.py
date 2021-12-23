# Lint as: python3
# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Train vision models on CIFAR10."""

import os
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'MobileNet', 'keras vision model name')
flags.DEFINE_string('model_path', '',
                    'Path to a location where model will be saved.')
flags.DEFINE_bool(
    'include_top', True,
    'Whether or not to include the final (top) layers of the model.')

# minimum size for keras vision models
INPUT_SHAPE = [1, 32, 32, 3]


def main(argv):
  del argv  # Unused.

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
  model_class = getattr(tf.keras.applications, FLAGS.model)
  model = model_class(weights=None,
                      include_top=FLAGS.include_top,
                      input_shape=INPUT_SHAPE[1:])
  model.summary()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # train model
  model.fit(train_images,
            train_labels,
            epochs=1,
            validation_data=(test_images, test_labels))

  file_name = os.path.join(
      FLAGS.model_path,
      f'cifar10_include_top_{FLAGS.include_top:d}_{FLAGS.model}.h5')
  try:
    model.save_weights(file_name)
  except IOError as e:
    raise IOError(f'Failed to save model at: {file_name}, error: {e}')

  # test model
  _, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  logging.info('Test accuracy: %f', test_acc)


if __name__ == '__main__':
  tf.app.run(main=main)
