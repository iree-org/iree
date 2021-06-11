# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test training a regression model with Keras optimizers."""

import os

from absl import app
from absl import flags
from iree.tf.support import tf_test_utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "optimizer", "sgd",
    "One of 'adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', "
    "'rmsprop' or 'sgd'")

np.random.seed(0)
INPUT_DIM = 8
OUTPUT_DIM = 2
BATCH_SIZE = 4
WEIGHTS = np.random.uniform(-1, 1, size=(INPUT_DIM, OUTPUT_DIM))
BIASES = np.random.uniform(-1, 1, size=(OUTPUT_DIM,))


def get_linear_data():
  x = np.random.uniform(-1, 1, size=(BATCH_SIZE, INPUT_DIM))
  y = np.dot(x, WEIGHTS) + BIASES
  return x.astype(np.float32), y.astype(np.float32)


class RegressionTrainingModule(tf.Module):
  """A module for model training."""

  def __init__(self):
    inputs = tf.keras.layers.Input(INPUT_DIM)
    outputs = tf.keras.layers.Dense(OUTPUT_DIM)(inputs)
    self.model = tf.keras.Model(inputs, outputs)

    self.loss = tf.keras.losses.MeanSquaredError()
    self.optimizer = tf.keras.optimizers.get(FLAGS.optimizer)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE, INPUT_DIM], tf.float32),
      tf.TensorSpec([BATCH_SIZE, OUTPUT_DIM], tf.float32)
  ])
  def train_on_batch(self, x, y_true):
    with tf.GradientTape() as tape:
      y_pred = self.model(x, training=True)
      loss = self.loss(y_pred, y_pred)
    variables = self.model.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return loss

  @tf.function(input_signature=[])
  def get_weights(self):
    return self.model.weights[0]

  @tf.function(input_signature=[])
  def get_bias(self):
    return self.model.weights[1]


class RegressionTrainingTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        RegressionTrainingModule,
        exported_names=["train_on_batch", "get_weights", "get_bias"],
        relative_artifacts_dir=os.path.join(RegressionTrainingModule.__name__,
                                            FLAGS.optimizer))

  def test_train_on_batch(self):

    def train_on_batch(module):
      x, y = get_linear_data()
      module.train_on_batch(x, y)
      # Ensures the weights are identical.
      module.get_weights()
      module.get_bias()

    self.compare_backends(train_on_batch, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == "__main__":
  app.run(main)
