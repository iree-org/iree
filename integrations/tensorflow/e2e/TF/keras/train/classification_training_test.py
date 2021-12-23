# Lint as: python3
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test training a classification model with Keras optimizers."""

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

SAMPLES_PER_SPIRAL = 8
INPUT_DIM = 2
NUM_CLASSES = 4
BATCH_SIZE = NUM_CLASSES * SAMPLES_PER_SPIRAL


def get_spiral_dataset(samples_per_spiral: int,
                       noise_scale: float = 0,
                       shuffle: bool = True):
  """Creates a dataset with four spiral arms."""
  t = np.linspace(0, 1, samples_per_spiral, dtype=np.float32)
  cos_term = t * np.sin(2 * np.pi * t)
  sin_term = t * np.cos(2 * np.pi * t)
  spirals = [
      np.stack([sin_term, cos_term], axis=-1),
      np.stack([-sin_term, -cos_term], axis=-1),
      np.stack([-cos_term, sin_term], axis=-1),
      np.stack([cos_term, -sin_term], axis=-1)
  ]
  inputs = np.concatenate(spirals)
  inputs = inputs + np.random.normal(scale=noise_scale, size=inputs.shape)
  labels = np.concatenate([i * np.ones_like(t) for i in range(4)])

  if shuffle:
    # Shuffle by batch dim.
    index = np.arange(inputs.shape[0])
    np.random.shuffle(index)
    inputs = inputs[index]
    labels = labels[index]

  return inputs.astype(np.float32), labels.astype(np.float32)


class ClassificationTrainingModule(tf.Module):
  """A module for model training."""

  def __init__(self):
    inputs = tf.keras.layers.Input(INPUT_DIM)
    x = tf.keras.layers.Dense(NUM_CLASSES)(inputs)
    outputs = tf.keras.layers.Softmax()(x)
    self.model = tf.keras.Model(inputs, outputs)

    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.get(FLAGS.optimizer)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE, INPUT_DIM], tf.float32),
      tf.TensorSpec([BATCH_SIZE], tf.float32)
  ])
  def train_on_batch(self, inputs, labels):
    with tf.GradientTape() as tape:
      probs = self.model(inputs, training=True)
      loss = self.loss(labels, probs)
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


class ClassificationTrainingTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        ClassificationTrainingModule,
        exported_names=["train_on_batch", "get_weights", "get_bias"],
        relative_artifacts_dir=os.path.join(
            ClassificationTrainingModule.__name__, FLAGS.optimizer))

  def test_train_on_batch(self):

    def train_on_batch(module):
      inputs, labels = get_spiral_dataset(SAMPLES_PER_SPIRAL, noise_scale=0.05)
      module.train_on_batch(inputs, labels)
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
