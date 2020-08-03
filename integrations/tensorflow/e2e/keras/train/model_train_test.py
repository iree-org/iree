# Lint as: python3
# Copyright 2019 Google LLC
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
"""Test keras Model training."""

from absl import flags
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "optimizer_name", "sgd",
    "optimizer name: sgd, rmsprop, nadam, adamax, adam, adagrad, adadelta")

_DEGREE = 3  # polynomial degree of input feature for regression test
_FEATURE_SIZE = _DEGREE + 1  # input feature size
_BATCH_SIZE = 8  # batch size has to be dynamic TODO(b/142948097)
_INPUT_DATA_SHAPE = [_BATCH_SIZE, _FEATURE_SIZE]
_OUTPUT_DATA_SHAPE = [_BATCH_SIZE, 1]


class ModelTrain(tf.Module):
  """A module for model training."""

  @staticmethod
  def CreateModule(input_dim=_FEATURE_SIZE, output_dim=1):
    """Creates a module for regression model training.

    Args:
      input_dim: input dimensionality
      output_dim: output dimensionality

    Returns:
      model for linear regression
    """

    tf_utils.set_random_seed()

    # build a single layer model
    inputs = tf.keras.layers.Input((input_dim))
    outputs = tf.keras.layers.Dense(output_dim)(inputs)
    model = tf.keras.Model(inputs, outputs)
    return ModelTrain(model)

  def __init__(self, model):
    self.model = model
    self.loss = tf.keras.losses.MeanSquaredError()
    self.optimizer = tf.keras.optimizers.get(FLAGS.optimizer_name)

  @tf.function(input_signature=[
      tf.TensorSpec(_INPUT_DATA_SHAPE, tf.float32),
      tf.TensorSpec(_OUTPUT_DATA_SHAPE, tf.float32)
  ])
  def train_step(self, inputs, targets):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss_value = self.loss(predictions, targets)
    gradients = tape.gradient(loss_value, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))
    return loss_value


@tf_test_utils.compile_module(
    ModelTrain.CreateModule, exported_names=["train_step"])
class ModelTrainTest(tf_test_utils.TracedModuleTestCase):

  def generate_regression_data(self, size=8):
    x = np.arange(size) - size // 2
    y = 1.0 * x**3 + 1.0 * x**2 + 1.0 * x + np.random.randn(size) * size
    return x, y

  def test_model_train(self):

    # Generate input and output data for regression problem.
    inputs, targets = self.generate_regression_data()

    # Normalize data.
    inputs = inputs / max(inputs)
    targets = targets / max(targets)

    # Generate polynomial features.
    inputs = np.expand_dims(inputs, axis=1)
    polynomial = PolynomialFeatures(_DEGREE)  # returns: [1, a, b, a^2, ab, b^2]
    inputs = polynomial.fit_transform(inputs)

    targets = np.expand_dims(targets, axis=1)

    def train_step(module):
      # Run one iteration of training step.
      module.train_step(inputs, targets)

    self.compare_backends(train_step)


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()

  tf.test.main()
