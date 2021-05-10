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
"""Generate artifacts for benchmarking models in tf.keras.applications."""

import os

from absl import app
from absl import flags
from iree.tf.support import tf_benchmark_utils
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# Testing all applications models automatically can take time
# so we test it one by one, with argument --model=MobileNet
flags.DEFINE_string("model", "ResNet50", "model name")
flags.DEFINE_string(
    "url", "", "url with model weights "
    "for example https://storage.googleapis.com/iree_models/")
flags.DEFINE_bool("use_external_weights", False,
                  "Whether or not to load external weights from the web")
flags.DEFINE_enum("data", "cifar10", ["cifar10", "imagenet"],
                  "data sets on which model was trained: imagenet, cifar10")
flags.DEFINE_bool(
    "include_top", True,
    "Whether or not to include the final (top) layers of the model.")

BATCH_SIZE = 1
IMAGE_DIM = 224


def get_static_shape(shape):
  input_shape = list([BATCH_SIZE] + shape[1:])

  # Some models accept dynamic image dimensions by default, so we use
  # IMAGE_DIM as a stand-in.
  for i, dim in enumerate(input_shape):
    if dim is None:
      input_shape[i] = IMAGE_DIM

  return input_shape


def get_model_input_shape_dtype():
  model_class = getattr(tf.keras.applications, FLAGS.model)
  model = model_class(include_top=FLAGS.include_top)
  return (get_static_shape(model.inputs[0].shape), model.inputs[0].dtype)


def load_cifar10_weights(model):
  file_name = "cifar10" + FLAGS.model
  # get_file will download the model weights from a publicly available folder,
  # save them to cache_dir=~/.keras/models/ and return a path to them.
  url = os.path.join(
      FLAGS.url, f"cifar10_include_top_{FLAGS.include_top:d}_{FLAGS.model}.h5")
  weights_path = tf.keras.utils.get_file(file_name, url)
  model.load_weights(weights_path)
  return model


def initialize_model():
  # If weights == "imagenet", the model will load the appropriate weights from
  # an external tf.keras URL.
  weights = None
  if FLAGS.use_external_weights and FLAGS.data == "imagenet":
    weights = "imagenet"

  model_class = getattr(tf.keras.applications, FLAGS.model)
  model = model_class(weights=weights, include_top=FLAGS.include_top)

  if FLAGS.use_external_weights and FLAGS.data == "cifar10":
    if not FLAGS.url:
      raise ValueError(
          "cifar10 weights cannot be loaded without the `--url` flag.")
    model = load_cifar10_weights(model)
  return model


class ApplicationsModule(tf.Module):

  def __init__(self):
    super().__init__()
    self.m = initialize_model()

    # Create a function with static shape name "call" to invoke the Keras application.
    input_shape = get_static_shape(self.m.inputs[0].shape)
    self.call = tf.function(input_signature=[tf.TensorSpec(input_shape)])(
        lambda x: self.m(x, training=False))
    self.call.__name__ = "call"


def main(argv):
  del argv  # Unused.
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()

  if not hasattr(tf.keras.applications, FLAGS.model):
    raise ValueError(f"Unsupported model: {FLAGS.model}")

  tf_benchmark_utils.compile_tf_module(
      ApplicationsModule,
      exported_name="call",
      input_shapes_dtypes=[get_model_input_shape_dtype()],
      relative_artifacts_dir=os.path.join(FLAGS.model, FLAGS.data))


if __name__ == "__main__":
  app.run(main)
