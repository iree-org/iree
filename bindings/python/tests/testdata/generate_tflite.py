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

import os

import tensorflow as tf


class Squared(tf.Module):

  @tf.function
  def __call__(self, x):
    return tf.square(x)


model = Squared()
concrete_func = model.__call__.get_concrete_function(
    tf.TensorSpec(shape=[4, 3], dtype=tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, "tflite_sample.fb"), "wb") as f:
  f.write(tflite_model)
