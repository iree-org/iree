# Lint as: python3
# Copyright 2021 Google LLC
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
"""Generates sample models for excercising various function signatures.

Usage:
  generate_signature_samples.py /tmp/sigs.sm

This can then be fed into iree-tf-import to process it:

Fully convert to IREE input (run all import passes):
  iree-tf-import /tmp/sigs.sm

Import only (useful for crafting test cases for the import pipeline):
  iree-tf-import -o /dev/null -save-temp-tf-input=- /tmp/sigs.sm

Can be further lightly pre-processed via:
  | iree-tf-opt --tf-standard-pipeline
"""

import sys

import tensorflow as tf


class SignaturesModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([16], tf.float32)])
  def unary_func(self, a):
    return a

  @tf.function(input_signature=2 * [tf.TensorSpec([16], tf.float32)])
  def binary_func(self, a, b):
    return a, b

  @tf.function(input_signature=[{
      "dict": {
          "b": tf.TensorSpec([16], tf.float32),
          "a": tf.TensorSpec([16], tf.float32),
      },
      "list": 2 * [tf.TensorSpec([16], tf.float32)],
  },
                                tf.TensorSpec([], tf.float32)])
  def dict_nest(self, mapping, scalar):
    return mapping

  @tf.function(input_signature=2 * [tf.TensorSpec([16], tf.float32)])
  def return_list(self, a, b):
    return [a, b]


try:
  file_name = sys.argv[1]
except IndexError:
  print("Expected output file name")
  sys.exit(1)

m = SignaturesModule()
tf.saved_model.save(m, file_name)
print(f"Saved to {file_name}")
