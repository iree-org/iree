# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates sample models for excercising various function signatures.

Usage:
  generate_signature_samples.py /tmp/sigs.sm

This can then be fed into iree-import-tf to process it:

Fully convert to IREE input (run all import passes):
  iree-import-tf /tmp/sigs.sm

Import only (useful for crafting test cases for the import pipeline):
  iree-import-tf -o /dev/null --save-temp-tf-input=- /tmp/sigs.sm

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
