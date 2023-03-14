# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates sample models for excercising various function signatures.

Usage:
  generate_errors_module.py /tmp/errors.sm

This can then be fed into iree-import-tf to process it:

Fully convert to IREE input (run all import passes):
  iree-import-tf /tmp/errors.sm

Import only (useful for crafting test cases for the import pipeline):
  iree-import-tf -o /dev/null --save-temp-tf-input=- /tmp/errors.sm

Can be further lightly pre-processed via:
  | iree-tf-opt --tf-standard-pipeline
"""

import sys

import tensorflow as tf


class ErrorsModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([16], tf.float32)])
  def string_op(self, a):
    tf.print(a)
    return a


try:
  file_name = sys.argv[1]
except IndexError:
  print("Expected output file name")
  sys.exit(1)

m = ErrorsModule()
tf.saved_model.save(m,
                    file_name,
                    options=tf.saved_model.SaveOptions(save_debug_info=True))
print(f"Saved to {file_name}")
