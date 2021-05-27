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
  generate_errors_module.py /tmp/errors.sm

This can then be fed into iree-import-tf to process it:

Fully convert to IREE input (run all import passes):
  iree-import-tf /tmp/errors.sm

Import only (useful for crafting test cases for the import pipeline):
  iree-import-tf -o /dev/null -save-temp-tf-input=- /tmp/errors.sm

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
