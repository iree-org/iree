# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
import numpy
from . import test_util

model_path = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-64.tflite"


# This test is a massive download and excluded due to causing timeouts.
class GPT2Test(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(GPT2Test, self).__init__(model_path, *args, **kwargs)

  # Inputs modified to be useful mobilebert inputs.
  def generate_inputs(self, input_details):
    args = []
    args.append(
        numpy.random.randint(low=0,
                             high=256,
                             size=input_details[0]["shape"],
                             dtype=input_details[0]["dtype"]))
    return args

  def compare_results(self, iree_results, tflite_results, details):
    super(GPT2Test, self).compare_results(iree_results, tflite_results, details)
    for i in range(len(iree_results)):
      self.assertTrue(
          numpy.isclose(iree_results[i], tflite_results[i], atol=5e-3).all())

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
