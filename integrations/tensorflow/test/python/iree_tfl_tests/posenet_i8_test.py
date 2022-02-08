# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
from . import test_util
import numpy
import urllib.request

from PIL import Image

model_path = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
model_input = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8_input.jpg"


class PosenetI8Test(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(PosenetI8Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(PosenetI8Test, self).compare_results(iree_results, tflite_results,
                                               details)
    # This value is a discretized location of the persons joints. If we are
    # *close* to the expected position we can consider this good enough.
    self.assertTrue(
        numpy.isclose(iree_results[0][:, :, :, 0],
                      tflite_results[0][:, :, :, 0],
                      atol=25e-3).all())
    self.assertTrue(
        numpy.isclose(iree_results[0][:, :, :, 1],
                      tflite_results[0][:, :, :, 1],
                      atol=25e-3).all())

  def generate_inputs(self, input_details):
    local_path = "/".join([self.workdir, "person.jpg"])
    urllib.request.urlretrieve(model_input, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
    return args

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
