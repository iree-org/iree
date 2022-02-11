# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
import numpy
from . import test_util
import urllib.request

from PIL import Image

model_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite"


class PersonDetectTest(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(PersonDetectTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(PersonDetectTest, self).compare_results(iree_results, tflite_results,
                                                  details)
    self.assertTrue(
        numpy.isclose(iree_results[0], tflite_results[0], atol=1e-3).all())

  # TFLite is broken with this model so we hardcode the input/output details.
  def setup_tflite(self):
    self.input_details = [{
        "shape": [1, 96, 96, 1],
        "dtype": numpy.int8,
        "index": 0,
    }]
    self.output_details = [{
        "shape": [1, 2],
        "dtype": numpy.int8,
    }]

  # The input has known expected values. We hardcode this value.
  def invoke_tflite(self, args):
    return [numpy.array([[-113, 113]], dtype=numpy.int8)]

  def generate_inputs(self, input_details):
    img_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
    local_path = "/".join([self.workdir, "person.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize(
        (shape[1], shape[2]))).astype(input_details[0]["dtype"])
    args = [im.reshape(shape)]
    return args

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
