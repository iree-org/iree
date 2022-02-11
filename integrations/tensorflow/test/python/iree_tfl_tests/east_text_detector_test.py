# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
import numpy
from . import test_util

model_path = "https://tfhub.dev/sayakpaul/lite-model/east-text-detector/dr/1?lite-format=tflite"


class EastTextDetectorTest(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(EastTextDetectorTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(EastTextDetectorTest, self).compare_results(iree_results,
                                                      tflite_results, details)
    self.assertTrue(
        numpy.isclose(iree_results[0], tflite_results[0], atol=1e-3).all())

    # The second return is extremely noisy as it is not a binary classification. To handle we
    # check normalized correlation with an expectation of "close enough".
    iree_norm = numpy.sqrt(iree_results[1] * iree_results[1])
    tflite_norm = numpy.sqrt(tflite_results[1] * tflite_results[1])

    correlation = numpy.average(iree_results[1] * tflite_results[1] /
                                iree_norm / tflite_norm)
    self.assertTrue(numpy.isclose(correlation, 1.0, atol=1e-2).all())

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
