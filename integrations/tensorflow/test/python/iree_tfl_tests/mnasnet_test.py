# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
import numpy
from . import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_224/1/metadata/1?lite-format=tflite"


class MnasnetTest(test_util.TFLiteModelTest):

  def __init__(self, *args, **kwargs):
    super(MnasnetTest, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute()


if __name__ == '__main__':
  absl.testing.absltest.main()
