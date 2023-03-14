# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for ops in the tf.math module."""

from absl import app
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf


class QuantizationDynModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
  def fake_quant(self, x):
    return tf.quantization.fake_quant_with_min_max_args(x,
                                                        min=-6,
                                                        max=6,
                                                        num_bits=8,
                                                        narrow_range=False,
                                                        name=None)


class QuantizationDynTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(QuantizationDynModule)

  def test_fake_quant(self):

    def abs(module):
      module.fake_quant(tf_utils.uniform([32], low=-6, high=6))

    self.compare_backends(abs, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
