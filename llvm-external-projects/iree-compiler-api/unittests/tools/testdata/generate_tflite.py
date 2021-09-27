# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import tensorflow as tf


class Squared(tf.Module):

  @tf.function
  def __call__(self, x):
    return tf.square(x)


model = Squared()
concrete_func = model.__call__.get_concrete_function(
    tf.TensorSpec(shape=[4, 3], dtype=tf.float32))

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, "tflite_sample.fb"), "wb") as f:
  f.write(tflite_model)
