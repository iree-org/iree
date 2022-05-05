# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates sample models for excercising various function signatures.

Usage:
  generate_dynamic_samples.py /tmp/dynsamples

This can then be fed into iree-import-tf to process it:

Fully convert to IREE input (run all import passes):
  iree-import-tf /tmp/dynsamples/broadcast_add.sm

Import only (useful for crafting test cases for the import pipeline):
  iree-import-tf /tmp/dynsamples/broadcast_add.sm

Can be further lightly pre-processed via:
  | iree-tf-opt --tf-standard-pipeline
"""

import os
import sys

import tensorflow as tf


class BroadcastAdd(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([None, 16], tf.float32),
      tf.TensorSpec([1, 16], tf.float32)
  ])
  def half_broadcast(self, a, b):
    return a + b

  @tf.function(input_signature=[
      tf.TensorSpec([None, 16], tf.float32),
      tf.TensorSpec([None, 16], tf.float32)
  ])
  def dyn_non_broadcast(self, a, b):
    return a + b


class StaticAdd(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([10, 16], tf.float32),
      tf.TensorSpec([1, 16], tf.float32)
  ])
  def broadcast(self, a, b):
    return a + b


class StaticNonBroadcastAdd(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([10, 16], tf.float32),
      tf.TensorSpec([10, 16], tf.float32)
  ])
  def non_broadcast(self, a, b):
    return a + b


class DynamicTanh(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([None, 16], tf.float32)])
  def f(self, a):
    return tf.tanh(tf.tanh(a))


MODELS = (
    (BroadcastAdd, "broadcast_add.sm"),
    (DynamicTanh, "dynamic_tanh.sm"),
    (StaticAdd, "static_add.sm"),
    (StaticNonBroadcastAdd, "static_non_broadcast_add.sm"),
)

try:
  output_dir = sys.argv[1]
except IndexError:
  print("Expected output directory")
  sys.exit(1)

for module_cls, dir_name in MODELS:
  m = module_cls()
  sm_dir = os.path.join(output_dir, dir_name)
  tf.saved_model.save(m, sm_dir)
  print(f"Saved {module_cls.__name__} to {sm_dir}")
