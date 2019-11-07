# Copyright 2019 Google LLC
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
"""Tests supported features of saved models."""

# pylint: disable=invalid-name
# pylint: disable=missing-docstring

import pyiree
import tensorflow.compat.v2 as tf

SAVED_MODEL_IMPORT_PASSES = [
    "tf-executor-graph-pruning",
    "tf-standard-pipeline",
    "iree-tf-saved-model-adopt-exports",
    "canonicalize",
]


# Tests that a simple example with flat args and a single result and no
# captures imports properly.
# CHECK-LABEL: RUN_TEST: T0001_FlatArgsResultsNoBoundGlobals
# CHECK: module
# CHECK-NOT: tf_saved_model.semantics
# CHECK: @simple_mul_no_capture
# CHECK-NEXT: iree.module.export
# CHECK: FINISH_TEST
class T0001_FlatArgsResultsNoBoundGlobals(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul_no_capture(self, a, b):
    return a * b


pyiree.tf_test_driver.add_test(
    test_name="T0001_FlatArgsResultsNoBoundGlobals",
    tf_module_builder=T0001_FlatArgsResultsNoBoundGlobals,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True)


# Tests that a bound global var imports properly.
# NOTE: This is currently an error and needs to be implemented
# CHECK-LABEL: RUN_TEST: T0002_FlatArgsResultsBoundGlobalVar
# CHECK: [ERROR]: This pass doesn't support global tensors yet
# CHECK: FINISH_TEST_WITH_EXCEPTION
class T0002_FlatArgsResultsBoundGlobalVar(tf.Module):

  def __init__(self):
    self.v = tf.Variable([1., 2., 3., 4.])

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b + self.v


pyiree.tf_test_driver.add_test(
    test_name="T0002_FlatArgsResultsBoundGlobalVar",
    tf_module_builder=T0002_FlatArgsResultsBoundGlobalVar,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True,
    expect_pass_failure=True)


# Tests that a structured argument is handled properly.
# NOTE: This is currently an error and needs to be implemented
# CHECK-LABEL: RUN_TEST: T0003_StructuredArgs
# CHECK: [ERROR]: This pass doesn't support structured arguments yet
# CHECK: FINISH_TEST_WITH_EXCEPTION
class T0003_StructuredArgs(tf.Module):

  @tf.function(input_signature=[{
      "x": tf.TensorSpec([4], tf.float32),
      "y": tf.TensorSpec([4], tf.float32)
  }])
  def simple_mul(self, d):
    return d["x"] * d["y"]


pyiree.tf_test_driver.add_test(
    test_name="T0003_StructuredArgs",
    tf_module_builder=T0003_StructuredArgs,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True,
    expect_pass_failure=True)


# Tests that a structured argument is handled properly.
# NOTE: This is currently an error and needs to be implemented
# CHECK-LABEL: RUN_TEST: T0003_StructuredMultipleResult
# CHECK: [ERROR]: This pass doesn't support multiple results yet
# CHECK: FINISH_TEST_WITH_EXCEPTION
class T0003_StructuredMultipleResult(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    product = a * b
    return {"x": product, "x_squared": product * product}


pyiree.tf_test_driver.add_test(
    test_name="T0003_StructuredMultipleResult",
    tf_module_builder=T0003_StructuredMultipleResult,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True,
    expect_pass_failure=True)


# Tests that a structured argument is handled properly.
# NOTE: This is currently an error and needs to be implemented
# CHECK-LABEL: RUN_TEST: T0004_StructuredSingleResult
# CHECK: [ERROR]: This pass doesn't support structured results yet
# CHECK: FINISH_TEST_WITH_EXCEPTION
class T0004_StructuredSingleResult(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    product = a * b
    return {"x": product}


pyiree.tf_test_driver.add_test(
    test_name="T0004_StructuredSingleResult",
    tf_module_builder=T0004_StructuredSingleResult,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True,
    expect_pass_failure=True)


# Tests that a structured argument is handled properly.
# NOTE: This is currently an error and needs to be implemented
# CHECK-LABEL: RUN_TEST: T0005_MultipleExportedFuncNames
# CHECK: [ERROR]: Multiple exported names not supported yet
# CHECK: FINISH_TEST_WITH_EXCEPTION
class T0005_MultipleExportedFuncNames(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    product = a * b
    return {"x": product}


# Force a function alias.
T0005_MultipleExportedFuncNames.another_copy = (
    T0005_MultipleExportedFuncNames.simple_mul)

pyiree.tf_test_driver.add_test(
    test_name="T0005_MultipleExportedFuncNames",
    tf_module_builder=T0005_MultipleExportedFuncNames,
    passes=SAVED_MODEL_IMPORT_PASSES,
    print_input_module=True,
    expect_pass_failure=True)

if __name__ == "__main__":
  pyiree.tf_test_driver.run_tests(__file__, with_filecheck=True)
