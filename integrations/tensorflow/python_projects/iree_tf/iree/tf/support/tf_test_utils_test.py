# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for iree.tf.support.tf_test_utils."""

from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow as tf


class TfFunctionUnitTestModule(tf_test_utils.TestModule):
    @tf_test_utils.tf_function_unit_test(input_signature=[])
    def no_args(self):
        return np.array([True], dtype=bool)

    @tf_test_utils.tf_function_unit_test(
        input_signature=[
            tf.TensorSpec([4]),
            tf.TensorSpec([4]),
        ]
    )
    def default_uniform_inputs(self, a, b):
        return a + b

    @tf_test_utils.tf_function_unit_test(
        input_signature=[
            tf.TensorSpec([4]),
            tf.TensorSpec([4]),
        ],
        input_generator=tf_utils.ndarange,
    )
    def custom_input_generator(self, a, b):
        return a + b

    @tf_test_utils.tf_function_unit_test(
        input_signature=[
            tf.TensorSpec([4]),
            tf.TensorSpec([4]),
        ],
        input_args=[
            np.array([0, 1, 2, 3], np.float32),
            -np.array([0, 1, 2, 3], np.float32),
        ],
    )
    def custom_input_args(self, a, b):
        return a + b

    # This test will fail if atol is not successfully set.
    @tf_test_utils.tf_function_unit_test(
        input_signature=[
            tf.TensorSpec([128, 3072], tf.float32),
            tf.TensorSpec([3072, 256], tf.float32),
        ],
        atol=1e-2,
    )
    def high_tolerance(self, a, b):
        return tf.matmul(a, b)


class TestUtilsTests(tf.test.TestCase):
    def test_tf_function_unittet(self):
        class TfFunctionUnittestTest(tf_test_utils.TracedModuleTestCase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._modules = tf_test_utils.compile_tf_module(
                    TfFunctionUnitTestModule
                )

        TfFunctionUnittestTest.generate_unit_tests(TfFunctionUnitTestModule)
        test_case = TfFunctionUnittestTest()
        self.assertTrue(hasattr(test_case, "test_no_args"))
        self.assertTrue(hasattr(test_case, "test_default_uniform_inputs"))
        self.assertTrue(hasattr(test_case, "test_custom_input_generator"))
        self.assertTrue(hasattr(test_case, "test_custom_input_args"))
        self.assertTrue(hasattr(test_case, "test_high_tolerance"))

        # Will throw an error if 'atol' is not set.
        test_case = TfFunctionUnittestTest()
        test_case.test_high_tolerance()


if __name__ == "__main__":
    tf.test.main()
