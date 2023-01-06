# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import collections
import os
from typing import Any, Dict, Sequence, Type, Union

from absl import app
from absl import flags
from iree.tf.support import tf_test_utils
from iree.tf.support import tf_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# As high as tf goes without breaking.
RANK_7_SHAPE = [2] * 7
UNARY_SIGNATURE_SHAPES = [[RANK_7_SHAPE]]
BINARY_SIGNATURE_SHAPES = [[RANK_7_SHAPE] * 2]
TERNARY_SIGNATURE_SHAPES = [[RANK_7_SHAPE] * 3]

# Reused UnitTestSpecs.
SEGMENT_UNIT_TEST_SPECS = tf_test_utils.unit_test_specs_from_args(
    names_to_input_args={
        "tf_doc_example": [
            tf.constant([
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [5, 6, 7, 8],
            ], np.float32),
            np.array([0, 0, 1], np.int32),
        ]
    })
UNSORTED_SEGMENT_UNIT_TEST_SPECS = tf_test_utils.unit_test_specs_from_args(
    names_to_input_args={
        "tf_doc_example": [
            tf.constant([
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [5, 6, 7, 8],
            ], np.float32),
            np.array([0, 0, 1], np.int32),
            2,
        ]
    })

REDUCE_KWARGS_TO_VALUES = {
    "axis": [None, 1],
    "keepdims": [False, True],
}

# A dictionary mapping tf.math function names to lists of UnitTestSpecs.
# Each unit_test_name will have the tf.math function name prepended to it.
FUNCTIONS_TO_UNIT_TEST_SPECS = {
    "abs":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "accumulate_n": [
        tf_test_utils.UnitTestSpec(
            unit_test_name='f32',
            input_signature=[[tf.TensorSpec(RANK_7_SHAPE, tf.float32)] * 5]),
        tf_test_utils.UnitTestSpec(
            unit_test_name='i32',
            input_signature=[[tf.TensorSpec(RANK_7_SHAPE, tf.int32)] * 5]),
    ],
    "acos":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "acosh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32],
            input_generators=[tf_utils.ndarange]),
    "add":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "add_n": [
        tf_test_utils.UnitTestSpec(
            unit_test_name='f32',
            input_signature=[[tf.TensorSpec(RANK_7_SHAPE, tf.float32)] * 5]),
        tf_test_utils.UnitTestSpec(
            unit_test_name='i32',
            input_signature=[[tf.TensorSpec(RANK_7_SHAPE, tf.int32)] * 5]),
    ],
    "angle":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "argmax":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "argmin":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "asin":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "asinh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "atan":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "atan2":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "atanh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "bessel_i0":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "bessel_i0e":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "bessel_i1":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "bessel_i1e":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "betainc":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=TERNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "bincount":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.int32],
            input_generators=[tf_utils.ndarange]),
    "ceil":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "confusion_matrix":
        tf_test_utils.unit_test_specs_from_args(names_to_input_args={
            "five_classes": [tf.constant([1, 2, 4]),
                             tf.constant([2, 2, 4])]
        }),
    "conj":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "cos":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "cosh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "count_nonzero":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64],
            input_generators=[tf_utils.ndarange]),
    "cumprod":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "cumsum":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "cumulative_logsumexp":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "digamma":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "divide":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "divide_no_nan":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "equal":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "erf":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "erfc":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "erfinv":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "exp":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "expm1":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "floor":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "floordiv":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            # Avoid integer division by 0.
            input_generators={
                "uniform_1_3":
                    lambda *args: tf_utils.uniform(*args, low=1.0, high=3.0)
            }),
    "floormod":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            # Avoid integer division by 0.
            input_generators={
                "uniform_1_3":
                    lambda *args: tf_utils.uniform(*args, low=1.0, high=3.0)
            }),
    "greater":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "greater_equal":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "igamma":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "igammac":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "imag":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "in_top_k": [
        tf_test_utils.UnitTestSpec(
            unit_test_name="k_3",
            input_signature=[
                tf.TensorSpec([8], tf.int32),
                tf.TensorSpec([8, 3])
            ],
            input_generator=tf_utils.ndarange,
            kwargs=dict(k=3),
        )
    ],
    "invert_permutation": [
        tf_test_utils.UnitTestSpec(
            unit_test_name="random",
            input_signature=[tf.TensorSpec([8], tf.int32)],
            input_generator=tf_utils.random_permutation,
        )
    ],
    "is_finite":
        tf_test_utils.unit_test_specs_from_args(names_to_input_args={
            "nan_and_inf": [tf.constant([[1., np.nan], [np.inf, 2.]])]
        }),
    "is_inf":
        tf_test_utils.unit_test_specs_from_args(names_to_input_args={
            "nan_and_inf": [tf.constant([[1., np.nan], [np.inf, 2.]])]
        }),
    "is_nan":
        tf_test_utils.unit_test_specs_from_args(names_to_input_args={
            "nan_and_inf": [tf.constant([[1., np.nan], [np.inf, 2.]])]
        }),
    "is_non_decreasing":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "is_strictly_increasing":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "l2_normalize":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "lbeta":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "less":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "less_equal":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "lgamma":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "log":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "log1p":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "log_sigmoid":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "log_softmax":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "logical_and":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool]),
    "logical_not":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool]),
    "logical_or":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool]),
    "logical_xor":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool]),
    "maximum":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "minimum":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "mod":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            input_generators={
                "positive_ndarange": lambda *args: tf_utils.ndarange(*args) + 1
            }),
    "multiply":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "multiply_no_nan":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "ndtri":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "negative":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "nextafter":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES),
    "not_equal":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32]),
    "polygamma":
        tf_test_utils.unit_test_specs_from_args(names_to_input_args={
            "nan_and_inf": [tf.ones(16), tf.linspace(0.5, 4, 16)]
        }),
    "polyval": [
        tf_test_utils.UnitTestSpec(
            unit_test_name="three_coeffs",
            input_signature=[[tf.TensorSpec(RANK_7_SHAPE)] * 3,
                             tf.TensorSpec([])],
        )
    ],
    "pow":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[[[1, 2, 2, 2], [1, 2, 2, 2]]],
            signature_dtypes=[tf.float32, tf.int32, tf.complex64],
            # Avoid numbers <1 or large ones that will overflow
            input_generators={
                "positive_moderate_ndarange":
                    lambda *args: tf_utils.uniform(*args, low=1.0, high=3.0)
            }),
    "real":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "reciprocal":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "reciprocal_no_nan":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "reduce_all": [
        # Explicitly test all True inputs to be absolutely sure that some
        # reduction axes return True.
        *tf_test_utils.unit_test_specs_from_args(
            names_to_input_args={
                "all_true": [np.ones(RANK_7_SHAPE, bool)],
            },
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
        *tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    ],
    "reduce_any": [
        # Explicitly test all False inputs to be absolutely sure that some
        # reduction axes return False.
        *tf_test_utils.unit_test_specs_from_args(
            names_to_input_args={
                "all_false": [np.zeros(RANK_7_SHAPE, bool)],
            },
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
        *tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.bool],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    ],
    "reduce_euclidean_norm":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_logsumexp":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_max":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_mean":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_min":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_prod":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_std":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_sum":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "reduce_variance":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64],
            kwargs_to_values=REDUCE_KWARGS_TO_VALUES),
    "rint":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "round":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "rsqrt":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "scalar_mul":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[[[], [8]]]),
    "segment_max":
        SEGMENT_UNIT_TEST_SPECS,
    "segment_mean":
        SEGMENT_UNIT_TEST_SPECS,
    "segment_min":
        SEGMENT_UNIT_TEST_SPECS,
    "segment_prod":
        SEGMENT_UNIT_TEST_SPECS,
    "segment_sum":
        SEGMENT_UNIT_TEST_SPECS,
    "sigmoid":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "sign":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "sin":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "sinh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "sobol_sample":
        tf_test_utils.unit_test_specs_from_args(
            names_to_input_args={"simple": [4, 3]}),
    "softmax":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "softplus":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "softsign":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32]),
    "sqrt":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "square":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "squared_difference":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "subtract":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "tan":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "tanh":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "top_k":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=[[[2, 2]]],
            signature_dtypes=[tf.float32],
            kwargs_to_values={"k": [1, 2]}),
    "truediv":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "unsorted_segment_max":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "unsorted_segment_mean":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "unsorted_segment_min":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "unsorted_segment_prod":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "unsorted_segment_sqrt_n":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "unsorted_segment_sum":
        UNSORTED_SEGMENT_UNIT_TEST_SPECS,
    "xdivy":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "xlog1py":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "xlogy":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.complex64]),
    "zero_fraction":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=UNARY_SIGNATURE_SHAPES,
            signature_dtypes=[tf.float32, tf.int32, tf.complex64]),
    "zeta":
        tf_test_utils.unit_test_specs_from_signatures(
            signature_shapes=BINARY_SIGNATURE_SHAPES,
            # The function is poorly behaved near zero, so we test this range
            # to avoid outputing all nans.
            input_generators={
                "uniform_3_4":
                    lambda *args: tf_utils.uniform(*args, low=3.0, high=4.0)
            },
        )
}

for function, specs in FUNCTIONS_TO_UNIT_TEST_SPECS.items():
  # Update using 'with_name' to avoid updating shared UnitTestSpecs.
  specs = [
      spec.with_name(f"{function}__{spec.unit_test_name}") for spec in specs
  ]
  FUNCTIONS_TO_UNIT_TEST_SPECS[function] = specs

  # Validate that there are not multiple UnitTestSpecs with the same name.
  seen_unit_test_names = set()
  for spec in specs:
    if spec.unit_test_name in seen_unit_test_names:
      raise ValueError(
          f"Found multiple UnitTestSpecs with the name '{spec.unit_test_name}'")
    seen_unit_test_names.add(spec.unit_test_name)

flags.DEFINE_list(
    "functions", None,
    f"Any of {list(FUNCTIONS_TO_UNIT_TEST_SPECS.keys())}. If more than one "
    "function is provided then len(--target_backends) must be one.")
flags.DEFINE_bool(
    "dynamic_dims", False,
    "Whether or not to compile the layer with dynamic dimensions.")
flags.DEFINE_bool(
    "test_complex", False,
    "Whether or not to test or ignore function signatures with complex types.")
flags.DEFINE_bool(
    'list_functions_with_complex_tests', False,
    'Whether or not to print out all functions with complex inputs '
    '(and skip running the tests).')


def _wrap_top_k(top_k):
  # top_k returns a tensorflow.python.ops.gen_nn_ops.TopKV2. Wrap it in a tuple
  # so we don't have to. (The lambda is wrapped to avoid a recursive capture).
  return lambda *args, **kwargs: tuple(top_k(*args, **kwargs))


def create_function_unit_test(
    function_name: str,
    unit_test_spec: tf_test_utils.UnitTestSpec) -> tf.function:
  """Creates a tf_function_unit_test from the provided UnitTestSpec."""
  function = getattr(tf.math, function_name)
  signature = unit_test_spec.input_signature

  if tf_utils.is_complex(signature):
    function, signature = tf_utils.rewrite_complex_signature(
        function, signature)
  if function_name == "top_k":
    function = _wrap_top_k(function)
  wrapped_function = lambda *args: function(*args, **unit_test_spec.kwargs)

  if FLAGS.dynamic_dims:
    signature = tf_utils.apply_function(signature, tf_utils.make_dims_dynamic)

  return tf_test_utils.tf_function_unit_test(
      input_signature=signature,
      input_generator=unit_test_spec.input_generator,
      input_args=unit_test_spec.input_args,
      name=unit_test_spec.unit_test_name,
      rtol=1e-5,
      atol=1e-5)(wrapped_function)


class TfMathModule(tf_test_utils.TestModule):

  def __init__(self):
    super().__init__()
    for function in FLAGS.functions:
      for unit_test_spec in FUNCTIONS_TO_UNIT_TEST_SPECS[function]:
        if not FLAGS.test_complex and tf_utils.is_complex(
            unit_test_spec.input_signature):
          continue
        function_unit_test = create_function_unit_test(function, unit_test_spec)
        setattr(self, unit_test_spec.unit_test_name, function_unit_test)


def get_relative_artifacts_dir() -> str:
  if len(FLAGS.functions) > 1:
    # We only allow testing multiple functions with a single target backend
    # so that we can store the artifacts under:
    #   'artifacts_dir/multiple_functions__backend/...'
    # We specialize the 'multiple_functions' dir by backend to avoid overwriting
    # tf_input.mlir and iree_input.mlir. These are typically identical across
    # backends, but are not when the functions to compile change per-backend.
    if len(FLAGS.target_backends) != 1:
      raise flags.IllegalFlagValueError(
          "Expected len(target_backends) == 1 when len(functions) > 1, but got "
          f"the following values for target_backends: {FLAGS.target_backends}.")
    function_str = f"multiple_functions__{FLAGS.target_backends[0]}"
  else:
    function_str = FLAGS.functions[0]
  dim_str = "dynamic_dims" if FLAGS.dynamic_dims else "static_dims"
  complex_str = "complex" if FLAGS.test_complex else "non_complex"
  return os.path.join("tf", "math", function_str, f"{dim_str}_{complex_str}")


class TfMathTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(
        TfMathModule,
        exported_names=TfMathModule.get_tf_function_unit_tests(),
        relative_artifacts_dir=get_relative_artifacts_dir())


def main(argv):
  del argv  # Unused.
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()

  if FLAGS.list_functions_with_complex_tests:
    for function_name, unit_test_specs in FUNCTIONS_TO_UNIT_TEST_SPECS.items():
      for spec in unit_test_specs:
        if tf_utils.is_complex(spec.input_signature):
          print(f'    "{function_name}",')
    return

  if FLAGS.functions is None:
    raise flags.IllegalFlagValueError(
        "'--functions' must be specified if "
        "'--list_functions_with_complex_tests' isn't")

  TfMathTest.generate_unit_tests(TfMathModule)
  tf.test.main()


if __name__ == "__main__":
  app.run(main)
