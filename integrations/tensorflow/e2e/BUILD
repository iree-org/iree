# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ***THIS FILE DOES NOT BUILD WITH BAZEL***
#
# It is open sourced to enable Bazel->CMake conversion to maintain test coverage
# of our integration tests in open source while we figure out a long term plan
# for our integration testing.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load(
    "@iree//integrations/tensorflow/e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# Create binaries for all test srcs to allow them to be run manually.
[
    py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = [
            "//third_party/py/absl:app",
            "//third_party/py/absl/flags",
            "//third_party/py/iree:pylib_tf_support",
            "//third_party/py/numpy",
            "//third_party/py/tensorflow",
            "//util/debuginfo:signalsafe_addr2line_installer",
        ],
    )
    for src in glob(["*_test.py"])
]

ALL_SRCS = [
    "batch_norm_test.py",
    "batch_to_space_nd_test.py",
    "broadcast_to_test.py",
    "broadcasting_test.py",
    "concat_test.py",
    "control_flow_test.py",
    "conv_test.py",
    "conv_transpose_test.py",
    "depth_conv_test.py",
    "dynamic_mlp_relu_test.py",
    "dynamic_mlp_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",
    "fill_test.py",
    "gather_test.py",
    "image_resize_test.py",
    "linspace_test.py",
    "mandelbrot_test.py",
    "matrix_ops_dynamic_test.py",
    "matrix_ops_static_test.py",
    "pytree_test.py",
    "quantization_dyn_test.py",
    "quantization_test.py",
    "range_test.py",
    "resource_ops_test.py",
    "ring_buffer_test.py",
    "scatter_update_test.py",
    "simple_arithmetic_test.py",
    "simple_stateful_test.py",
    "sliding_window_test.py",
    "space_to_batch_nd_test.py",
]

# keep sorted
TFLITE_FAILING = [
    "concat_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",
    "image_resize_test.py",
    "mandelbrot_test.py",
    "pytree_test.py",
    "resource_ops_test.py",
    "ring_buffer_test.py",
    "scatter_update_test.py",
    "simple_stateful_test.py",
    "sliding_window_test.py",
]

# keep sorted
LLVM_FAILING = [
    "broadcast_to_test.py",
    "broadcasting_test.py",
    "conv_transpose_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",  # TODO(natashaknk): Get this working after kernel is in.
    "fill_test.py",  # TODO(jennik): Get this test working on IREE.
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py",
    "quantization_dyn_test.py",
    "range_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
]

# keep sorted
VULKAN_FAILING = [
    "broadcast_to_test.py",
    "broadcasting_test.py",
    "conv_transpose_test.py",
    "dynamic_mlp_relu_test.py",
    "dynamic_mlp_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",  # TODO(natashaknk): Get this working after kernel is in.
    "fill_test.py",  # TODO(jennik): Get this test working on IREE.
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py",
    "quantization_dyn_test.py",
    "range_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
]

iree_e2e_cartesian_product_test_suite(
    name = "e2e_tests",
    failing_configurations = [
        {
            "target_backends": "tflite",
            "src": TFLITE_FAILING,
        },
        {
            "target_backends": "iree_llvmaot",
            "src": LLVM_FAILING,
        },
        {
            "target_backends": "iree_vulkan",
            "src": VULKAN_FAILING,
        },
    ],
    matrix = {
        "src": enforce_glob(
            ALL_SRCS,
            include = ["*_test.py"],
            exclude = ["mobile_bert_squad_test.py"],
        ),
        "target_backends": [
            "tf",
            "tflite",
            "iree_llvmaot",
            "iree_vulkan",
        ],
        "reference_backend": "tf",
    },
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)

iree_e2e_cartesian_product_test_suite(
    name = "mobile_bert_squad_tests",
    size = "enormous",
    matrix = {
        "src": "mobile_bert_squad_test.py",
        "reference_backend": "tf",
        "target_backends": [
            "tf",
            "tflite",
            "iree_llvmaot",
            "iree_vulkan",
        ],
    },
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)
