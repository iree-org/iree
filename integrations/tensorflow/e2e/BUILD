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

# Test coverage across backends for e2e tests is defined directly in the BUILD
# files. Coverage tables generated from this file can be viewed here:
#   https://google.github.io/iree/tensorflow-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//e2e:iree_e2e_cartesian_product_test_suite.bzl",
    "iree_e2e_cartesian_product_test_suite",
)
load("@iree//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

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
    "strings_test.py",
    "tensorlist_test.py",
]

# keep sorted
TFLITE_FAILING = [
    "concat_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",
    "gather_test.py",
    "image_resize_test.py",
    "mandelbrot_test.py",
    "resource_ops_test.py",
    "ring_buffer_test.py",
    "scatter_update_test.py",
    "simple_stateful_test.py",
    "sliding_window_test.py",
    "strings_test.py",
]

# keep sorted
VMLA_FAILING = [
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",  # TODO(natashaknk): Get diffs within tolerance
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "ring_buffer_test.py",  # TODO(b/148747011)
    "strings_test.py",
    "tensorlist_test.py",  # TODO(suderman): Re-enable once dependencies resolved
]

# keep sorted
LLVM_FAILING = [
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
    "linspace_test.py",  # TODO(https://github.com/google/iree/issues/1521)
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py",
    "quantization_dyn_test.py",
    "range_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
    "strings_test.py",
    "tensorlist_test.py",  # TODO(suderman): Re-enable once dependencies resolved
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
    "linspace_test.py",  # TODO(https://github.com/google/iree/issues/1521)
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "matrix_ops_dynamic_test.py",
    "quantization_dyn_test.py",
    "range_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
    "strings_test.py",
    "tensorlist_test.py",  # TODO(suderman): Re-enable once dependencies resolved
]

iree_e2e_cartesian_product_test_suite(
    name = "e2e_tests",
    failing_configurations = [
        {
            "target_backends": "tflite",
            "src": TFLITE_FAILING,
        },
        {
            "target_backends": "iree_vmla",
            "src": VMLA_FAILING,
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
            "iree_vmla",
            "iree_llvmaot",
            "iree_vulkan",
        ],
        "reference_backend": "tf",
    },
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
            "iree_vmla",
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
)
