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

# ***THIS FILE DOES NOT BUILD WITH BAZEL***
#
# It is open sourced to enable Bazel->CMake conversion to maintain test coverage
# of our integration tests in open source while we figure out a long term plan
# for our integration testing.

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
        "src": glob(
            ["*_test.py"],
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
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/iree:pylib_tf_support",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/debuginfo:signalsafe_addr2line_installer",
    ],
)
