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
# files. A coverage table generated from this file can be viewed here:
#   https://google.github.io/iree/tf-e2e-coverage
# Updates made to test suite names should also be reflected here:
#   https://github.com/google/iree/blob/main/scripts/update_e2e_coverage.py

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_binary",
)
load(
    "//integrations/tensorflow/e2e:iree_e2e_test_suite.bzl",
    "iree_e2e_test_suite",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

# Create binaries for all test srcs to allow them to be run manually.
[
    iree_py_binary(
        name = src.replace(".py", "_manual"),
        srcs = [src],
        main = src,
        python_version = "PY3",
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for src in glob(["*_test.py"])
]

# Special cases to exclude from automatically expanding targets for all
# backends.
# keep sorted
SPECIAL_CASES = [
    "mobile_bert_squad_test.py",
]

# keep sorted
TFLITE_FAILING = [
    "broadcasting_test.py",
    "complex_test.py",
    "concat_test.py",
    "dynamic_mlp_relu_test.py",
    "dynamic_mlp_test.py",
    "einsum_dynamic_test.py",
    "einsum_static_test.py",
    "einsum_vector_test.py",
    "fft_test.py",
    "finite_test.py",
    "gather_test.py",
    "image_resize_test.py",
    "mandelbrot_test.py",
    "math_dyn_test.py",
    "matrix_ops_dynamic_test.py",
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
    "fft_test.py",  # TODO(natashaknk): Get this working after kernel is in.
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "ring_buffer_test.py",  # TODO(b/148747011)
    "strings_test.py",
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
    "logical_ops_test.py",
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "math_dyn_test.py",
    "matrix_ops_dynamic_test.py",
    "range_test.py",
    "reduce_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
    "strings_test.py",
]

# keep sorted
VULKAN_FAILING = [
    "bool_test.py",
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
    "logical_ops_test.py",
    "mandelbrot_test.py",  # TODO(silvasean): Get this working on IREE.
    "math_dyn_test.py",
    "matrix_ops_dynamic_test.py",
    "range_test.py",
    "reduce_test.py",
    "ring_buffer_test.py",  # TODO(b/148747011)
    "scatter_update_test.py",
    "strings_test.py",
]

TF_PASSING = glob(
    ["*_test.py"],
    exclude = SPECIAL_CASES,
)

TFLITE_PASSING = glob(
    ["*_test.py"],
    exclude = TFLITE_FAILING + SPECIAL_CASES,
)

VMLA_PASSING = glob(
    ["*_test.py"],
    exclude = VMLA_FAILING + SPECIAL_CASES,
)

LLVM_PASSING = glob(
    ["*_test.py"],
    exclude = LLVM_FAILING + SPECIAL_CASES,
)

VULKAN_PASSING = glob(
    ["*_test.py"],
    exclude = VULKAN_FAILING + SPECIAL_CASES,
)

iree_e2e_test_suite(
    name = "e2e_tests",
    backends_to_srcs = {
        "tf": TF_PASSING,
        "tflite": TFLITE_PASSING,
        "iree_vmla": VMLA_PASSING,
        "iree_llvmjit": LLVM_PASSING,
        "iree_vulkan": VULKAN_PASSING,
    },
    reference_backend = "tf",
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_e2e_test_suite(
    name = "e2e_tests_failing",
    backends_to_srcs = {
        "tflite": TFLITE_FAILING,
        "iree_vmla": VMLA_FAILING,
        "iree_llvmjit": LLVM_FAILING,
        "iree_vulkan": VULKAN_FAILING,
    },
    reference_backend = "tf",
    tags = [
        "failing",
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)

iree_e2e_test_suite(
    name = "mobile_bert_squad_tests",
    size = "enormous",
    backends_to_srcs = {
        "tf": ["mobile_bert_squad_test.py"],
        "tflite": ["mobile_bert_squad_test.py"],
        "iree_vmla": ["mobile_bert_squad_test.py"],
        "iree_llvmjit": ["mobile_bert_squad_test.py"],
        "iree_vulkan": ["mobile_bert_squad_test.py"],
    },
    reference_backend = "tf",
    tags = [
        "external",
        "guitar",
        "manual",
        "no-remote",
        "nokokoro",
        "notap",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
