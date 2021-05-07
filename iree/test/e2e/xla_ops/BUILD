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

# Tests of end-to-end IREE support for individual ops in the XLA HLO dialect.
# Each test file should have a name matching the corresponding XLA HLO op and test only the
# functionality of that op (though may make use of other ops where necessary). Tests should be
# written using the IREE Check framework and should always pass on the reference VMLA backend.
# See https://github.com/google/iree/blob/main/docs/developers/developing_iree/testing_guide.md#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

DYLIB_AND_VULKAN_CHECK_TESTS = [
    "abs.mlir",
    "add.mlir",
    "batch_norm_inference.mlir",
    "broadcast.mlir",
    "broadcast_add.mlir",
    "broadcast_in_dim.mlir",
    "clamp.mlir",
    "compare.mlir",
    "concatenate.mlir",
    "constant.mlir",
    "convert.mlir",
    "convolution.mlir",
    "cosine.mlir",
    "divide.mlir",
    "dot.mlir",
    "dot_general.mlir",
    "exponential.mlir",
    "exponential_minus_one.mlir",
    "floor.mlir",
    "gather.mlir",
    "iota.mlir",
    "log.mlir",
    "log_plus_one.mlir",
    "maximum.mlir",
    "minimum.mlir",
    "multiply.mlir",
    "negate.mlir",
    "pad.mlir",
    "reduce.mlir",
    "reduce_window.mlir",
    "remainder.mlir",
    "reshape.mlir",
    "reverse.mlir",
    "rsqrt.mlir",
    "select.mlir",
    "sine.mlir",
    "slice.mlir",
    "sqrt.mlir",
    "subtract.mlir",
    "tanh.mlir",
    "torch_index_select.mlir",
    "transpose.mlir",
    "while.mlir",
]

iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = enforce_glob(
        [
            "abs.mlir",
            "add.mlir",
            "batch_norm_inference.mlir",
            "broadcast.mlir",
            "broadcast_add.mlir",
            "broadcast_in_dim.mlir",
            "clamp.mlir",
            "compare.mlir",
            "concatenate.mlir",
            "constant.mlir",
            "convert.mlir",
            "convolution.mlir",
            "cosine.mlir",
            "divide.mlir",
            "dot.mlir",
            "dot_general.mlir",
            "exponential.mlir",
            "exponential_minus_one.mlir",
            "fft.mlir",
            "finite.mlir",
            "floor.mlir",
            "gather.mlir",
            "iota.mlir",
            "log.mlir",
            "log_plus_one.mlir",
            "maximum.mlir",
            "minimum.mlir",
            "multiply.mlir",
            "negate.mlir",
            "pad.mlir",
            "reduce.mlir",
            "reduce_window.mlir",
            "remainder.mlir",
            "reshape.mlir",
            "reverse.mlir",
            "round.mlir",
            "rsqrt.mlir",
            "select.mlir",
            "sine.mlir",
            "slice.mlir",
            "sort.mlir",
            "sqrt.mlir",
            "subtract.mlir",
            "tanh.mlir",
            "torch_index_select.mlir",
            "transpose.mlir",
            "while.mlir",
        ],
        include = ["*.mlir"],
    ),
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = DYLIB_AND_VULKAN_CHECK_TESTS,
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = DYLIB_AND_VULKAN_CHECK_TESTS + ["fft.mlir"],  #TODO(hanchung): clean this up, like VMLA test suite.
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs = [
        "add.mlir",
        "broadcast.mlir",
        "broadcast_in_dim.mlir",
        "clamp.mlir",
        "compare.mlir",
        "constant.mlir",
        "divide.mlir",
        "iota.mlir",
        "multiply.mlir",
        "negate.mlir",
        "select.mlir",
        "subtract.mlir",
    ],
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vmla_vmla",
        ":check_vulkan-spirv_vulkan",
    ],
)
