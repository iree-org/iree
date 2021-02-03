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
# See https://google.github.io/iree/TestingGuide#iree-core-end-to-end-tests.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = glob(["*.mlir"]),
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = [
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
    ],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = [
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
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_linalg_on_tensors_dylib-llvm-aot_dylib",
    srcs = [
        "abs.mlir",
        "add.mlir",
        "batch_norm_inference.mlir",
        "broadcast.mlir",
        "broadcast_add.mlir",
        "broadcast_in_dim.mlir",
        # https://github.com/google/iree/issues/4692
        "clamp.mlir",
        "compare.mlir",
        # https://github.com/google/iree/issues/4079
        # "concatenate.mlir",
        "constant.mlir",
        # https://github.com/google/iree/issues/4079
        # "convolution.mlir",
        "cosine.mlir",
        "divide.mlir",
        # https://github.com/google/iree/issues/4079
        # "dot.mlir",
        # "dot_general.mlir",
        "exponential.mlir",
        # https://github.com/google/iree/issues/4692
        # "exponential_minus_one.mlir",
        "floor.mlir",
        # https://github.com/google/iree/issues/4692
        # "gather.mlir",
        "iota.mlir",
        "log.mlir",
        # https://github.com/google/iree/issues/4692
        # "log_plus_one.mlir",
        # "maximum.mlir",
        # "minimum.mlir",
        "multiply.mlir",
        "negate.mlir",
        # https://github.com/google/iree/issues/4079
        # "pad.mlir",
        # "reduce.mlir",
        # "reduce_window.mlir",
        "remainder.mlir",
        # "reshape.mlir",
        # "reverse.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sine.mlir",
        # https://github.com/google/iree/issues/4692
        # "slice.mlir",
        "sqrt.mlir",
        "subtract.mlir",
        "tanh.mlir",
        # https://github.com/google/iree/issues/4079
        # "torch_index_select.mlir",
        "transpose.mlir",
        # "while.mlir",
    ],
    compiler_flags = [
        "-iree-flow-dispatch-linalg-on-tensors",
        "-iree-codegen-llvm-experimental-linalg-on-tensors",
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vmla_vmla",
        ":check_vulkan-spirv_vulkan",
    ],
)
