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

# Tests of end-to-end IREE support for individual ops in the TOSA dialect.
# Each test file should have a name matching the corresponding TOSA op and test only the
# functionality of that op (though may make use of other ops where necessary). Tests should be
# written using the IREE Check framework.
# See https://google.github.io/iree/developing-iree/testing-guide#iree-core-end-to-end-tests.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

VULKAN_SRCS = enforce_glob(
    [
        "abs.mlir",
        "add.mlir",
        "bitwise_and.mlir",
        "bitwise_or.mlir",
        "bitwise_xor.mlir",
        "ceil.mlir",
        "clamp.mlir",
        "const.mlir",
        "exp.mlir",
        "floor.mlir",
        "greater.mlir",
        "greater_equal.mlir",
        "if.mlir",
        "log.mlir",
        "logical_left_shift.mlir",
        "logical_right_shift.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "mul.mlir",
        "negate.mlir",
        "pad.mlir",
        "reluN.mlir",
        "reshape.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sub.mlir",
        "table.mlir",
        "tanh.mlir",
        "transpose.mlir",
        "while.mlir",
    ],
    include = ["*.mlir"],
    exclude = [
        "reduce.mlir",  # Currently failing on vulkan.
        "max_pool.mlir",  # Currently failing on vulkan.
    ],
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = VULKAN_SRCS,
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

LLVM_SRCS = enforce_glob(
    [
        "abs.mlir",
        "add.mlir",
        "bitwise_and.mlir",
        "bitwise_or.mlir",
        "bitwise_xor.mlir",
        "ceil.mlir",
        "clamp.mlir",
        "const.mlir",
        "exp.mlir",
        "floor.mlir",
        "greater.mlir",
        "greater_equal.mlir",
        "if.mlir",
        "log.mlir",
        "logical_left_shift.mlir",
        "logical_right_shift.mlir",
        "max_pool.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "mul.mlir",
        "negate.mlir",
        "pad.mlir",
        "reduce.mlir",
        "reluN.mlir",
        "reshape.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sub.mlir",
        "table.mlir",
        "tanh.mlir",
        "transpose.mlir",
        "while.mlir",
    ],
    include = ["*.mlir"],
    exclude = [
    ],
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = LLVM_SRCS,
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vulkan-spirv_vulkan",
    ],
)
