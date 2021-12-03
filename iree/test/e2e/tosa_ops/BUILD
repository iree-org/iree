# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests of end-to-end IREE support for individual ops in the TOSA dialect.
# Each test file should have a name matching the corresponding TOSA op and test only the
# functionality of that op (though may make use of other ops where necessary). Tests should be
# written using the IREE Check framework.
# See https://github.com/google/iree/blob/main/docs/developers/developing_iree/testing_guide.md#iree-core-end-to-end-tests.

load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

LLVM_SRCS = enforce_glob(
    [
        "abs.mlir",
        "add.mlir",
        "arithmetic_right_shift.mlir",
        "bitwise_and.mlir",
        "bitwise_or.mlir",
        "bitwise_xor.mlir",
        "ceil.mlir",
        "clamp.mlir",
        "clz.mlir",
        "const.mlir",
        "equal.mlir",
        "exp.mlir",
        "floor.mlir",
        "fully_connected.mlir",
        "gather.mlir",
        "greater.mlir",
        "greater_equal.mlir",
        "if.mlir",
        "log.mlir",
        "logical_left_shift.mlir",
        "logical_right_shift.mlir",
        "matmul.mlir",
        "max_pool.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "mul.mlir",
        "negate.mlir",
        "pad.mlir",
        "reciprocal.mlir",
        "reduce.mlir",
        "reluN.mlir",
        "reshape.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sigmoid.mlir",
        "sub.mlir",
        "table.mlir",
        "tanh.mlir",
        "transpose.mlir",
        "while.mlir",
    ],
    include = ["*.mlir"],
)

iree_check_single_backend_test_suite(
    name = "check_dylib-llvm-aot_dylib",
    srcs = LLVM_SRCS,
    compiler_flags = ["-iree-input-type=tosa"],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

iree_check_single_backend_test_suite(
    name = "check_dylib_embedded-llvm-aot_dylib",
    srcs = LLVM_SRCS,
    compiler_flags = [
        "-iree-input-type=tosa",
        "-iree-llvm-link-embedded=true",
    ],
    driver = "dylib",
    target_backend = "dylib-llvm-aot",
)

VMVX_SRCS = enforce_glob(
    [
        "abs.mlir",
        "add.mlir",
        "arithmetic_right_shift.mlir",
        "bitwise_and.mlir",
        "bitwise_or.mlir",
        "bitwise_xor.mlir",
        "ceil.mlir",
        "clamp.mlir",
        "clz.mlir",
        "const.mlir",
        "equal.mlir",
        "exp.mlir",
        "floor.mlir",
        "fully_connected.mlir",
        "gather.mlir",
        "greater.mlir",
        "greater_equal.mlir",
        "if.mlir",
        "log.mlir",
        "logical_left_shift.mlir",
        "logical_right_shift.mlir",
        "matmul.mlir",
        "max_pool.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "mul.mlir",
        "negate.mlir",
        "pad.mlir",
        "reciprocal.mlir",
        "reluN.mlir",
        "reshape.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sigmoid.mlir",
        "sub.mlir",
        "table.mlir",
        "tanh.mlir",
        "transpose.mlir",
        "while.mlir",
    ],
    include = ["*.mlir"],
    exclude = [
        "reduce.mlir",  # Currently flakey https://github.com/google/iree/issues/5885
    ],
)

iree_check_single_backend_test_suite(
    name = "check_vmvx_vmvx",
    srcs = VMVX_SRCS,
    compiler_flags = [
        "-iree-input-type=tosa",
        # tosa.apply_scale currently expands to i64 math, requiring this
        # extension. Nothing else about the tests require it, though, and we
        # really should not be emulating it with 64-bit math. Having this
        # prevents any of these tests from being executed on a runtime compiled
        # without the VM's I64 extension.
        "-iree-vm-target-extension-i64",
    ],
    driver = "vmvx",
    target_backend = "vmvx",
)

VULKAN_SRCS = enforce_glob(
    [
        "abs.mlir",
        "add.mlir",
        "arithmetic_right_shift.mlir",
        "bitwise_and.mlir",
        "bitwise_or.mlir",
        "bitwise_xor.mlir",
        "ceil.mlir",
        "clamp.mlir",
        "clz.mlir",
        "const.mlir",
        "equal.mlir",
        "exp.mlir",
        "floor.mlir",
        "fully_connected.mlir",
        "gather.mlir",
        "greater.mlir",
        "greater_equal.mlir",
        "if.mlir",
        "log.mlir",
        "logical_left_shift.mlir",
        "logical_right_shift.mlir",
        "matmul.mlir",
        "max_pool.mlir",
        "maximum.mlir",
        "minimum.mlir",
        "mul.mlir",
        "negate.mlir",
        "pad.mlir",
        "reciprocal.mlir",
        "reduce.mlir",
        "reluN.mlir",
        "reshape.mlir",
        "rsqrt.mlir",
        "select.mlir",
        "sigmoid.mlir",
        "sub.mlir",
        "table.mlir",
        "tanh.mlir",
        "transpose.mlir",
        "while.mlir",
    ],
    include = ["*.mlir"],
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = VULKAN_SRCS,
    compiler_flags = ["-iree-input-type=tosa"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

test_suite(
    name = "check",
    tests = [
        ":check_dylib-llvm-aot_dylib",
        ":check_vmvx_vmvx",
        ":check_vulkan-spirv_vulkan",
    ],
)
