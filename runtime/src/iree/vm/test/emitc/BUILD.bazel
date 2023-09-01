# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_c_module.bzl", "iree_c_module")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_test(
    name = "module_test",
    srcs = ["module_test.cc"],
    deps = [
        ":arithmetic_ops",
        ":arithmetic_ops_f32",
        ":arithmetic_ops_i64",
        ":assignment_ops",
        ":assignment_ops_f32",
        ":assignment_ops_i64",
        ":buffer_ops",
        ":call_ops",
        ":comparison_ops",
        ":comparison_ops_f32",
        ":comparison_ops_i64",
        ":control_flow_ops",
        ":conversion_ops",
        ":conversion_ops_f32",
        ":conversion_ops_i64",
        ":global_ops",
        ":global_ops_f32",
        ":global_ops_i64",
        ":list_ops",
        ":list_variant_ops",
        ":ref_ops",
        ":shift_ops",
        ":shift_ops_i64",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:ops",
        "//runtime/src/iree/vm:ops_emitc",
        "//runtime/src/iree/vm:shims_emitc",
    ],
)

iree_c_module(
    name = "arithmetic_ops",
    src = "//runtime/src/iree/vm/test:arithmetic_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "arithmetic_ops.h",
)

iree_c_module(
    name = "arithmetic_ops_f32",
    src = "//runtime/src/iree/vm/test:arithmetic_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "arithmetic_ops_f32.h",
)

iree_c_module(
    name = "arithmetic_ops_i64",
    src = "//runtime/src/iree/vm/test:arithmetic_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "arithmetic_ops_i64.h",
)

iree_c_module(
    name = "assignment_ops",
    src = "//runtime/src/iree/vm/test:assignment_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "assignment_ops.h",
)

iree_c_module(
    name = "assignment_ops_f32",
    src = "//runtime/src/iree/vm/test:assignment_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "assignment_ops_f32.h",
)

iree_c_module(
    name = "assignment_ops_i64",
    src = "//runtime/src/iree/vm/test:assignment_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "assignment_ops_i64.h",
)

iree_c_module(
    name = "buffer_ops",
    src = "//runtime/src/iree/vm/test:buffer_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "buffer_ops.h",
)

iree_c_module(
    name = "call_ops",
    src = "//runtime/src/iree/vm/test:call_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "call_ops.h",
)

iree_c_module(
    name = "comparison_ops",
    src = "//runtime/src/iree/vm/test:comparison_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "comparison_ops.h",
)

iree_c_module(
    name = "comparison_ops_f32",
    src = "//runtime/src/iree/vm/test:comparison_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "comparison_ops_f32.h",
)

iree_c_module(
    name = "comparison_ops_i64",
    src = "//runtime/src/iree/vm/test:comparison_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "comparison_ops_i64.h",
)

iree_c_module(
    name = "control_flow_ops",
    src = "//runtime/src/iree/vm/test:control_flow_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "control_flow_ops.h",
)

iree_c_module(
    name = "conversion_ops",
    src = "//runtime/src/iree/vm/test:conversion_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "conversion_ops.h",
)

iree_c_module(
    name = "conversion_ops_f32",
    src = "//runtime/src/iree/vm/test:conversion_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "conversion_ops_f32.h",
)

iree_c_module(
    name = "conversion_ops_i64",
    src = "//runtime/src/iree/vm/test:conversion_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "conversion_ops_i64.h",
)

iree_c_module(
    name = "global_ops",
    src = "//runtime/src/iree/vm/test:global_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "global_ops.h",
)

iree_c_module(
    name = "global_ops_f32",
    src = "//runtime/src/iree/vm/test:global_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "global_ops_f32.h",
)

iree_c_module(
    name = "global_ops_i64",
    src = "//runtime/src/iree/vm/test:global_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "global_ops_i64.h",
)

iree_c_module(
    name = "list_ops",
    src = "//runtime/src/iree/vm/test:list_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "list_ops.h",
)

iree_c_module(
    name = "list_variant_ops",
    src = "//runtime/src/iree/vm/test:list_variant_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "list_variant_ops.h",
)

iree_c_module(
    name = "ref_ops",
    src = "//runtime/src/iree/vm/test:ref_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "ref_ops.h",
)

iree_c_module(
    name = "shift_ops",
    src = "//runtime/src/iree/vm/test:shift_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "shift_ops.h",
)

iree_c_module(
    name = "shift_ops_i64",
    src = "//runtime/src/iree/vm/test:shift_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "shift_ops_i64.h",
)
