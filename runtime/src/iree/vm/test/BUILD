# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    default_visibility = ["//runtime/src/iree/vm:__subpackages__"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_BUILD_COMPILER OR NOT IREE_BUILD_TESTS)
  return()
endif()
""",
    inline = True,
)

c_embed_data(
    name = "all_bytecode_modules_c",
    srcs = [
        ":arithmetic_ops.vmfb",
        ":arithmetic_ops_f32.vmfb",
        ":arithmetic_ops_i64.vmfb",
        ":assignment_ops.vmfb",
        ":assignment_ops_f32.vmfb",
        ":assignment_ops_i64.vmfb",
        ":buffer_ops.vmfb",
        ":call_ops.vmfb",
        ":comparison_ops.vmfb",
        ":comparison_ops_f32.vmfb",
        ":comparison_ops_i64.vmfb",
        ":control_flow_ops.vmfb",
        ":conversion_ops.vmfb",
        ":conversion_ops_f32.vmfb",
        ":conversion_ops_i64.vmfb",
        ":global_ops.vmfb",
        ":global_ops_f32.vmfb",
        ":global_ops_i64.vmfb",
        ":list_ops.vmfb",
        ":list_ops_i64.vmfb",
        ":list_variant_ops.vmfb",
        ":ref_ops.vmfb",
        ":shift_ops.vmfb",
        ":shift_ops_i64.vmfb",
    ],
    c_file_output = "all_bytecode_modules.c",
    flatten = True,
    h_file_output = "all_bytecode_modules.h",
)

iree_bytecode_module(
    name = "arithmetic_ops",
    src = "arithmetic_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "arithmetic_ops_f32",
    src = "arithmetic_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "arithmetic_ops_i64",
    src = "arithmetic_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "assignment_ops",
    src = "assignment_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "assignment_ops_f32",
    src = "assignment_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "assignment_ops_i64",
    src = "assignment_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "buffer_ops",
    src = "buffer_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "call_ops",
    src = "call_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "comparison_ops",
    src = "comparison_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "comparison_ops_f32",
    src = "comparison_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "comparison_ops_i64",
    src = "comparison_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "control_flow_ops",
    src = "control_flow_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "conversion_ops",
    src = "conversion_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "conversion_ops_f32",
    src = "conversion_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "conversion_ops_i64",
    src = "conversion_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "global_ops",
    src = "global_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "global_ops_f32",
    src = "global_ops_f32.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "global_ops_i64",
    src = "global_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "list_ops",
    src = "list_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "list_ops_i64",
    src = "list_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "list_variant_ops",
    src = "list_variant_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "ref_ops",
    src = "ref_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "shift_ops",
    src = "shift_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

iree_bytecode_module(
    name = "shift_ops_i64",
    src = "shift_ops_i64.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)

c_embed_data(
    name = "async_bytecode_modules_c",
    srcs = [
        ":async_ops.vmfb",
    ],
    c_file_output = "async_bytecode_modules.c",
    flatten = True,
    h_file_output = "async_bytecode_modules.h",
)

iree_bytecode_module(
    name = "async_ops",
    src = "async_ops.mlir",
    flags = [
        "--compile-mode=vm",
    ],
)
