# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "utils",
    srcs = [
        "block_list.c",
        "features.c",
    ],
    hdrs = [
        "block_list.h",
        "features.h",
        "generated/op_table.h",
        "isa.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal/flatcc:parsing",
        "//runtime/src/iree/schemas:bytecode_module_def_c_fbs",
        "//runtime/src/iree/vm",
    ],
)

# TODO(#357): Add a script to update op_table.h.
# iree_gentbl_cc_library(
#     name = "op_table_gen",
#     tbl_outs = [
#         (["--gen-iree-vm-op-table-defs"], "op_table.h"),
#     ],
#     tblgen = "//tools:iree-tblgen",
#     td_file = "//compiler/src/iree/compiler/Dialect/VM/IR:VMOps.td",
#     deps = [
#         "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
#         "//compiler/src/iree/compiler/Dialect/VM/IR:td_files",
#         "@llvm-project//mlir:CallInterfacesTdFiles",
#         "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
#         "@llvm-project//mlir:FunctionInterfacesTdFiles",
#         "@llvm-project//mlir:OpBaseTdFiles",
#         "@llvm-project//mlir:SideEffectInterfacesTdFiles",
#     ],
# )

iree_runtime_cc_test(
    name = "block_list_test",
    srcs = ["block_list_test.cc"],
    deps = [
        ":utils",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
    ],
)
