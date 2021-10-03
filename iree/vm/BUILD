# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")
# load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
    name = "vm",
    hdrs = [
        "api.h",
    ],
    deps = [
        ":impl",
        "//iree/base",
    ],
)

# TODO(benvanik): make these srcs and only expose an api_cc.h.
cc_library(
    name = "cc",
    hdrs = [
        "native_module_cc.h",
        "native_module_packing.h",
        "ref_cc.h",
    ],
    deps = [
        ":vm",
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:core_headers",
        "//iree/base/internal:span",
    ],
)

#===------------------------------------------------------------------------===#
# Implementation
#===------------------------------------------------------------------------===#

cc_library(
    name = "impl",
    srcs = [
        "buffer.c",
        "builtin_types.c",
        "context.c",
        "instance.c",
        "invocation.c",
        "list.c",
        "module.c",
        "native_module.c",
        "ref.c",
        "shims.c",
        "stack.c",
    ],
    hdrs = [
        "buffer.h",
        "builtin_types.h",
        "context.h",
        "instance.h",
        "invocation.h",
        "list.h",
        "module.h",
        "native_module.h",
        "ref.h",
        "shims.h",
        "stack.h",
        "type_def.h",
        "value.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "native_module_test",
    srcs = ["native_module_test.cc"],
    deps = [
        ":cc",
        ":impl",
        ":native_module_test_hdrs",
        "//iree/base",
        "//iree/base:cc",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "native_module_test_hdrs",
    hdrs = [
        "native_module_test.h",
    ],
    deps = [
        ":impl",
        "//iree/base",
    ],
)

cc_test(
    name = "native_module_benchmark",
    srcs = ["native_module_benchmark.cc"],
    deps = [
        ":impl",
        ":native_module_test_hdrs",
        "//iree/base",
        "//iree/base:logging",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "ref_test",
    srcs = ["ref_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "stack_test",
    srcs = ["stack_test.cc"],
    deps = [
        ":impl",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Bytecode interpreter module
#===------------------------------------------------------------------------===#

cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_dispatch.c",
        "bytecode_dispatch_util.h",
        "bytecode_module.c",
        "bytecode_module_impl.h",
        "generated/bytecode_op_table.h",
    ],
    hdrs = [
        "bytecode_module.h",
    ],
    deps = [
        ":ops",
        ":vm",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal/flatcc:parsing",
        "//iree/schemas:bytecode_module_def_c_fbs",
    ],
)

# TODO(#357): Add a script to update bytecode_op_table.h.
# gentbl_cc_library(
#     name = "bytecode_op_table_gen",
#     tbl_outs = [
#         (["-gen-iree-vm-op-table-defs"], "bytecode_op_table.h"),
#     ],
#     tblgen = "//iree/tools:iree-tblgen",
#     td_file = "//iree/compiler/Dialect/VM/IR:VMOps.td",
#     td_srcs = [
#         "//iree/compiler/Dialect/Util/IR:td_files",
#         "//iree/compiler/Dialect/VM/IR:td_files",
#         "@llvm-project//mlir:OpBaseTdFiles",
#         "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
#         "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
#         "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
#         "@llvm-project//mlir:SideEffectTdFiles",
#     ],
# )

iree_cmake_extra_content(
    content = """
if(${IREE_BUILD_COMPILER})
""",
    inline = True,
)

cc_test(
    name = "bytecode_module_test",
    srcs = [
        "bytecode_dispatch_test.cc",
        "bytecode_module_test.cc",
    ],
    deps = [
        ":bytecode_module",
        ":vm",
        "//iree/base:cc",
        "//iree/base:logging",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm/test:all_bytecode_modules_c",
    ],
)

cc_binary(
    name = "bytecode_module_benchmark",
    testonly = True,
    srcs = ["bytecode_module_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_benchmark_module_c",
        ":vm",
        "//iree/base",
        "//iree/base:logging",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

run_binary_test(
    name = "bytecode_module_benchmark_test",
    args = ["--benchmark_min_time=0"],
    test_binary = ":bytecode_module_benchmark",
)

iree_bytecode_module(
    name = "bytecode_module_benchmark_module",
    testonly = True,
    src = "bytecode_module_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_benchmark_module",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

cc_test(
    name = "bytecode_module_size_benchmark",
    srcs = ["bytecode_module_size_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_size_benchmark_module_c",
        ":vm",
        "//iree/base",
    ],
)

iree_bytecode_module(
    name = "bytecode_module_size_benchmark_module",
    testonly = True,
    src = "bytecode_module_size_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_size_benchmark_module",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

#===------------------------------------------------------------------------===#
# Common VM op implementations
#===------------------------------------------------------------------------===#

cc_library(
    name = "ops",
    hdrs = [
        "ops.h",
    ],
    deps = [
        "//iree/base",
    ],
)

cc_library(
    name = "ops_emitc",
    hdrs = [
        "ops_emitc.h",
    ],
)

cc_library(
    name = "shims_emitc",
    hdrs = [
        "shims_emitc.h",
    ],
)
