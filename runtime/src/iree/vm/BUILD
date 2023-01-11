# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:cc_binary_benchmark.bzl", "cc_binary_benchmark")
# load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "vm",
    hdrs = [
        "api.h",
    ],
    deps = [
        ":impl",
        "//runtime/src/iree/base",
    ],
)

# TODO(benvanik): make these srcs and only expose an api_cc.h.
iree_runtime_cc_library(
    name = "cc",
    hdrs = [
        "native_module_cc.h",
        "native_module_packing.h",
    ],
    deps = [
        ":vm",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base/internal:span",
    ],
)

#===------------------------------------------------------------------------===#
# Implementation
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "impl",
    srcs = [
        "buffer.c",
        "context.c",
        "instance.c",
        "invocation.c",
        "list.c",
        "module.c",
        "native_module.c",
        "ref.c",
        "ref_cc.h",
        "shims.c",
        "stack.c",
    ],
    hdrs = [
        "buffer.h",
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
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
    ],
)

iree_runtime_cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "native_module_test",
    srcs = ["native_module_test.cc"],
    deps = [
        ":cc",
        ":impl",
        ":native_module_test_hdrs",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_library(
    name = "native_module_test_hdrs",
    hdrs = [
        "native_module_test.h",
    ],
    deps = [
        ":impl",
        "//runtime/src/iree/base",
    ],
)

cc_binary_benchmark(
    name = "native_module_benchmark",
    srcs = ["native_module_benchmark.cc"],
    deps = [
        ":impl",
        ":native_module_test_hdrs",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_runtime_cc_test(
    name = "ref_test",
    srcs = ["ref_test.cc"],
    deps = [
        ":cc",
        ":impl",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "stack_test",
    srcs = ["stack_test.cc"],
    deps = [
        ":impl",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Bytecode interpreter module
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_disasm.c",
        "bytecode_disasm.h",
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
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal/flatcc:parsing",
        "//runtime/src/iree/schemas:bytecode_module_def_c_fbs",
    ],
)

# TODO(#357): Add a script to update bytecode_op_table.h.
# iree_gentbl_cc_library(
#     name = "bytecode_op_table_gen",
#     tbl_outs = [
#         (["--gen-iree-vm-op-table-defs"], "bytecode_op_table.h"),
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

iree_cmake_extra_content(
    content = """
if(IREE_BUILD_COMPILER)
""",
    inline = True,
)

iree_runtime_cc_test(
    name = "bytecode_module_test",
    srcs = [
        "bytecode_dispatch_async_test.cc",
        "bytecode_dispatch_test.cc",
        "bytecode_module_test.cc",
    ],
    deps = [
        ":bytecode_module",
        ":vm",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm/test:all_bytecode_modules_c",
        "//runtime/src/iree/vm/test:async_bytecode_modules_c",
    ],
)

cc_binary_benchmark(
    name = "bytecode_module_benchmark",
    testonly = True,
    srcs = ["bytecode_module_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_benchmark_module_c",
        ":vm",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_bytecode_module(
    name = "bytecode_module_benchmark_module",
    testonly = True,
    src = "bytecode_module_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_benchmark_module",
    compile_tool = "//tools:iree-compile",
    flags = ["--compile-mode=vm"],
)

cc_binary_benchmark(
    name = "bytecode_module_size_benchmark",
    srcs = ["bytecode_module_size_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_size_benchmark_module_c",
        ":vm",
        "//runtime/src/iree/base",
    ],
)

iree_bytecode_module(
    name = "bytecode_module_size_benchmark_module",
    testonly = True,
    src = "bytecode_module_size_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_size_benchmark_module",
    compile_tool = "//tools:iree-compile",
    flags = ["--compile-mode=vm"],
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

iree_runtime_cc_library(
    name = "ops",
    hdrs = [
        "ops.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal",
    ],
)

iree_runtime_cc_library(
    name = "ops_emitc",
    hdrs = [
        "ops_emitc.h",
    ],
)

iree_runtime_cc_library(
    name = "shims_emitc",
    hdrs = [
        "shims_emitc.h",
    ],
    deps = [
        ":impl",
        "//runtime/src/iree/base:core_headers",
    ],
)

exports_files(["module_impl_emitc.c"])
