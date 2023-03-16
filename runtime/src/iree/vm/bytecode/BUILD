# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:cc_binary_benchmark.bzl", "cc_binary_benchmark")
# load(//build_tools/bazel:build_defs.oss.bzl, "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Bytecode interpreter module
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "module",
    srcs = [
        "archive.c",
        "disassembler.c",
        "disassembler.h",
        "dispatch.c",
        "dispatch_util.h",
        "module.c",
        "module_impl.h",
        "verifier.c",
        "verifier.h",
    ],
    hdrs = [
        "archive.h",
        "module.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:ops",
        "//runtime/src/iree/vm/bytecode/utils",
    ],
)

iree_cmake_extra_content(
    content = """
if(IREE_BUILD_COMPILER)
""",
    inline = True,
)

iree_runtime_cc_test(
    name = "module_test",
    srcs = [
        "dispatch_async_test.cc",
        "dispatch_test.cc",
        "module_test.cc",
    ],
    deps = [
        ":module",
        ":module_test_module_c",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm/test:all_bytecode_modules_c",
        "//runtime/src/iree/vm/test:async_bytecode_modules_c",
    ],
)

iree_bytecode_module(
    name = "module_test_module",
    testonly = True,
    src = "module_test.mlir",
    c_identifier = "iree_vm_bytecode_module_test_module",
    flags = ["--compile-mode=vm"],
)

cc_binary_benchmark(
    name = "module_benchmark",
    testonly = True,
    srcs = ["module_benchmark.cc"],
    deps = [
        ":module",
        ":module_benchmark_module_c",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:benchmark_main",
        "//runtime/src/iree/vm",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_bytecode_module(
    name = "module_benchmark_module",
    testonly = True,
    src = "module_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_benchmark_module",
    flags = ["--compile-mode=vm"],
)

cc_binary_benchmark(
    name = "module_size_benchmark",
    srcs = ["module_size_benchmark.cc"],
    deps = [
        ":module",
        ":module_size_benchmark_module_c",
        "//runtime/src/iree/base",
        "//runtime/src/iree/vm",
    ],
)

iree_bytecode_module(
    name = "module_size_benchmark_module",
    testonly = True,
    src = "module_size_benchmark.mlir",
    c_identifier = "iree_vm_bytecode_module_size_benchmark_module",
    flags = ["--compile-mode=vm"],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
