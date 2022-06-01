# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

# Allow all python sources to be built externally.
exports_files(glob([
    "python/*.cpp",
    "python/*.c",
    "python/**/*.py",
    "unittests/**/*.py",
]))

################################################################################
# Package-by-package filegroup exports.
# Some tools can make use of such non-nested groups.
################################################################################

filegroup(
    name = "python_compiler_tools",
    srcs = glob(["python/iree/compiler/tools/*.py"]),
)

filegroup(
    name = "python_compiler_tools_deprecated_redirects",
    srcs = [
        "python/iree/compiler/__init__.py",
        "python/iree/compiler/tf.py",
        "python/iree/compiler/tflite.py",
        "python/iree/compiler/xla.py",
    ],
)

filegroup(
    name = "python_compiler_api",
    srcs = glob(["python/iree/compiler/transforms/*.py"]),
)

filegroup(
    name = "unittests_tools_testdata",
    srcs = glob([
        "python/test/tools/testdata/*",
    ]),
)

################################################################################
# CAPI
################################################################################

iree_compiler_cc_library(
    name = "CAPI",
    srcs = [
        "Compiler.cpp",
        "Lld.cpp",
        "Tools.cpp",
    ],
    hdrs = [
        "Compiler.h",
        "Tools.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/ConstEval",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "//compiler/src/iree/compiler/Dialect/VM/Target/Bytecode",
        "//compiler/src/iree/compiler/InputConversion/MHLO",
        "//compiler/src/iree/compiler/InputConversion/TOSA",
        "//compiler/src/iree/compiler/Pipelines",
        "//compiler/src/iree/compiler/Tools:init_llvmir_translations",
        "//compiler/src/iree/compiler/Tools:init_passes_and_dialects",
        "//compiler/src/iree/compiler/Tools:init_targets",
        "//compiler/src/iree/compiler/Tools:iree_compile_lib",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//lld:COFF",
        "@llvm-project//lld:Common",
        "@llvm-project//lld:ELF",
        "@llvm-project//lld:MachO",
        "@llvm-project//lld:Wasm",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
    ],
)
