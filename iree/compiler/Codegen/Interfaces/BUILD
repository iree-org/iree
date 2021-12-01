# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "ProcessorOpInterfaces.td",
])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "ProcessorOpInterfaces.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "Interfaces",
    srcs = [
        "Interfaces.cpp",
    ],
    hdrs = [
        "Interfaces.h",
    ],
    deps = [
        ":BufferizationInterfaces",
        ":ProcessorOpInterfaces",
    ],
)

cc_library(
    name = "BufferizationInterfaces",
    srcs = [
        "BufferizationInterfaces.cpp",
    ],
    hdrs = [
        "BufferizationInterfaces.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "@llvm-project//mlir:AffineBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:ArithBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:BufferizableOpInterface",
        "@llvm-project//mlir:ComprehensiveBufferize",
        "@llvm-project//mlir:LinalgBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:TensorBufferizableOpInterfaceImpl",
        "@llvm-project//mlir:VectorBufferizableOpInterfaceImpl",
    ],
)

cc_library(
    name = "ProcessorOpInterfaces",
    srcs = [
        "ProcessorOpInterfaces.cpp",
    ],
    hdrs = [
        "ProcessorOpInterfaces.h",
    ],
    textual_hdrs = [
        "ProcessorOpInterfaces.cpp.inc",
        "ProcessorOpInterfaces.h.inc",
    ],
    deps = [
        ":ProcessorOpInterfaceGen",
        "//iree/compiler/Dialect/HAL/IR",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "ProcessorOpInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "ProcessorOpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ProcessorOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ProcessorOpInterfaces.td",
    deps = [":td_files"],
)
