# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

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

iree_compiler_cc_library(
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
        # TODO: Remove this dependency once the transform dialect extensions
        # have a better registration mechanism.
        "//compiler/src/iree/compiler/Codegen/TransformDialectExtensions",
    ],
)

iree_compiler_cc_library(
    name = "BufferizationInterfaces",
    srcs = [
        "BufferizationInterfaces.cpp",
    ],
    hdrs = [
        "BufferizationInterfaces.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "@llvm-project//mlir:ArithmeticTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

iree_compiler_cc_library(
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
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
    ],
)

iree_gentbl_cc_library(
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
