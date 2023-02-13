# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "MicroKernelOpInterface.td",
    "ProcessorOpInterfaces.td",
])

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "MicroKernelOpInterface.td",
            "PartitionableLoopsInterface.td",
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
        ":MicroKernelOpInterface",
        ":PartitionableLoopsInterface",
        ":ProcessorOpInterfaces",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransformOps",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        # TransformExtensions
        "//compiler/src/iree/compiler/Codegen/Common/TransformExtensions:CommonExtensions",
        "//compiler/src/iree/compiler/Codegen/LLVMCPU/TransformExtensions:LLVMCPUExtensions",
        "//compiler/src/iree/compiler/Codegen/LLVMGPU/TransformExtensions:LLVMGPUExtensions",
        "//compiler/src/iree/compiler/Dialect/Flow/TransformExtensions:FlowExtensions",
        "@llvm-project//mlir:AffineTransformOps",
        "@llvm-project//mlir:BufferizationTransformOps",
        "@llvm-project//mlir:GPUTransformOps",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:MemRefTransformOps",
        "@llvm-project//mlir:SCFTransformOps",
        "@llvm-project//mlir:VectorTransformOps",
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
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:VectorTransforms",
    ],
)

iree_compiler_cc_library(
    name = "MicroKernelOpInterface",
    srcs = [
        "MicroKernelOpInterface.cpp",
    ],
    hdrs = [
        "MicroKernelOpInterface.h",
    ],
    textual_hdrs = [
        "MicroKernelOpInterface.cpp.inc",
        "MicroKernelOpInterface.h.inc",
    ],
    deps = [
        ":MicroKernelOpInterfaceGen",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

iree_gentbl_cc_library(
    name = "MicroKernelOpInterfaceGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "MicroKernelOpInterface.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "MicroKernelOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MicroKernelOpInterface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
)

iree_compiler_cc_library(
    name = "PartitionableLoopsInterface",
    srcs = [
        "PartitionableLoopsInterface.cpp",
    ],
    hdrs = [
        "PartitionableLoopsInterface.h",
    ],
    textual_hdrs = [
        "PartitionableLoopsInterface.cpp.inc",
        "PartitionableLoopsInterface.h.inc",
    ],
    deps = [
        ":PartitionableLoopsInterfaceGen",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgStructuredOpsIncGen",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)

iree_gentbl_cc_library(
    name = "PartitionableLoopsInterfaceGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "PartitionableLoopsInterface.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "PartitionableLoopsInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "PartitionableLoopsInterface.td",
    deps = ["@llvm-project//mlir:OpBaseTdFiles"],
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
            ["--gen-op-interface-decls"],
            "ProcessorOpInterfaces.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "ProcessorOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ProcessorOpInterfaces.td",
    deps = [":td_files"],
)
