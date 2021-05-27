# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VulkanAttributes.td",
            "VulkanBase.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "VulkanAttributes.cpp",
        "VulkanAttributes.cpp.inc",
        "VulkanDialect.cpp",
        "VulkanEnums.cpp.inc",
        "VulkanTypes.cpp",
    ],
    hdrs = [
        "VulkanAttributes.h",
        "VulkanAttributes.h.inc",
        "VulkanDialect.h",
        "VulkanEnums.h.inc",
        "VulkanTypes.h",
    ],
    deps = [
        ":VulkanAttrsGen",
        ":VulkanEnumsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "VulkanAttrsGen",
    tbl_outs = [
        (
            ["-gen-struct-attr-decls"],
            "VulkanAttributes.h.inc",
        ),
        (
            ["-gen-struct-attr-defs"],
            "VulkanAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanAttributes.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "VulkanEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "VulkanEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "VulkanEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VulkanBase.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)
