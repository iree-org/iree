# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "PassHeaders",
    hdrs = [
        "PassDetail.h",
        "Passes.h",
        "Passes.h.inc",
    ],
    deps = [
        ":PassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "ConstEval",
    srcs = [
        "JitGlobals.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":PassHeaders",
        ":PassesIncGen",
        ":Runtime",
        "//iree/compiler/Pipelines",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "Runtime",
    srcs = [
        "Runtime.cpp",
    ],
    hdrs = [
        "Runtime.h",
    ],
    deps = [
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/tools/utils:vm_util",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/vmvx/registration",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
        "//runtime/src/iree/vm:cc",
        "@llvm-project//mlir:IR",
    ],
)
