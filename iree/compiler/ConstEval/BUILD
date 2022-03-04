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
        # TODO: Prune.
        "//iree/compiler/Bindings/Native/Transforms",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/Stream/Transforms",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
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
        "//iree/hal",
        "//iree/hal/vmvx/registration",
        "//iree/modules/hal",
        "//iree/tools/utils:vm_util",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@llvm-project//mlir:IR",
    ],
)
