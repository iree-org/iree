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

cc_library(
    name = "Transforms",
    srcs = [
        "AnnotateDispatchArguments.cpp",
        "ConvertToStream.cpp",
        "DumpStatistics.cpp",
        "ElideAsyncCopies.cpp",
        "EncodeTensors.cpp",
        "FoldUniformOperands.cpp",
        "FuseDispatchBindings.cpp",
        "LayoutSlices.cpp",
        "MaterializeBuiltins.cpp",
        "MaterializeCopyOnWrite.cpp",
        "OutlineConstants.cpp",
        "PackAllocations.cpp",
        "PackConstants.cpp",
        "PassDetail.h",
        "Passes.cpp",
        "PropagateSubviews.cpp",
        "PropagateTimepoints.cpp",
        "RefineUsage.cpp",
        "ScheduleAllocation.cpp",
        "ScheduleConcurrency.cpp",
        "ScheduleExecution.cpp",
        "SpecializeDispatches.cpp",
        "VerifyLowerings.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Passes.h.inc",
    ],
    deps = [
        ":PassesIncGen",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Stream/Analysis",
        "//iree/compiler/Dialect/Stream/Conversion",
        "//iree/compiler/Dialect/Stream/Conversion/FlowToStream",
        "//iree/compiler/Dialect/Stream/Conversion/HALToStream",
        "//iree/compiler/Dialect/Stream/Conversion/StandardToStream",
        "//iree/compiler/Dialect/Stream/Conversion/UtilToStream",
        "//iree/compiler/Dialect/Stream/IR",
        "//iree/compiler/Dialect/Util/Analysis",
        "//iree/compiler/Dialect/Util/Analysis/DFX",
        "//iree/compiler/Dialect/Util/Conversion",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorOps",
    ],
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
