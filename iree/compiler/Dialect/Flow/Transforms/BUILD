# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")
load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

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
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "Transforms",
    srcs = [
        "ConvertToFlowTensorOps.cpp",
        "DeduplicateExecutables.cpp",
        "DestructiveUpdateUtils.cpp",
        "DispatchLinalgOnTensors.cpp",
        "ExpandVariableDynamicDims.cpp",
        "ExportBenchmarkFuncs.cpp",
        "FormStreams.cpp",
        "HLOToHLOPreprocessing.cpp",
        "HoistUnstreamableOps.cpp",
        "InjectDispatchTracing.cpp",
        "LegalizeInputTypes.cpp",
        "OutlineDispatchRegions.cpp",
        "OutlineLargeConstants.cpp",
        "PassDetail.h",
        "Passes.cpp",
        "PromoteI1ToI8Pass.cpp",
        "PromoteTensorLoads.cpp",
        "StripAndSplatConstantVariables.cpp",
        "VerifyCompilerInputLegality.cpp",
    ],
    hdrs = [
        "DestructiveUpdateUtils.h",
        "Passes.h",
        "Passes.h.inc",
    ],
    deps = [
        ":PassesIncGen",
        "//iree/compiler/Conversion:PassHeaders",
        "//iree/compiler/Conversion/Common",
        "//iree/compiler/Conversion/HLOToLinalg",
        "//iree/compiler/Conversion/LinalgToLinalg",
        "//iree/compiler/Dialect/Flow/Conversion",
        "//iree/compiler/Dialect/Flow/Conversion/HLOToFlow",
        "//iree/compiler/Dialect/Flow/Conversion/StandardToFlow",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Dialect/Shape/Utils:TypeConversion",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:TosaToLinalg",
        "@llvm-project//mlir:TosaToSCF",
        "@llvm-project//mlir:TosaToStandard",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:chlo_legalize_to_hlo",
        "@mlir-hlo//:hlo",
        "@mlir-hlo//:legalize_control_flow",
        "@mlir-hlo//:legalize_gather_to_torch_index_select",
        "@mlir-hlo//:materialize_broadcasts",
        "@mlir-hlo//:mhlo_to_mhlo_lowering_patterns",
        "@mlir-hlo//:unfuse_batch_norm",
    ],
)

# TODO(#4919): For an unknown reason, GCC's devirtualization optimization wreaks
# havoc on this file. Needs to be further root caused. Seems to affect both 9.x
# and 10.x.
iree_cmake_extra_content(
    content = """
set_property(SOURCE
  DispatchLinalgOnTensors.cpp
  PROPERTY COMPILE_FLAGS $<$<CXX_COMPILER_ID:GNU>:-fno-devirtualize>)
""",
    inline = True,
)
