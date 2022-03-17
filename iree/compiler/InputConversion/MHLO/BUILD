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
        "Rewriters.h",
    ],
    deps = [
        ":PassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "MHLO",
    srcs = [
        "BroadcastingToLinalgPatterns.cpp",
        "ConvertComplexToReal.cpp",
        "ConvertMHLOToFlow.cpp",
        "ConvertMHLOToFlow.h",
        "ConvertMHLOToLinalgExt.cpp",
        "FlattenTuplesInCFG.cpp",
        "LegalizeInputTypes.cpp",
        "MHLOToLinalgOnTensors.cpp",
        "MHLOToMHLOPreprocessing.cpp",
        "Passes.cpp",
        "VerifyCompilerMHLOInputLegality.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":PassHeaders",
        ":PassesIncGen",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/InputConversion/Common",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtPasses",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Affine",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@mlir-hlo//:chlo_legalize_to_hlo",
        "@mlir-hlo//:hlo",
        "@mlir-hlo//:hlo_legalize_shape_ops_to_standard",
        "@mlir-hlo//:hlo_legalize_to_arithmetic",
        "@mlir-hlo//:hlo_legalize_to_lhlo",
        "@mlir-hlo//:hlo_legalize_to_memref",
        "@mlir-hlo//:legalize_control_flow",
        "@mlir-hlo//:legalize_einsum_to_dot_general",
        "@mlir-hlo//:legalize_gather_to_torch_index_select",
        "@mlir-hlo//:legalize_to_linalg",
        "@mlir-hlo//:legalize_to_standard",
        "@mlir-hlo//:map_lmhlo_to_scalar_op",
        "@mlir-hlo//:map_mhlo_to_scalar_op",
        "@mlir-hlo//:materialize_broadcasts",
        "@mlir-hlo//:mhlo_to_mhlo_lowering_patterns",
        "@mlir-hlo//:unfuse_batch_norm",
    ],
)
