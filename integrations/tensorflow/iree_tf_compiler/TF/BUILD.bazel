# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "TF",
    srcs = [
        "ConvertToMHLO.cpp",
        "DirectLoweringPatterns.cpp",
        "LowerGlobalTensors.cpp",
        "Passes.cpp",
        "PrettifyDebugInfo.cpp",
        "PropagateResourceCasts.cpp",
        "SavedModelToIreeABI.cpp",
        "StripAsserts.cpp",
        "StripMetadata.cpp",
        "VerifyFullyConverted.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    defines = [
        "IREE_COMPILER_TENSORFLOW_ENABLED",
    ],
    deps = [
        "//iree-dialects:IREEInputDialect",
        "//iree_tf_compiler/MHLO",
        "//iree_tf_compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "@org_tensorflow//tensorflow/compiler/mlir/tf2xla:xla_legalize_tf",
        "@org_tensorflow//tensorflow/compiler/mlir/tosa:tf_passes",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:chlo_legalize_to_hlo",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:chlo_ops",
    ],
)
