# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
    name = "Transforms",
    srcs = [
        "AssignTargetDevices.cpp",
        "BenchmarkBatchDispatches.cpp",
        "ConvertToHAL.cpp",
        "DumpExecutableBenchmarks.cpp",
        "DumpExecutableSources.cpp",
        "ElideRedundantCommands.cpp",
        "FixupLegacySync.cpp",
        "InlineDeviceSwitches.cpp",
        "LinkExecutables.cpp",
        "MaterializeInterfaces.cpp",
        "MaterializeResourceCaches.cpp",
        "MemoizeDeviceQueries.cpp",
        "Passes.cpp",
        "ResolveExportOrdinals.cpp",
        "SerializeExecutables.cpp",
        "SubstituteExecutables.cpp",
        "TranslateExecutables.cpp",
        "VerifyTargetEnvironment.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//compiler/src/iree/compiler/Dialect/Flow/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/Analysis",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion/HALToHAL",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion/StandardToHAL",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion/StreamToHAL",
        "//compiler/src/iree/compiler/Dialect/HAL/Conversion/UtilToHAL",
        "//compiler/src/iree/compiler/Dialect/HAL/IR",
        "//compiler/src/iree/compiler/Dialect/HAL/IR:HALDialect",
        "//compiler/src/iree/compiler/Dialect/HAL/Target",
        "//compiler/src/iree/compiler/Dialect/HAL/Utils",
        "//compiler/src/iree/compiler/Dialect/Stream/IR",
        "//compiler/src/iree/compiler/Dialect/Stream/Transforms",
        "//compiler/src/iree/compiler/Dialect/Util/Conversion",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/Util/Transforms",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
