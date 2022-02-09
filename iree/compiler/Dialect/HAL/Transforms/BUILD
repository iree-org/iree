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
    name = "Transforms",
    srcs = [
        "AssignTargetDevices.cpp",
        "BenchmarkBatchDispatches.cpp",
        "ConvertToHAL.cpp",
        "ElideRedundantCommands.cpp",
        "InlineDeviceSwitches.cpp",
        "LinkExecutables.cpp",
        "MaterializeInterfaces.cpp",
        "MaterializeResourceCaches.cpp",
        "MemoizeDeviceQueries.cpp",
        "PackDispatchOperands.cpp",
        "Passes.cpp",
        "ResolveEntryPointOrdinals.cpp",
        "SerializeExecutables.cpp",
        "TranslateExecutables.cpp",
        "VerifyTargetEnvironment.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/Analysis",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/HAL/Conversion/StandardToHAL",
        "//iree/compiler/Dialect/HAL/Conversion/StreamToHAL",
        "//iree/compiler/Dialect/HAL/Conversion/UtilToHAL",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/IR:HALDialect",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/HAL/Utils",
        "//iree/compiler/Dialect/Stream/IR",
        "//iree/compiler/Dialect/Stream/Transforms",
        "//iree/compiler/Dialect/Util/Conversion",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/Util/Transforms",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:ControlFlowOps",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
