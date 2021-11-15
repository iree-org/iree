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
        "ConvertToHAL2.cpp",
        "ElideRedundantCommands.cpp",
        "InlineDeviceSwitches.cpp",
        "LinkExecutables.cpp",
        "MaterializeInterfaces2.cpp",
        "MaterializeResourceCaches.cpp",
        "MemoizeDeviceQueries.cpp",
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
        "//iree/compiler/Dialect/HAL/Conversion2/StandardToHAL",
        "//iree/compiler/Dialect/HAL/Conversion2/StreamToHAL",
        "//iree/compiler/Dialect/HAL/Conversion2/UtilToHAL",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
