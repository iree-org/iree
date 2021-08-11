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
    name = "HALToVM",
    srcs = [
        "ConvertAllocatorOps.cpp",
        "ConvertBufferOps.cpp",
        "ConvertBufferViewOps.cpp",
        "ConvertCommandBufferOps.cpp",
        "ConvertConstantOps.cpp",
        "ConvertControlFlowOps.cpp",
        "ConvertDeviceOps.cpp",
        "ConvertExecutableOps.cpp",
        "ConvertExperimentalOps.cpp",
        "ConvertHALToVM.cpp",
        "ConvertSemaphoreOps.cpp",
        "ConvertVariableOps.cpp",
    ],
    hdrs = [
        "ConvertHALToVM.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL:hal_imports",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/Utils",
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/Conversion/UtilToVM",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)
