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
    name = "Utils",
    srcs = [
        "InferCustomKernelsTargetInfoFromParent.cpp",
    ],
    hdrs = [
        "DeviceSwitchBuilder.h",
        "InferCustomKernelsTargetInfoFromParent.h",
    ],
    deps = [
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)
