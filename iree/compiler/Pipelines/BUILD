# Copyright 2022 The IREE Authors
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
    name = "Options",
    srcs = ["Options.cpp"],
    hdrs = ["Options.h"],
    deps = [
        "//iree/compiler/Utils",
    ],
)

cc_library(
    name = "Pipelines",
    srcs = [
        "Pipelines.cpp",
    ],
    hdrs = [
        "Pipelines.h",
    ],
    deps = [
        ":Options",
        "//iree/compiler/Bindings/Native/Transforms",
        "//iree/compiler/Bindings/TFLite/Transforms",
        "//iree/compiler/Dialect/Flow/Transforms",
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "//iree/compiler/Dialect/HAL/Transforms",
        "//iree/compiler/Dialect/Stream/Transforms",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/Conversion/StandardToVM",
        "//iree/compiler/Dialect/VM/Target/Bytecode",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/InputConversion/Common",
        "//iree/compiler/InputConversion/MHLO",
        "//iree/compiler/InputConversion/TOSA",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
