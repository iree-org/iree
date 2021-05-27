# Copyright 2020 The IREE Authors
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
    name = "XlaHloShapeBuilder",
    srcs = [
        "XlaHloShapeBuilder.cpp",
    ],
    hdrs = [
        "XlaHloShapeBuilder.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@mlir-hlo//:hlo",
    ],
)
