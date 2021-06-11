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
    name = "CallingConventionUtils",
    srcs = ["CallingConventionUtils.cpp"],
    hdrs = ["CallingConventionUtils.h"],
    deps = [
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ConstantEncodingUtils",
    srcs = ["ConstantEncodingUtils.cpp"],
    hdrs = ["ConstantEncodingUtils.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "init_targets",
    hdrs = ["init_targets.h"],
    deps = [
        "//iree/compiler/Dialect/VM/Target/Bytecode",
    ],
)
