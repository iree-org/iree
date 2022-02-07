# Copyright 2021 The IREE Authors
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
    name = "Builtins",
    srcs = [
        "Device.cpp",
        "Musl.cpp",
    ],
    hdrs = [
        "Device.h",
        "Musl.h",
    ],
    deps = [
        "//iree/builtins/device/bin:libdevice",
        "//iree/builtins/musl/bin:libmusl",
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:Support",
    ],
)
