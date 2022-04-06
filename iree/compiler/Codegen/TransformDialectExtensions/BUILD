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
    name = "TransformDialectExtensions",
    srcs = [
        "TransformDialectExtensions.cpp",
    ],
    hdrs = [
        "TransformDialectExtensions.h",
    ],
    deps = [
        "//llvm-external-projects/iree-dialects:IREEDialectsTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgExtDialect",
        "//llvm-external-projects/iree-dialects:IREELinalgExtTransforms",
        "//llvm-external-projects/iree-dialects:IREELinalgTransformDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:PDLDialect",
    ],
)
