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
    name = "Utils",
    srcs = [
        "Signature.cpp",
        "SignatureBuilder.cpp",
        "SignatureParser.cpp",
    ],
    hdrs = [
        "Signature.h",
        "SignatureBuilder.h",
        "SignatureParser.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_test(
    name = "SignatureTest",
    srcs = ["SignatureTest.cpp"],
    deps = [
        ":Utils",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
