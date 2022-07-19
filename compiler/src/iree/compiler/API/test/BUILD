# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:native_binary.bzl", "native_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "api-test-binary",
    srcs = ["api-test-main.c"],
    deps = [
        "//compiler/src/iree/compiler/API:CAPI",
        "//runtime/src/iree/base",
        "@llvm-project//mlir:CAPIIR",
    ],
    testonly = True,
)

native_test(
    name = "api-test",
    src = ":api-test-binary",
)
