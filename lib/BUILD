# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "libIREECompiler.so",
    srcs = [
        "//compiler/src/iree/compiler/API2:api_exports.c",
    ],
    linkopts = [
        "-Wl,--version-script=$(location //compiler/src/iree/compiler/API2:api_exports.ld)",
        "-Wl,--no-undefined",
    ],
    linkshared = 1,
    deps = [
        "//compiler/bindings/c:headers",
        "//compiler/src/iree/compiler/API2:MLIRInteropHeaders",
        "//compiler/src/iree/compiler/API2:StaticImpl",
        "//compiler/src/iree/compiler/API2:api_exports.ld",
    ],
)
