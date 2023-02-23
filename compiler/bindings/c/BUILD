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
    name = "headers",
    hdrs = [
        "iree/compiler/api_support.h",
        "iree/compiler/embedding_api.h",
        "iree/compiler/loader.h",
        "iree/compiler/tool_entry_points_api.h",
    ],
    includes = ["."],
)

cc_library(
    name = "loader",
    srcs = [
        "iree/compiler/loader/handle_symbols.inc",
        "iree/compiler/loader/loader.cpp",
    ],
    deps = [
        ":headers",
        "//build_tools:dl",
    ],
)

# TODO: Support loader_test in Bazel.
