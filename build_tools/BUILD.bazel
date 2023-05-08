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

# Helper for getting linkopts into libraries in a way we can easily spot and
# rewrite from bazel-to-cmake and copybara.
cc_library(
    name = "default_linkopts",
    linkopts = select({
        "//build_tools/bazel:iree_is_msvc": [],
        "//build_tools/bazel:iree_is_android": [
            # Android provides its own pthreads support with no linking required.
        ],
        "//conditions:default": [
            # Just include libraries that should be presumed in 2020.
            "-lpthread",
        ],
    }),
)

cc_library(
    name = "dl",
    linkopts = select({
        "//build_tools/bazel:iree_is_msvc": [],
        "//build_tools/bazel:iree_is_android": [
            "-ldl",
        ],
        "//conditions:default": [
            "-ldl",
        ],
    }),
)
