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

cc_test(
    name = "mmt4d_benchmark",
    srcs = ["mmt4d_benchmark.c"],
    deps = [
        "//iree/base",
        "//iree/base/internal:flags",
        "//iree/builtins/libmmt4d",
        "//iree/testing:benchmark",
    ],
)

cc_test(
    name = "mmt4d_test",
    srcs = ["mmt4d_test.cc"],
    deps = [
        "//iree/base",
        "//iree/base/internal:flags",
        "//iree/builtins/libmmt4d",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
