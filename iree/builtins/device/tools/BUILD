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
    name = "libdevice_benchmark",
    srcs = ["libdevice_benchmark.c"],
    deps = [
        "//iree/base",
        "//iree/base/internal:flags",
        "//iree/builtins/device",
        "//iree/testing:benchmark",
    ],
)

cc_test(
    name = "libdevice_test",
    srcs = ["libdevice_test.cc"],
    deps = [
        "//iree/base",
        "//iree/base/internal:flags",
        "//iree/builtins/device",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
