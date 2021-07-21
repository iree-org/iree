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
    name = "check",
    testonly = True,
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base/internal",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/vm",
        "//iree/vm:cc",
    ],
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":check",
        "//iree/base",
        "//iree/base:cc",
        "//iree/base/internal",
        "//iree/base/internal:span",
        "//iree/hal",
        "//iree/hal/vmvx/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
    ],
)
