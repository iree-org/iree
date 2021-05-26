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
    name = "cts_test_base",
    testonly = True,
    hdrs = ["cts_test_base.h"],
    deps = [
        "//iree/base",
        "//iree/hal",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "buffer_mapping_test",
    srcs = ["buffer_mapping_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "command_buffer_test",
    srcs = ["command_buffer_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "descriptor_set_test",
    srcs = ["descriptor_set_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "descriptor_set_layout_test",
    srcs = ["descriptor_set_layout_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "driver_test",
    srcs = ["driver_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "event_test",
    srcs = ["event_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "executable_layout_test",
    srcs = ["executable_layout_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "semaphore_submission_test",
    srcs = ["semaphore_submission_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/hal/testing:driver_registry",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
