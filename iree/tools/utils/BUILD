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
    name = "trace_replay",
    srcs = ["trace_replay.c"],
    hdrs = ["trace_replay.h"],
    deps = [
        ":yaml_util",
        "//iree/base",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:file_path",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "@com_github_yaml_libyaml//:yaml",
    ],
)

# TODO(benvanik): fold these into iree/runtime and use that instead.
cc_library(
    name = "vm_util",
    srcs = ["vm_util.cc"],
    hdrs = ["vm_util.h"],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:logging",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:span",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
    ],
)

cc_test(
    name = "vm_util_test",
    srcs = ["vm_util_test.cc"],
    deps = [
        ":vm_util",
        "//iree/base",
        "//iree/hal",
        "//iree/hal/vmvx/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:cc",
    ],
)

cc_library(
    name = "yaml_util",
    srcs = ["yaml_util.c"],
    hdrs = ["yaml_util.h"],
    deps = [
        "//iree/base",
        "@com_github_yaml_libyaml//:yaml",
    ],
)
