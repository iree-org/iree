# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "string_list_flag")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

ALL_EXECUTABLE_PLUGINS = [
    "embedded-elf",
    "system-library",
]

string_list_flag(
    name = "enabled_executable_plugins",
    build_setting_default = [
        "embedded-elf",
        "system-library",
    ],
)

[
    config_setting(
        name = "{}_enabled".format(plugin),
        flag_values = {
            ":enabled_executable_plugins": plugin,
        },
    )
    for plugin in ALL_EXECUTABLE_PLUGINS
]

iree_runtime_cc_library(
    name = "registration",
    srcs = ["init.c"],
    hdrs = ["init.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local:executable_plugin_manager",
    ] + select({
        ":embedded-elf_enabled": ["//runtime/src/iree/hal/local/plugins:embedded_elf_plugin"],
        "//conditions:default": [],
    }) + select({
        ":system-library_enabled": ["//runtime/src/iree/hal/local/plugins:system_library_plugin"],
        "//conditions:default": [],
    }),
)
