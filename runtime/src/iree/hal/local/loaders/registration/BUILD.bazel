# Copyright 2022 The IREE Authors
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

ALL_EXECUTABLE_LOADERS = [
    "embedded-elf",
    "system-library",
    "vmvx-module",
]

string_list_flag(
    name = "enabled_executable_loaders",
    build_setting_default = [
        "embedded-elf",
        "system-library",
        "vmvx-module",
    ],
)

[
    config_setting(
        name = "{}_enabled".format(loader),
        flag_values = {
            ":enabled_executable_loaders": loader,
        },
    )
    for loader in ALL_EXECUTABLE_LOADERS
]

iree_runtime_cc_library(
    name = "registration",
    srcs = ["init.c"],
    hdrs = ["init.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
    ] + select({
        ":embedded-elf_enabled": ["//runtime/src/iree/hal/local/loaders:embedded_elf_loader"],
        "//conditions:default": [],
    }) + select({
        ":system-library_enabled": ["//runtime/src/iree/hal/local/loaders:system_library_loader"],
        "//conditions:default": [],
    }) + select({
        ":vmvx-module_enabled": ["//runtime/src/iree/hal/local/loaders:vmvx_module_loader"],
        "//conditions:default": [],
    }),
)
