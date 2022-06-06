# Copyright 2020 The IREE Authors
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

ALL_DRIVERS = [
    "dylib",
    "dylib-sync",
    "vmvx",
    "vmvx-sync",
    "vulkan",
    "cuda",
]

string_list_flag(
    name = "enabled_drivers",
    build_setting_default = [
        "dylib",
        "dylib-sync",
        "vmvx",
        "vmvx-sync",
        "vulkan",
    ],
)

[
    config_setting(
        name = "{}_enabled".format(driver),
        flag_values = {
            ":enabled_drivers": driver,
        },
    )
    for driver in ALL_DRIVERS
]

iree_runtime_cc_library(
    name = "drivers",
    srcs = ["init.c"],
    hdrs = ["init.h"],
    deps = [
               "//runtime/src/iree/base",
               "//runtime/src/iree/base:tracing",
           ] + select({
               ":dylib_enabled": ["//runtime/src/iree/hal/dylib/registration"],
               "//conditions:default": [],
           }) +
           select({
               ":dylib-sync_enabled": ["//runtime/src/iree/hal/dylib/registration:sync"],
               "//conditions:default": [],
           }) +
           select({
               ":vmvx_enabled": ["//runtime/src/iree/hal/vmvx/registration"],
               "//conditions:default": [],
           }) +
           select({
               ":vmvx-sync_enabled": ["//runtime/src/iree/hal/vmvx/registration:sync"],
               "//conditions:default": [],
           }) +
           select({
               ":vulkan_enabled": ["//runtime/src/iree/hal/drivers/vulkan/registration"],
               "//conditions:default": [],
           }) +
           select({
               ":cuda_enabled": ["//runtime/src/iree/hal/drivers/cuda/registration"],
               "//conditions:default": [],
           }),
)
