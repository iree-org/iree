# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)
""",
    inline = True,
)

iree_runtime_cc_library(
    name = "embedded_elf_plugin",
    srcs = ["embedded_elf_plugin.c"],
    hdrs = ["embedded_elf_plugin.h"],
    defines = [
        "IREE_HAVE_HAL_EXECUTABLE_EMBEDDED_ELF_PLUGIN=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:file_io",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local:executable_plugin_manager",
        "//runtime/src/iree/hal/local/elf:elf_module",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)

iree_runtime_cc_library(
    name = "static_plugin",
    srcs = ["static_plugin.c"],
    hdrs = ["static_plugin.h"],
    defines = [
        "IREE_HAVE_HAL_EXECUTABLE_STATIC_PLUGIN=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local:executable_plugin_manager",
    ],
)

iree_cmake_extra_content(
    content = """
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
""",
    inline = True,
)

iree_runtime_cc_library(
    name = "system_library_plugin",
    srcs = ["system_library_plugin.c"],
    hdrs = ["system_library_plugin.h"],
    defines = [
        "IREE_HAVE_HAL_EXECUTABLE_SYSTEM_LIBRARY_PLUGIN=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:dynamic_library",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local:executable_plugin_manager",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
