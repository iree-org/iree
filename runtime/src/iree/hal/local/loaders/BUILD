# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Default implementations for HAL types that use the host resources.
# These are generally just wrappers around host heap memory and host threads.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "embedded_library_loader",
    srcs = ["embedded_library_loader.c"],
    hdrs = ["embedded_library_loader.h"],
    defines = [
        "IREE_HAL_HAVE_EMBEDDED_LIBRARY_LOADER=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:executable_library",
        "//runtime/src/iree/hal/local/elf:elf_module",
    ],
)

iree_runtime_cc_library(
    name = "static_library_loader",
    srcs = ["static_library_loader.c"],
    hdrs = ["static_library_loader.h"],
    defines = [
        "IREE_HAL_HAVE_STATIC_LIBRARY_LOADER=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:executable_environment",
        "//runtime/src/iree/hal/local:executable_library",
    ],
)

iree_runtime_cc_library(
    name = "system_library_loader",
    srcs = ["system_library_loader.c"],
    hdrs = ["system_library_loader.h"],
    defines = [
        "IREE_HAL_HAVE_SYSTEM_LIBRARY_LOADER=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:dynamic_library",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:executable_library",
    ],
)

iree_cmake_extra_content(
    content = """
if(IREE_HAL_DRIVER_VMVX OR IREE_HAL_DRIVER_VMVX_SYNC)
""",
    inline = True,
)

iree_runtime_cc_library(
    name = "vmvx_module_loader",
    srcs = ["vmvx_module_loader.c"],
    hdrs = ["vmvx_module_loader.h"],
    defines = [
        "IREE_HAL_HAVE_VMVX_MODULE_LOADER=1",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:executable_library",
        "//runtime/src/iree/modules/vmvx",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
