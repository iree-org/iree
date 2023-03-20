# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_binary", "iree_runtime_cc_library")
load("//build_tools/bazel:native_binary.bzl", "native_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
# Disable on platforms/architectures where ELF is not supported.
if(EMSCRIPTEN)
  return()
endif()
""",
)

#===------------------------------------------------------------------------===#
# Runtime ELF module loader/linker
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "elf_module",
    srcs = [
        "elf_module.c",
    ],
    hdrs = [
        "elf_module.h",
        "elf_types.h",
    ],
    deps = [
        ":arch",
        ":platform",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
    ],
)

iree_runtime_cc_binary(
    name = "elf_module_test_binary",
    srcs = ["elf_module_test_main.c"],
    deps = [
        ":elf_module",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/hal/local:executable_environment",
        "//runtime/src/iree/hal/local:executable_library",
        "//runtime/src/iree/hal/local/elf/testdata:elementwise_mul",
    ],
)

native_test(
    name = "elf_module_test",
    src = ":elf_module_test_binary",
)

#===------------------------------------------------------------------------===#
# Architecture and platform support
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "arch",
    srcs = [
        "arch/arm_32.c",
        "arch/arm_64.c",
        "arch/riscv.c",
        "arch/x86_32.c",
        "arch/x86_64.c",
        "elf_types.h",
    ],
    hdrs = [
        "arch.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
    ],
)

iree_runtime_cc_library(
    name = "platform",
    srcs = [
        "platform/apple.c",
        "platform/generic.c",
        "platform/linux.c",
        "platform/windows.c",
    ],
    hdrs = [
        "platform.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
    ],
)
