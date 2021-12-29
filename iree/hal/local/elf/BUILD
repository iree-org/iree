# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Runtime ELF module loader/linker
#===------------------------------------------------------------------------===#

cc_library(
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
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_binary(
    name = "elf_module_test_binary",
    srcs = ["elf_module_test_main.c"],
    deps = [
        ":elf_module",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/hal/local:executable_library",
        "//iree/hal/local/elf/testdata:elementwise_mul",
    ],
)

run_binary_test(
    name = "elf_module_test",
    test_binary = ":elf_module_test_binary",
)

#===------------------------------------------------------------------------===#
# Architecture and platform support
#===------------------------------------------------------------------------===#

cc_library(
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
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_library(
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
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)
