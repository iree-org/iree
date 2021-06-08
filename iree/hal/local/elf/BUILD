# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

cc_test(
    name = "elf_module_test",
    srcs = ["elf_module_test.cc"],
    deps = [
        ":elf_module",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/hal/local:executable_library",
        "//iree/hal/local/elf/testdata:simple_mul_dispatch",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
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
