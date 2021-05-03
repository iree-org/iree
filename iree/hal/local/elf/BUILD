# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
