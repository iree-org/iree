# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# HAL implementation using Vulkan and (likely) SPIR-V executables.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_VULKAN})
  return()
endif()
""",
)

cc_library(
    name = "vulkan",
    srcs = [
        "api.cc",
        "builtin_executables.cc",
        "builtin_executables.h",
        "command_queue.h",
        "debug_reporter.cc",
        "debug_reporter.h",
        "descriptor_pool_cache.cc",
        "descriptor_pool_cache.h",
        "descriptor_set_arena.cc",
        "descriptor_set_arena.h",
        "direct_command_buffer.cc",
        "direct_command_buffer.h",
        "direct_command_queue.cc",
        "direct_command_queue.h",
        "emulated_semaphore.cc",
        "emulated_semaphore.h",
        "extensibility_util.cc",
        "extensibility_util.h",
        "handle_util.h",
        "internal_vk_mem_alloc.cc",
        "internal_vk_mem_alloc.h",
        "native_descriptor_set.cc",
        "native_descriptor_set.h",
        "native_descriptor_set_layout.cc",
        "native_descriptor_set_layout.h",
        "native_event.cc",
        "native_event.h",
        "native_executable.cc",
        "native_executable.h",
        "native_executable_layout.cc",
        "native_executable_layout.h",
        "native_semaphore.cc",
        "native_semaphore.h",
        "nop_executable_cache.cc",
        "nop_executable_cache.h",
        "serializing_command_queue.cc",
        "serializing_command_queue.h",
        "status_util.c",
        "status_util.h",
        "timepoint_util.cc",
        "timepoint_util.h",
        "tracing.cc",
        "tracing.h",
        "vma_allocator.cc",
        "vma_allocator.h",
        "vma_buffer.cc",
        "vma_buffer.h",
        "vulkan_device.cc",
        "vulkan_driver.cc",
        "vulkan_headers.h",
    ],
    hdrs = [
        # TODO(benvanik): hide all but api.h.
        "api.h",
        "vulkan_device.h",
        "vulkan_driver.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_symbols",
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:core_headers",
        "//iree/base:logging",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:synchronization",
        "//iree/base/internal/flatcc:parsing",
        "//iree/hal",
        "//iree/hal/utils:buffer_transfer",
        "//iree/hal/vulkan/builtin",
        "//iree/hal/vulkan/util:arena",
        "//iree/hal/vulkan/util:intrusive_list",
        "//iree/hal/vulkan/util:ref_ptr",
        "//iree/schemas:spirv_executable_def_c_fbs",
        "@vulkan_headers",
        "@vulkan_memory_allocator//:impl_header_only",
    ],
)

cc_library(
    name = "dynamic_symbols",
    srcs = [
        "dynamic_symbols.cc",
        "vulkan_headers.h",
    ],
    hdrs = [
        "dynamic_symbols.h",
    ],
    textual_hdrs = [
        "dynamic_symbol_tables.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:cc",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal:dynamic_library",
        "//iree/hal/vulkan/util:ref_ptr",
        "@vulkan_headers",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    tags = ["driver=vulkan"],
    deps = [
        ":dynamic_symbols",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
