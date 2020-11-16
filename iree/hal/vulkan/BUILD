# Copyright 2019 Google LLC
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

# HAL implementation using Vulkan and (likely) SPIR-V executables.

load(
    "//iree:build_defs.oss.bzl",
    "iree_cmake_extra_content",
    "PLATFORM_VULKAN_TEST_DEPS",
)

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
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        ":vulkan",
        "//iree/base:api",
        "//iree/base:tracing",
        "//iree/hal:api",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "debug_reporter.cc",
        "dynamic_symbols.cc",
        "extensibility_util.cc",
        "renderdoc_capture_manager.cc",
        "status_util.cc",
        "timepoint_util.cc",
    ],
    hdrs = [
        "debug_reporter.h",
        "dynamic_symbol_tables.h",
        "dynamic_symbols.h",
        "extensibility_util.h",
        "handle_util.h",
        "renderdoc_capture_manager.h",
        "status_util.h",
        "timepoint_util.h",
        "vulkan_headers.h",
    ],
    deps = [
        "//iree/base:dynamic_library",
        "//iree/base:intrusive_list",
        "//iree/base:logging",
        "//iree/base:memory",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:time",
        "//iree/base:tracing",
        "//iree/hal",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
        "@iree_vulkan_headers//:vulkan_headers",
        "@renderdoc_api//:renderdoc_app",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    tags = ["driver=vulkan"],
    deps = [
        ":utils",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ] + PLATFORM_VULKAN_TEST_DEPS,
)

cc_library(
    name = "vma_allocator",
    srcs = [
        "internal_vk_mem_alloc.cc",
        "internal_vk_mem_alloc.h",
        "vma_allocator.cc",
        "vma_buffer.cc",
    ],
    hdrs = [
        "vma_allocator.h",
        "vma_buffer.h",
    ],
    deps = [
        ":utils",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:tracing",
        "//iree/hal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@vulkan_memory_allocator//:impl_header_only",
    ],
)

cc_library(
    name = "vulkan",
    srcs = [
        "descriptor_pool_cache.cc",
        "descriptor_set_arena.cc",
        "direct_command_buffer.cc",
        "direct_command_queue.cc",
        "emulated_timeline_semaphore.cc",
        "native_descriptor_set.cc",
        "native_event.cc",
        "native_timeline_semaphore.cc",
        "pipeline_cache.cc",
        "pipeline_executable.cc",
        "pipeline_executable_layout.cc",
        "serializing_command_queue.cc",
        "vulkan_device.cc",
        "vulkan_driver.cc",
    ],
    hdrs = [
        "descriptor_pool_cache.h",
        "descriptor_set_arena.h",
        "direct_command_buffer.h",
        "direct_command_queue.h",
        "emulated_timeline_semaphore.h",
        "native_descriptor_set.h",
        "native_event.h",
        "native_timeline_semaphore.h",
        "pipeline_cache.h",
        "pipeline_executable.h",
        "pipeline_executable_layout.h",
        "serializing_command_queue.h",
        "vulkan_device.h",
        "vulkan_driver.h",
    ],
    deps = [
        ":utils",
        ":vma_allocator",
        "//iree/base:alignment",
        "//iree/base:api",
        "//iree/base:arena",
        "//iree/base:intrusive_list",
        "//iree/base:math",
        "//iree/base:memory",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:time",
        "//iree/base:tracing",
        "//iree/hal",
        "//iree/hal:command_buffer_validation",
        "//iree/schemas:spirv_executable_def_c_fbs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/utility",
    ],
)

cc_library(
    name = "vulkan_driver_module",
    srcs = ["vulkan_driver_module.cc"],
    deps = [
        ":utils",
        ":vulkan",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:driver_registry",
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = 1,
)
