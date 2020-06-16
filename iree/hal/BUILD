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

# HAL (Hardware Abstraction Layer).
# Subdirectories contain implementations for different hardware and
# software backends.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "allocator",
    srcs = ["allocator.cc"],
    hdrs = ["allocator.h"],
    deps = [
        ":buffer",
        "//iree/base:ref_ptr",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = [
        "api.h",
        "api_detail.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":api_hdrs",
        ":buffer",
        ":command_buffer",
        ":device",
        ":driver",
        ":driver_registry",
        ":heap_buffer",
        ":semaphore",
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:memory",
        "//iree/base:ref_ptr",
        "//iree/base:tracing",
        "//iree/hal/host:host_local_allocator",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
    deps = [
        "//iree/base:api_hdrs",
    ],
)

cc_test(
    name = "api_string_util_test",
    srcs = ["api_string_util_test.cc"],
    deps = [
        ":api",
        "//iree/base:api_util",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":resource",
        "//iree/base:bitfield",
        "//iree/base:logging",
        "//iree/base:source_location",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = [
        "buffer_mapping_test.cc",
        "buffer_test.cc",
    ],
    deps = [
        ":buffer",
        ":heap_buffer",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "command_buffer",
    srcs = ["command_buffer.cc"],
    hdrs = ["command_buffer.h"],
    deps = [
        ":buffer",
        ":descriptor_set",
        ":event",
        ":executable",
        ":executable_layout",
        ":resource",
        "//iree/base:bitfield",
        "//iree/base:status",
    ],
)

cc_library(
    name = "command_buffer_validation",
    srcs = ["command_buffer_validation.cc"],
    hdrs = ["command_buffer_validation.h"],
    deps = [
        ":allocator",
        ":command_buffer",
        "//iree/base:logging",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "command_queue",
    hdrs = ["command_queue.h"],
    deps = [
        ":command_buffer",
        ":semaphore",
        "//iree/base:bitfield",
        "//iree/base:status",
        "//iree/base:time",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "debug_capture_manager",
    hdrs = ["debug_capture_manager.h"],
    deps = [
        "//iree/base:status",
    ],
)

cc_library(
    name = "deferred_buffer",
    srcs = ["deferred_buffer.cc"],
    hdrs = ["deferred_buffer.h"],
    deps = [
        ":allocator",
        ":buffer",
        "//iree/base:status",
    ],
)

cc_test(
    name = "deferred_buffer_test",
    srcs = ["deferred_buffer_test.cc"],
    deps = [
        ":deferred_buffer",
        ":heap_buffer",
        "//iree/base:status_matchers",
        "//iree/hal/testing:mock_allocator",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "descriptor_set",
    hdrs = ["descriptor_set.h"],
    deps = [
        ":buffer",
        ":resource",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "descriptor_set_layout",
    hdrs = ["descriptor_set_layout.h"],
    deps = [
        ":buffer",
        ":resource",
    ],
)

cc_library(
    name = "device",
    hdrs = ["device.h"],
    deps = [
        ":allocator",
        ":buffer",
        ":command_queue",
        ":descriptor_set",
        ":descriptor_set_layout",
        ":device_info",
        ":event",
        ":executable_cache",
        ":executable_layout",
        ":semaphore",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:time",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "device_info",
    hdrs = ["device_info.h"],
    deps = [
        "//iree/base:bitfield",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_manager",
    srcs = ["device_manager.cc"],
    hdrs = ["device_manager.h"],
    deps = [
        ":allocator",
        ":buffer",
        ":command_queue",
        ":device",
        ":device_placement",
        ":executable_format",
        ":heap_buffer",
        ":semaphore",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:time",
        "//iree/base:tracing",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "device_placement",
    hdrs = ["device_placement.h"],
)

cc_library(
    name = "driver",
    hdrs = ["driver.h"],
    deps = [
        ":debug_capture_manager",
        ":device",
        ":device_info",
        "//iree/base:ref_ptr",
        "//iree/base:status",
    ],
)

cc_library(
    name = "driver_registry",
    srcs = ["driver_registry.cc"],
    hdrs = ["driver_registry.h"],
    deps = [
        ":driver",
        "//iree/base:initializer",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "event",
    hdrs = ["event.h"],
    deps = [
        ":resource",
    ],
)

cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    deps = [":resource"],
)

cc_library(
    name = "executable_cache",
    srcs = ["executable_cache.cc"],
    hdrs = ["executable_cache.h"],
    deps = [
        ":executable",
        ":executable_format",
        ":executable_layout",
        ":executable_spec",
        "//iree/base:bitfield",
        "//iree/base:ref_ptr",
        "//iree/base:status",
    ],
)

cc_library(
    name = "executable_format",
    hdrs = ["executable_format.h"],
)

cc_library(
    name = "executable_layout",
    hdrs = ["executable_layout.h"],
    deps = [":resource"],
)

cc_library(
    name = "executable_spec",
    hdrs = ["executable_spec.h"],
    deps = [
        ":executable_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "heap_buffer",
    srcs = ["heap_buffer.cc"],
    hdrs = ["heap_buffer.h"],
    deps = [
        ":allocator",
        ":buffer",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal/host:host_buffer",
    ],
)

cc_library(
    name = "resource",
    hdrs = ["resource.h"],
    deps = [
        "//iree/base:ref_ptr",
    ],
)

cc_library(
    name = "semaphore",
    hdrs = ["semaphore.h"],
    deps = [
        ":resource",
        "//iree/base:status",
        "//iree/base:time",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "stack_trace",
    hdrs = ["stack_trace.h"],
)
