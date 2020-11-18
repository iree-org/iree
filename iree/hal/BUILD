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
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
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
        ":driver_registry",
        ":hal",
        ":heap_buffer",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:ref_ptr",
        "//iree/base:tracing",
        "//iree/hal/host:host_local_allocator",
        "@com_google_absl//absl/container:inlined_vector",
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
        "//iree/base:core_headers",
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "command_buffer_validation",
    srcs = ["command_buffer_validation.cc"],
    hdrs = ["command_buffer_validation.h"],
    deps = [
        ":hal",
        "//iree/base:logging",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_manager",
    srcs = ["device_manager.cc"],
    hdrs = ["device_manager.h"],
    deps = [
        ":hal",
        ":heap_buffer",
        "//iree/base:status",
        "//iree/base:time",
        "//iree/base:tracing",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "driver_registry",
    srcs = ["driver_registry.cc"],
    hdrs = ["driver_registry.h"],
    deps = [
        ":hal",
        "//iree/base:initializer",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "hal",
    srcs = [
        "allocator.cc",
        "buffer.cc",
        "command_buffer.cc",
        "deferred_buffer.cc",
        "executable_cache.cc",
    ],
    hdrs = [
        "allocator.h",
        "buffer.h",
        "command_buffer.h",
        "command_queue.h",
        "debug_capture_manager.h",
        "deferred_buffer.h",
        "descriptor_set.h",
        "descriptor_set_layout.h",
        "device.h",
        "device_info.h",
        "device_placement.h",
        "driver.h",
        "event.h",
        "executable.h",
        "executable_cache.h",
        "executable_format.h",
        "executable_layout.h",
        "executable_spec.h",
        "resource.h",
        "semaphore.h",
        "stack_trace.h",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:logging",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:time",
        "//iree/base:tracing",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = [
        "buffer_mapping_test.cc",
        "buffer_test.cc",
    ],
    deps = [
        ":hal",
        ":heap_buffer",
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "deferred_buffer_test",
    srcs = ["deferred_buffer_test.cc"],
    deps = [
        ":hal",
        ":heap_buffer",
        "//iree/hal/testing:mock_allocator",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "heap_buffer",
    srcs = ["heap_buffer.cc"],
    hdrs = ["heap_buffer.h"],
    deps = [
        ":hal",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal/host:host_buffer",
    ],
)
