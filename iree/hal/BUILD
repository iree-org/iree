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

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#
# TODO(benvanik): rename to :hal

cc_library(
    name = "api",
    srcs = [
        "api.c",
        "api.cc",
    ],
    hdrs = [
        "api.h",
        "api_detail.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":hal",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:ref_ptr",
        "//iree/base:synchronization",
        "//iree/base:threading",
        "//iree/base:tracing",
        "//iree/hal/host:host_local_allocator",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@half//:includes",
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

#===------------------------------------------------------------------------===#
# Implementation
#===------------------------------------------------------------------------===#
# TODO(benvanik): rename to :cc and expose via an api_cc.h.

cc_library(
    name = "hal",
    srcs = [
        "buffer.cc",
    ],
    hdrs = [
        "allocator.h",
        "buffer.h",
        "command_buffer.h",
        "command_queue.h",
        "debug_capture_manager.h",
        "descriptor_set.h",
        "descriptor_set_layout.h",
        "device.h",
        "device_info.h",
        "driver.h",
        "event.h",
        "executable.h",
        "executable_cache.h",
        "executable_format.h",
        "executable_layout.h",
        "resource.h",
        "semaphore.h",
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
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/types:span",
    ],
)

#===------------------------------------------------------------------------===#
# Debugging utilities and tools
#===------------------------------------------------------------------------===#

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
