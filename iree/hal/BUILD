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
        "allocator.c",
        "allocator.h",
        "allocator_heap.c",
        "buffer.c",
        "buffer.h",
        "buffer_heap.c",
        "buffer_view.c",
        "buffer_view.cc",
        "buffer_view.h",
        "command_buffer.c",
        "command_buffer.h",
        "command_buffer_validation.c",
        "descriptor_set.c",
        "descriptor_set.h",
        "descriptor_set_layout.c",
        "descriptor_set_layout.h",
        "detail.h",
        "device.c",
        "device.h",
        "driver.c",
        "driver.h",
        "driver_registry.c",
        "driver_registry.h",
        "event.c",
        "event.h",
        "executable.c",
        "executable.h",
        "executable_cache.c",
        "executable_cache.h",
        "executable_layout.c",
        "executable_layout.h",
        "resource.h",
        "semaphore.c",
        "semaphore.h",
        "string_util.cc",
        "string_util.h",
    ],
    hdrs = [
        "api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:synchronization",
        "//iree/base:threading",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/hal/vulkan/util:ref_ptr",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@half//:includes",
    ],
)

cc_test(
    name = "string_util_test",
    srcs = ["string_util_test.cc"],
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
