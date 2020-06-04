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

# Default implementations for HAL types that use the host resources.
# These are generally just wrappers around host heap memory and host threads.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "async_command_queue",
    srcs = ["async_command_queue.cc"],
    hdrs = ["async_command_queue.h"],
    deps = [
        ":serial_submission_queue",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:semaphore",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "async_command_queue_test",
    srcs = ["async_command_queue_test.cc"],
    deps = [
        ":async_command_queue",
        ":serial_submission_queue",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/base:time",
        "//iree/hal:command_queue",
        "//iree/hal/testing:mock_command_buffer",
        "//iree/hal/testing:mock_command_queue",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    deps = [
        "//iree/base:logging",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/hal:buffer",
    ],
)

cc_library(
    name = "host_descriptor_set",
    srcs = ["host_descriptor_set.cc"],
    hdrs = ["host_descriptor_set.h"],
    deps = [
        "//iree/hal:descriptor_set",
        "//iree/hal:descriptor_set_layout",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "host_event",
    srcs = ["host_event.cc"],
    hdrs = ["host_event.h"],
    deps = [
        "//iree/hal:event",
    ],
)

cc_library(
    name = "host_executable_layout",
    srcs = ["host_executable_layout.cc"],
    hdrs = ["host_executable_layout.h"],
    deps = [
        "//iree/base:memory",
        "//iree/hal:descriptor_set_layout",
        "//iree/hal:executable_layout",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "host_local_allocator",
    srcs = ["host_local_allocator.cc"],
    hdrs = ["host_local_allocator.h"],
    deps = [
        ":host_buffer",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:buffer",
    ],
)

cc_library(
    name = "host_semaphore",
    srcs = ["host_semaphore.cc"],
    hdrs = ["host_semaphore.h"],
    deps = [
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:semaphore",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "host_semaphore_test",
    srcs = ["host_semaphore_test.cc"],
    deps = [
        ":host_semaphore",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "inproc_command_buffer",
    srcs = ["inproc_command_buffer.cc"],
    hdrs = ["inproc_command_buffer.h"],
    deps = [
        "//iree/base:arena",
        "//iree/base:intrusive_list",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
    ],
)

cc_library(
    name = "scheduling_model",
    hdrs = ["scheduling_model.h"],
    deps = [
        "//iree/hal:command_queue",
    ],
)

cc_library(
    name = "serial_command_processor",
    srcs = ["serial_command_processor.cc"],
    hdrs = ["serial_command_processor.h"],
    deps = [
        ":host_descriptor_set",
        ":host_executable_layout",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "serial_submission_queue",
    srcs = ["serial_submission_queue.cc"],
    hdrs = ["serial_submission_queue.h"],
    deps = [
        ":host_semaphore",
        "//iree/base:intrusive_list",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
    ],
)
