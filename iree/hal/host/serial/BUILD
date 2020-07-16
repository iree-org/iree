# Copyright 2020 Google LLC
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
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "async_command_queue",
    srcs = ["async_command_queue.cc"],
    hdrs = ["async_command_queue.h"],
    deps = [
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:semaphore",
        "//iree/hal/host/serial:serial_submission_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "async_command_queue_test",
    srcs = ["async_command_queue_test.cc"],
    deps = [
        ":async_command_queue",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/base:time",
        "//iree/hal:command_queue",
        "//iree/hal/host/serial:serial_submission_queue",
        "//iree/hal/testing:mock_command_buffer",
        "//iree/hal/testing:mock_command_queue",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "serial_command_processor",
    srcs = ["serial_command_processor.cc"],
    hdrs = ["serial_command_processor.h"],
    deps = [
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
        "//iree/hal/host:host_descriptor_set",
        "//iree/hal/host:host_executable",
        "//iree/hal/host:host_executable_layout",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "serial_scheduling_model",
    srcs = ["serial_scheduling_model.cc"],
    hdrs = ["serial_scheduling_model.h"],
    deps = [
        ":async_command_queue",
        ":serial_command_processor",
        ":serial_submission_queue",
        "//iree/base:memory",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal/host:condvar_semaphore",
        "//iree/hal/host:inproc_command_buffer",
        "//iree/hal/host:nop_event",
        "//iree/hal/host:scheduling_model",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "serial_submission_queue",
    srcs = ["serial_submission_queue.cc"],
    hdrs = ["serial_submission_queue.h"],
    deps = [
        "//iree/base:intrusive_list",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal/host:condvar_semaphore",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
    ],
)
