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
        ":host_submission_queue",
        "///base:status",
        "///base:tracing",
        "///hal:command_queue",
        "///hal:fence",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "async_command_queue_test",
    srcs = ["async_command_queue_test.cc"],
    deps = [
        ":async_command_queue",
        ":host_submission_queue",
        "///base:status",
        "///base:status_matchers",
        "///base:time",
        "///hal:command_queue",
        "///hal/testing:mock_command_buffer",
        "///hal/testing:mock_command_queue",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    deps = [
        "///base:logging",
        "///base:source_location",
        "///base:status",
        "///hal:buffer",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "host_event",
    srcs = ["host_event.cc"],
    hdrs = ["host_event.h"],
    deps = [
        "///hal:event",
    ],
)

cc_library(
    name = "host_fence",
    srcs = ["host_fence.cc"],
    hdrs = ["host_fence.h"],
    deps = [
        "///base:status",
        "///base:tracing",
        "///hal:fence",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "host_fence_test",
    srcs = ["host_fence_test.cc"],
    deps = [
        ":host_fence",
        "///base:status",
        "///base:status_matchers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "host_local_allocator",
    srcs = ["host_local_allocator.cc"],
    hdrs = ["host_local_allocator.h"],
    deps = [
        ":host_buffer",
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "///hal:allocator",
        "///hal:buffer",
    ],
)

cc_library(
    name = "host_local_command_processor",
    srcs = ["host_local_command_processor.cc"],
    hdrs = ["host_local_command_processor.h"],
    deps = [
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "///hal:command_buffer",
    ],
)

cc_library(
    name = "host_submission_queue",
    srcs = ["host_submission_queue.cc"],
    hdrs = ["host_submission_queue.h"],
    deps = [
        ":host_fence",
        "///base:intrusive_list",
        "///base:status",
        "///base:tracing",
        "///hal:command_queue",
        "///hal:fence",
        "///hal:semaphore",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "host_submission_queue_test",
    srcs = ["host_submission_queue_test.cc"],
    deps = [
        ":host_submission_queue",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "inproc_command_buffer",
    srcs = ["inproc_command_buffer.cc"],
    hdrs = ["inproc_command_buffer.h"],
    deps = [
        "///base:arena",
        "///base:intrusive_list",
        "///base:status",
        "///base:tracing",
        "///hal:command_buffer",
    ],
)
