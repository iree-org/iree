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
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:fence",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/synchronization",
    ],
)

cc_test(
    name = "async_command_queue_test",
    srcs = ["async_command_queue_test.cc"],
    deps = [
        ":async_command_queue",
        ":host_submission_queue",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/base:time",
        "//iree/hal:command_queue",
        "//iree/hal/testing:mock_command_buffer",
        "//iree/hal/testing:mock_command_queue",
        "//testing/base/public:gunit_main",
        "//third_party/absl/memory",
        "//third_party/absl/time",
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
        "//third_party/absl/base:core_headers",
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
    name = "host_fence",
    srcs = ["host_fence.cc"],
    hdrs = ["host_fence.h"],
    deps = [
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:fence",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/synchronization",
        "//third_party/absl/types:span",
    ],
)

cc_test(
    name = "host_fence_test",
    srcs = ["host_fence_test.cc"],
    deps = [
        ":host_fence",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//testing/base/public:gunit_main",
        "//third_party/absl/time",
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
    name = "host_local_command_processor",
    srcs = ["host_local_command_processor.cc"],
    hdrs = ["host_local_command_processor.h"],
    deps = [
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer",
    ],
)

cc_library(
    name = "host_submission_queue",
    srcs = ["host_submission_queue.cc"],
    hdrs = ["host_submission_queue.h"],
    deps = [
        ":host_fence",
        "//iree/base:intrusive_list",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:fence",
        "//iree/hal:semaphore",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/synchronization",
    ],
)

cc_test(
    name = "host_submission_queue_test",
    srcs = ["host_submission_queue_test.cc"],
    deps = [
        ":host_submission_queue",
        "//testing/base/public:gunit_main",
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
