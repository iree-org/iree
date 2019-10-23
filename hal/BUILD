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
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "@com_google_absl//absl/base:core_headers",
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
        ":buffer_view",
        ":fence",
        ":heap_buffer",
        ":semaphore",
        "///base:api",
        "///base:api_util",
        "///base:shape",
        "///base:tracing",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
    deps = [
        "///base:api_hdrs",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":resource",
        "///base:bitfield",
        "///base:logging",
        "///base:source_location",
        "///base:status",
        "@com_google_absl//absl/base:core_headers",
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
        "///base:status",
        "///base:status_matchers",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_view",
    srcs = ["buffer_view.cc"],
    hdrs = ["buffer_view.h"],
    deps = [
        ":buffer",
        "///base:shape",
        "///base:source_location",
        "///base:status",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "buffer_view_test",
    srcs = [
        "buffer_view_test.cc",
    ],
    deps = [
        ":buffer",
        ":buffer_view",
        ":heap_buffer",
        "///base:status",
        "///base:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_view_string_util",
    srcs = ["buffer_view_string_util.cc"],
    hdrs = ["buffer_view_string_util.h"],
    deps = [
        ":allocator",
        ":buffer_view",
        ":heap_buffer",
        "///base:source_location",
        "///base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_test(
    name = "buffer_view_string_util_test",
    srcs = ["buffer_view_string_util_test.cc"],
    deps = [
        ":buffer_view_string_util",
        "///base:status",
        "///base:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "command_buffer",
    srcs = ["command_buffer.cc"],
    hdrs = ["command_buffer.h"],
    deps = [
        ":allocator",
        ":buffer",
        ":buffer_view",
        ":event",
        ":executable",
        ":resource",
        "///base:bitfield",
        "///base:shape",
        "///base:status",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "command_buffer_validation",
    srcs = ["command_buffer_validation.cc"],
    hdrs = ["command_buffer_validation.h"],
    deps = [
        ":command_buffer",
        "///base:logging",
        "///base:status",
    ],
)

cc_library(
    name = "command_queue",
    hdrs = ["command_queue.h"],
    deps = [
        ":command_buffer",
        ":fence",
        ":semaphore",
        "///base:bitfield",
        "///base:status",
        "///base:time",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "deferred_buffer",
    srcs = ["deferred_buffer.cc"],
    hdrs = ["deferred_buffer.h"],
    deps = [
        ":allocator",
        ":buffer",
        "///base:status",
    ],
)

cc_test(
    name = "deferred_buffer_test",
    srcs = ["deferred_buffer_test.cc"],
    deps = [
        ":deferred_buffer",
        ":heap_buffer",
        "///base:status_matchers",
        "///hal/testing:mock_allocator",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "device",
    hdrs = ["device.h"],
    deps = [
        ":allocator",
        ":buffer",
        ":command_queue",
        ":device_info",
        ":event",
        ":executable_cache",
        ":semaphore",
        "///base:status",
        "///base:time",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "device_info",
    hdrs = ["device_info.h"],
    deps = [
        "///base:bitfield",
        "@com_google_absl//absl/base:core_headers",
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
        ":fence",
        ":heap_buffer",
        "///base:source_location",
        "///base:status",
        "///base:time",
        "///base:tracing",
        "@com_google_absl//absl/synchronization",
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
        ":device",
        ":device_info",
        "///base:status",
    ],
)

cc_library(
    name = "driver_registry",
    srcs = ["driver_registry.cc"],
    hdrs = ["driver_registry.h"],
    deps = [
        ":driver",
        "///base:init",
        "///base:status",
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
        ":executable_spec",
        "///base:bitfield",
        "///base:ref_ptr",
        "///base:status",
    ],
)

cc_library(
    name = "executable_format",
    hdrs = ["executable_format.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
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
    name = "fence",
    hdrs = ["fence.h"],
    deps = [
        ":resource",
        "///base:status",
    ],
)

cc_library(
    name = "heap_buffer",
    srcs = ["heap_buffer.cc"],
    hdrs = ["heap_buffer.h"],
    deps = [
        ":allocator",
        ":buffer",
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "///hal/host:host_buffer",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "resource",
    hdrs = ["resource.h"],
    deps = [
        "///base:ref_ptr",
    ],
)

cc_library(
    name = "semaphore",
    hdrs = ["semaphore.h"],
    deps = [
        ":resource",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "stack_trace",
    hdrs = ["stack_trace.h"],
)
