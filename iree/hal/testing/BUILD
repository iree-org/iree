# Test utilities for HAL-specific code.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mock_allocator",
    testonly = True,
    hdrs = ["mock_allocator.h"],
    deps = [
        "//iree/hal:allocator",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_command_buffer",
    testonly = True,
    hdrs = ["mock_command_buffer.h"],
    deps = [
        "//iree/hal:command_buffer",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mock_command_queue",
    testonly = True,
    hdrs = ["mock_command_queue.h"],
    deps = [
        "//iree/hal:command_queue",
        "@com_google_googletest//:gtest",
    ],
)
