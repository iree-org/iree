# Conformance Test Suite (CTS) for HAL implementations.

load("//iree:build_defs.bzl", "PLATFORM_VULKAN_TEST_DEPS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "cts_test_base",
    testonly = True,
    hdrs = ["cts_test_base.h"],
    data = [
        # When building with --config=asan you must specify the following
        # envvar when using Vulkan + a local Nvidia GPU:
        #   LSAN_OPTIONS=suppressions=third_party/iree/tools/sanitizer_suppressions.txt
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    deps = [
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/hal:driver_registry",

        # HAL driver modules.
        "//iree/hal/interpreter:interpreter_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
        # "//iree/hal/dawn:dawn_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)

cc_test(
    name = "allocator_test",
    srcs = ["allocator_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)

cc_test(
    name = "command_buffer_test",
    srcs = ["command_buffer_test.cc"],
    deps = [
        ":cts_test_base",
        "//iree/base:status",
        "//iree/base:status_matchers",
        "//iree/hal:driver_registry",
        "//iree/testing:gtest",
    ],
)
