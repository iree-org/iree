# HAL implementation using Dawn and SPIR-V executables.
#   https://dawn.googlesource.com/dawn

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "dawn_device",
    srcs = ["dawn_device.cc"],
    hdrs = ["dawn_device.h"],
    deps = [
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal:executable_cache",
        "//iree/hal:fence",
        "//iree/hal/host:host_local_allocator",
        "//third_party/dawn:dawn_headers",
        "//third_party/dawn:dawn_native",
        "//third_party/dawn:dawn_static_proc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dawn_driver",
    srcs = ["dawn_driver.cc"],
    hdrs = ["dawn_driver.h"],
    deps = [
        ":dawn_device",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:device_info",
        "//iree/hal:driver",
        "//third_party/dawn:dawn_headers",
        "//third_party/dawn:dawn_native",
        "//third_party/dawn:dawn_static_proc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(scotttodd): Use SwiftShader to test Vulkan backend
cc_test(
    name = "dawn_driver_test",
    srcs = ["dawn_driver_test.cc"],
    deps = [
        ":dawn_driver",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "dawn_driver_module",
    srcs = ["dawn_driver_module.cc"],
    deps = [
        ":dawn_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:driver_registry",
        "@com_google_absl//absl/flags:flag",
    ],
    alwayslink = 1,
)
