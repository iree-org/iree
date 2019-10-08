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
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/memory",
        "//third_party/absl/types:span",
    ],
)

cc_library(
    name = "dawn_driver",
    srcs = ["dawn_driver.cc"],
    hdrs = ["dawn_driver.h"],
    deps = [
        ":dawn_device",
        "//iree/base:status",
        "//iree/hal:device_info",
        "//iree/hal:driver",
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
        "//third_party/absl/flags:flag",
    ],
    alwayslink = 1,
)
