# HAL implementation running on the CPU using the IREE bytecode.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bytecode_cache",
    srcs = ["bytecode_cache.cc"],
    hdrs = ["bytecode_cache.h"],
    deps = [
        ":bytecode_executable",
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "///hal:allocator",
        "///hal:executable",
        "///hal:executable_cache",
        "///hal:executable_format",
        "///rt",
    ],
)

cc_library(
    name = "bytecode_dispatch",
    srcs = [
        "bytecode_dispatch.cc",
        "bytecode_dispatch_conversion.h",
        "bytecode_dispatch_util.cc",
        "bytecode_dispatch_util.h",
    ],
    hdrs = ["bytecode_dispatch.h"],
    deps = [
        ":bytecode_kernels",
        "///base:logging",
        "///base:memory",
        "///base:status",
        "///hal:allocator",
        "///hal:buffer_view",
        "///hal:heap_buffer",
        "///rt",
        "///schemas/bytecode:interpreter_bytecode_v0",
        "///vm:bytecode_module",
        "///vm:bytecode_reader",
        "///vm:bytecode_tables_interpreter",
        "///vm:bytecode_util",
        "///vm:opcode_info",
        "///vm:type",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bytecode_executable",
    srcs = ["bytecode_executable.cc"],
    hdrs = ["bytecode_executable.h"],
    deps = [
        ":interpreter_module",
        "///base:status",
        "///hal:allocator",
        "///hal:executable",
        "///hal:executable_spec",
        "///rt",
        "///vm:bytecode_tables_interpreter",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bytecode_kernels",
    hdrs = ["bytecode_kernels.h"],
    textual_hdrs = [
        # TODO(benvanik): SIMD variants.
        "bytecode_kernels_generic.h",
        "bytecode_kernels_ruy.h",
    ],
    deps = [
        "///base:shape",
        "///base:status",
        "///hal:buffer_view",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow//tensorflow/lite/experimental/ruy",
        "@org_tensorflow//tensorflow/lite/experimental/ruy:context",
    ],
)

cc_test(
    name = "bytecode_kernels_test",
    srcs = ["bytecode_kernels_test.cc"],
    deps = [
        ":bytecode_kernels",
        "///base:memory",
        "///base:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "interpreter_command_processor",
    srcs = ["interpreter_command_processor.cc"],
    hdrs = ["interpreter_command_processor.h"],
    deps = [
        ":bytecode_executable",
        "///base:source_location",
        "///base:status",
        "///base:tracing",
        "///hal:buffer_view",
        "///hal/host:host_local_command_processor",
        "///rt",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "interpreter_device",
    srcs = ["interpreter_device.cc"],
    hdrs = ["interpreter_device.h"],
    deps = [
        ":bytecode_cache",
        ":bytecode_kernels",
        ":interpreter_command_processor",
        "///base:memory",
        "///base:status",
        "///base:tracing",
        "///hal:command_buffer_validation",
        "///hal:command_queue",
        "///hal:device",
        "///hal:fence",
        "///hal/host:async_command_queue",
        "///hal/host:host_event",
        "///hal/host:host_local_allocator",
        "///hal/host:host_submission_queue",
        "///hal/host:inproc_command_buffer",
        "///rt",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "interpreter_driver",
    srcs = ["interpreter_driver.cc"],
    hdrs = ["interpreter_driver.h"],
    deps = [
        ":interpreter_device",
        "///hal:device_info",
        "///hal:driver",
    ],
)

cc_library(
    name = "interpreter_driver_module",
    srcs = ["interpreter_driver_module.cc"],
    deps = [
        ":interpreter_driver",
        "///base:init",
        "///base:status",
        "///hal:driver_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "interpreter_module",
    srcs = ["interpreter_module.cc"],
    hdrs = ["interpreter_module.h"],
    deps = [
        ":bytecode_dispatch",
        ":bytecode_kernels",
        "///base:flatbuffer_util",
        "///base:status",
        "///base:tracing",
        "///hal:allocator",
        "///hal:buffer_view",
        "///rt",
        "///vm:bytecode_module",
        "///vm:bytecode_tables_interpreter",
        "@com_google_absl//absl/types:span",
    ],
)
