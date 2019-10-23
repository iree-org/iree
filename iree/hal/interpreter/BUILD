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
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_format",
        "//iree/rt",
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
        "//iree/base:logging",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/hal:allocator",
        "//iree/hal:buffer_view",
        "//iree/hal:heap_buffer",
        "//iree/rt",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "//iree/vm:bytecode_module",
        "//iree/vm:bytecode_reader",
        "//iree/vm:bytecode_tables_interpreter",
        "//iree/vm:bytecode_util",
        "//iree/vm:opcode_info",
        "//iree/vm:type",
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
        "//iree/base:status",
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/rt",
        "//iree/vm:bytecode_tables_interpreter",
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
        "//iree/base:shape",
        "//iree/base:status",
        "//iree/hal:buffer_view",
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
        "//iree/base:memory",
        "//iree/base:status_matchers",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "interpreter_command_processor",
    srcs = ["interpreter_command_processor.cc"],
    hdrs = ["interpreter_command_processor.h"],
    deps = [
        ":bytecode_executable",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:buffer_view",
        "//iree/hal/host:host_local_command_processor",
        "//iree/rt",
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
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer_validation",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal:fence",
        "//iree/hal/host:async_command_queue",
        "//iree/hal/host:host_event",
        "//iree/hal/host:host_local_allocator",
        "//iree/hal/host:host_submission_queue",
        "//iree/hal/host:inproc_command_buffer",
        "//iree/rt",
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
        "//iree/hal:device_info",
        "//iree/hal:driver",
    ],
)

cc_library(
    name = "interpreter_driver_module",
    srcs = ["interpreter_driver_module.cc"],
    deps = [
        ":interpreter_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
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
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:allocator",
        "//iree/hal:buffer_view",
        "//iree/rt",
        "//iree/vm:bytecode_module",
        "//iree/vm:bytecode_tables_interpreter",
        "@com_google_absl//absl/types:span",
    ],
)
