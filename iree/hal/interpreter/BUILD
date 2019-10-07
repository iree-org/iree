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
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "//iree/vm:bytecode_reader",
        "//iree/vm:bytecode_tables_interpreter",
        "//iree/vm:bytecode_util",
        "//iree/vm:function",
        "//iree/vm:opcode_info",
        "//iree/vm:stack",
        "//iree/vm:type",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/types:span",
    ],
)

cc_library(
    name = "bytecode_executable",
    srcs = ["bytecode_executable.cc"],
    hdrs = ["bytecode_executable.h"],
    deps = [
        ":interpreter_context",
        "//iree/base:status",
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/vm:bytecode_tables_interpreter",
        "//iree/vm:bytecode_validator",
        "//iree/vm:context",
        "//iree/vm:module",
        "//iree/vm:module_printer",
        "//third_party/absl/types:span",
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
        "//third_party/absl/algorithm",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/memory",
        "//third_party/absl/types:span",
        "//third_party/tensorflow/lite/experimental/ruy",
        "//third_party/tensorflow/lite/experimental/ruy:context",
    ],
)

cc_test(
    name = "bytecode_kernels_test",
    srcs = ["bytecode_kernels_test.cc"],
    deps = [
        ":bytecode_kernels",
        "//iree/base:memory",
        "//iree/base:status_matchers",
        "//testing/base/public:gunit_main",
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
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/types:span",
    ],
)

cc_library(
    name = "interpreter_context",
    srcs = ["interpreter_context.cc"],
    hdrs = ["interpreter_context.h"],
    deps = [
        ":bytecode_dispatch",
        ":bytecode_kernels",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/hal:allocator",
        "//iree/hal:buffer_view",
        "//iree/vm:context",
        "//iree/vm:function",
        "//iree/vm:stack",
        "//third_party/absl/types:span",
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
        "//third_party/absl/container:inlined_vector",
        "//third_party/absl/memory",
        "//third_party/absl/types:span",
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
