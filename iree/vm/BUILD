# Bytecode VM used by the IREE sequencer and interpreter.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# Enable the debug service and other profiling features.
# $ bazel build --define=IREE_DEBUG=1 :some_target
config_setting(
    name = "debug",
    define_values = {
        "IREE_DEBUG": "1",
    },
)

cc_library(
    name = "bytecode_printer",
    srcs = ["bytecode_printer.cc"],
    hdrs = ["bytecode_printer.h"],
    deps = [
        ":bytecode_util",
        ":executable_table",
        ":function_table",
        ":module",
        ":opcode_info",
        ":source_map",
        ":type",
        "//iree/base:status",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bytecode_reader",
    srcs = ["bytecode_reader.cc"],
    hdrs = ["bytecode_reader.h"],
    deps = [
        ":function",
        ":stack",
        ":type",
        "//iree/base:shape",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "//iree/hal:heap_buffer",
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "bytecode_tables_interpreter",
    srcs = ["bytecode_tables_interpreter.cc"],
    hdrs = ["bytecode_tables_interpreter.h"],
    deps = [
        ":opcode_info",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_tables_sequencer",
    srcs = ["bytecode_tables_sequencer.cc"],
    hdrs = ["bytecode_tables_sequencer.h"],
    deps = [
        ":opcode_info",
        "//iree/schemas/bytecode:sequencer_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_util",
    srcs = ["bytecode_util.cc"],
    hdrs = ["bytecode_util.h"],
    deps = [
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bytecode_validator",
    srcs = ["bytecode_validator.cc"],
    hdrs = ["bytecode_validator.h"],
    deps = [
        ":context",
        ":module",
        "//iree/base:status",
        "//iree/schemas",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    deps = [
        ":function",
        ":module",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_table",
    srcs = ["executable_table.cc"],
    hdrs = ["executable_table.h"],
    deps = [
        "//iree/base:flatbuffer_util",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/schemas",
    ],
)

cc_library(
    name = "fiber_state",
    srcs = ["fiber_state.cc"],
    hdrs = ["fiber_state.h"],
    deps = [
        ":instance",
        ":stack",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function",
    srcs = ["function.cc"],
    hdrs = ["function.h"],
    deps = [
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "//iree/schemas",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function_table",
    srcs = ["function_table.cc"],
    hdrs = ["function_table.h"],
    deps = [
        ":function",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/schemas",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "instance",
    srcs = ["instance.cc"],
    hdrs = ["instance.h"],
    deps = [
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/hal:device_manager",
        "//iree/vm/debug:debug_server_interface",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "module",
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        ":executable_table",
        ":function_table",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/schemas",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "module_printer",
    srcs = ["module_printer.cc"],
    hdrs = ["module_printer.h"],
    deps = [
        ":bytecode_printer",
        ":module",
        ":opcode_info",
        ":source_map",
        "//iree/base:bitfield",
        "//iree/base:status",
    ],
)

cc_library(
    name = "opcode_info",
    hdrs = ["opcode_info.h"],
    deps = [
        "//iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sequencer_context",
    srcs = ["sequencer_context.cc"],
    hdrs = ["sequencer_context.h"],
    deps = [
        ":context",
        ":fiber_state",
        ":function",
        ":instance",
        ":module",
        ":sequencer_dispatch",
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sequencer_dispatch",
    srcs = ["sequencer_dispatch.cc"],
    hdrs = ["sequencer_dispatch.h"],
    deps = [
        ":bytecode_reader",
        ":bytecode_tables_sequencer",
        ":bytecode_util",
        ":function",
        ":opcode_info",
        ":stack",
        "//iree/base:logging",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal:device_placement",
        "//iree/hal:heap_buffer",
        "//iree/schemas/bytecode:sequencer_bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "source_map",
    srcs = ["source_map.cc"],
    hdrs = ["source_map.h"],
    deps = [
        "//iree/base:flatbuffer_util",
        "//iree/base:status",
        "//iree/schemas",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "stack",
    srcs = [
        "stack.cc",
        "stack_frame.cc",
    ],
    hdrs = [
        "stack.h",
        "stack_frame.h",
    ],
    deps = [
        ":function",
        ":module",
        "//iree/base:status",
        "//iree/hal:buffer_view",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        "//iree/base:status",
        "//iree/schemas",
        "//iree/schemas/bytecode:bytecode_v0",
    ],
)
