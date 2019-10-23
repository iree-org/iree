# Bytecode VM used by the IREE sequencer and interpreter.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":api_hdrs",
        ":sequencer_module",
        "///base:api",
        "///base:api_util",
        "///base:flatbuffer_util",
        "///base:tracing",
        "///rt:api",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
    deps = [
        "///base:api_hdrs",
        "///rt:api_hdrs",
    ],
)

cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_disassembler.cc",
        "bytecode_module.cc",
    ],
    hdrs = [
        "bytecode_disassembler.h",
        "bytecode_module.h",
    ],
    deps = [
        ":bytecode_util",
        ":opcode_info",
        ":source_map_resolver",
        ":type",
        "///base:flatbuffer_util",
        "///base:status",
        "///base:tracing",
        "///hal:buffer_view",
        "///rt",
        "///schemas",
        "///schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bytecode_reader",
    srcs = ["bytecode_reader.cc"],
    hdrs = ["bytecode_reader.h"],
    deps = [
        ":bytecode_module",
        ":type",
        "///base:shape",
        "///base:status",
        "///hal:buffer_view",
        "///hal:heap_buffer",
        "///rt",
        "///schemas/bytecode:bytecode_v0",
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
        "///schemas/bytecode:interpreter_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_tables_sequencer",
    srcs = ["bytecode_tables_sequencer.cc"],
    hdrs = ["bytecode_tables_sequencer.h"],
    deps = [
        ":opcode_info",
        "///schemas/bytecode:sequencer_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_util",
    srcs = ["bytecode_util.cc"],
    hdrs = ["bytecode_util.h"],
    deps = [
        "///schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bytecode_validator",
    srcs = ["bytecode_validator.cc"],
    hdrs = ["bytecode_validator.h"],
    deps = [
        ":bytecode_module",
        "///base:status",
        "///schemas",
    ],
)

cc_library(
    name = "opcode_info",
    hdrs = ["opcode_info.h"],
    deps = [
        "///schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sequencer_dispatch",
    srcs = ["sequencer_dispatch.cc"],
    hdrs = ["sequencer_dispatch.h"],
    deps = [
        ":bytecode_module",
        ":bytecode_reader",
        ":bytecode_tables_sequencer",
        ":bytecode_util",
        ":opcode_info",
        "///base:logging",
        "///base:memory",
        "///base:status",
        "///hal:buffer_view",
        "///hal:command_queue",
        "///hal:device",
        "///hal:device_placement",
        "///hal:heap_buffer",
        "///rt",
        "///schemas/bytecode:sequencer_bytecode_v0",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sequencer_module",
    srcs = ["sequencer_module.cc"],
    hdrs = ["sequencer_module.h"],
    deps = [
        ":bytecode_module",
        ":bytecode_tables_sequencer",
        ":sequencer_dispatch",
        "///base:status",
        "///base:tracing",
        "///hal:buffer_view",
        "///rt",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "source_map_resolver",
    srcs = ["source_map_resolver.cc"],
    hdrs = ["source_map_resolver.h"],
    deps = [
        "///base:flatbuffer_util",
        "///base:status",
        "///rt",
        "///schemas",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        "///base:status",
        "///schemas",
        "///schemas/bytecode:bytecode_v0",
    ],
)
