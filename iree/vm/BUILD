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
        "///iree/base:api",
        "///iree/base:api_util",
        "///iree/base:flatbuffer_util",
        "///iree/base:tracing",
        "///iree/rt:api",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
    deps = [
        "///iree/base:api_hdrs",
        "///iree/rt:api_hdrs",
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
        "///iree/base:flatbuffer_util",
        "///iree/base:status",
        "///iree/base:tracing",
        "///iree/hal:buffer_view",
        "///iree/rt",
        "///iree/schemas",
        "///iree/schemas/bytecode:bytecode_v0",
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
        "///iree/base:shape",
        "///iree/base:status",
        "///iree/hal:buffer_view",
        "///iree/hal:heap_buffer",
        "///iree/rt",
        "///iree/schemas/bytecode:bytecode_v0",
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
        "///iree/schemas/bytecode:interpreter_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_tables_sequencer",
    srcs = ["bytecode_tables_sequencer.cc"],
    hdrs = ["bytecode_tables_sequencer.h"],
    deps = [
        ":opcode_info",
        "///iree/schemas/bytecode:sequencer_bytecode_v0",
    ],
)

cc_library(
    name = "bytecode_util",
    srcs = ["bytecode_util.cc"],
    hdrs = ["bytecode_util.h"],
    deps = [
        "///iree/schemas/bytecode:bytecode_v0",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bytecode_validator",
    srcs = ["bytecode_validator.cc"],
    hdrs = ["bytecode_validator.h"],
    deps = [
        ":bytecode_module",
        "///iree/base:status",
        "///iree/schemas",
    ],
)

cc_library(
    name = "opcode_info",
    hdrs = ["opcode_info.h"],
    deps = [
        "///iree/schemas/bytecode:bytecode_v0",
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
        "///iree/base:logging",
        "///iree/base:memory",
        "///iree/base:status",
        "///iree/hal:buffer_view",
        "///iree/hal:command_queue",
        "///iree/hal:device",
        "///iree/hal:device_placement",
        "///iree/hal:heap_buffer",
        "///iree/rt",
        "///iree/schemas/bytecode:sequencer_bytecode_v0",
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
        "///iree/base:status",
        "///iree/base:tracing",
        "///iree/hal:buffer_view",
        "///iree/rt",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "source_map_resolver",
    srcs = ["source_map_resolver.cc"],
    hdrs = ["source_map_resolver.h"],
    deps = [
        "///iree/base:flatbuffer_util",
        "///iree/base:status",
        "///iree/rt",
        "///iree/schemas",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "type",
    srcs = ["type.cc"],
    hdrs = ["type.h"],
    deps = [
        "///iree/base:status",
        "///iree/schemas",
        "///iree/schemas/bytecode:bytecode_v0",
    ],
)
