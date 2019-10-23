package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "bytecode_v0",
    hdrs = ["bytecode_v0.h"],
    deps = [
        "///base:bitfield",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "interpreter_bytecode_v0",
    hdrs = ["interpreter_bytecode_v0.h"],
    deps = [
        ":bytecode_v0",
    ],
)

cc_library(
    name = "sequencer_bytecode_v0",
    hdrs = ["sequencer_bytecode_v0.h"],
    deps = [
        ":bytecode_v0",
    ],
)
