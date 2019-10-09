package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Sequencer",
    srcs = ["SequencerModuleTranslation.cpp"],
    hdrs = ["SequencerModuleTranslation.h"],
    deps = [
        "//iree/base:status",
        "//iree/compiler/IR",
        "//iree/compiler/IR/Sequencer",
        "//iree/compiler/Serialization",
        "//iree/compiler/Transforms",
        "//iree/compiler/Transforms/Sequencer",
        "//iree/compiler/Utils",
        "//iree/hal:executable_format",
        "//iree/schemas",
        "@flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:Translation",
        "@tensorflow///compiler/mlir/xla:hlo",
        "@tensorflow///compiler/mlir/xla:xla_dialect_registration",
        "@tensorflow///compiler/mlir/xla:xla_legalize_control_flow",
    ],
    alwayslink = 1,
)
