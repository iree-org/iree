# Transforms specific to the IREE interpreter.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Interpreter",
    srcs = [
        "ExpandReductionsToOps.cpp",
        "LegalizeInterpreterOps.cpp",
        "LowerInterpreterDialect.cpp",
        "LowerStdToInterpreterDialect.cpp",
        "LowerToInterpreterDialect.cpp",
        "LowerXLAToInterpreterDialect.cpp",
        "MakeExecutableABI.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Rewrites.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/IR/Interpreter",
        "//iree/compiler/Serialization",
        "//iree/compiler/Transforms",
        "//iree/compiler/Utils",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@tensorflow///compiler/mlir/xla:hlo",
        "@tensorflow///compiler/mlir/xla:xla_legalize_to_standard",
        "@tensorflow///compiler/mlir/xla:xla_lower_general_dot",
    ],
    alwayslink = 1,
)
