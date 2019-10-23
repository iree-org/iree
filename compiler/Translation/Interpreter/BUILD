package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Interpreter",
    srcs = ["InterpreterExecutableTranslation.cpp"],
    hdrs = ["InterpreterExecutableTranslation.h"],
    deps = [
        "///compiler/IR",
        "///compiler/IR/Interpreter",
        "///compiler/Serialization",
        "///compiler/Transforms",
        "///compiler/Transforms/Interpreter",
        "///compiler/Utils",
        "///schemas",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:Translation",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
    ],
    alwayslink = 1,
)
