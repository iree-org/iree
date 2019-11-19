package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Bytecode",
    srcs = [
        "BytecodeEncoder.cpp",
        "BytecodeEncoder.h",
        "BytecodeModuleTarget.cpp",
        "ConstantEncoder.cpp",
        "ConstantEncoder.h",
        "TranslationFlags.cpp",
        "TranslationRegistration.cpp",
    ],
    hdrs = [
        "BytecodeModuleTarget.h",
        "TranslationFlags.h",
    ],
    deps = [
        "//iree/compiler/Dialect",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/schemas:bytecode_module_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)
