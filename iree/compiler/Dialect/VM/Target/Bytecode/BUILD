package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
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
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/IREE/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/schemas:bytecode_module_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Translation",
    ],
)
