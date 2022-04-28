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
        "DebugDatabaseBuilder.cpp",
        "DebugDatabaseBuilder.h",
        "TranslationRegistration.cpp",
    ],
    hdrs = [
        "BytecodeModuleTarget.h",
    ],
    deps = [
        "//iree/compiler/Dialect/Util/IR",
        "//iree/compiler/Dialect/Util/Transforms",
        "//iree/compiler/Dialect/VM/Analysis",
        "//iree/compiler/Dialect/VM/IR",
        "//iree/compiler/Dialect/VM/Transforms",
        "//iree/compiler/Dialect/VM/Utils:CallingConvention",
        "//iree/compiler/Utils",
        "//runtime/src/iree/schemas:bytecode_module_def_c_fbs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:Translation",
    ],
)
