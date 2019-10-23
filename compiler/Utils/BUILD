# Utilities for working with IREE MLIR types.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Utils",
    srcs = [
        "DispatchUtils.cpp",
        "MemRefUtils.cpp",
        "ModuleUtils.cpp",
        "OpCreationUtils.cpp",
        "OpUtils.cpp",
        "TranslationUtils.cpp",
        "TypeConversionUtils.cpp",
    ],
    hdrs = [
        "DispatchUtils.h",
        "Macros.h",
        "MemRefUtils.h",
        "ModuleUtils.h",
        "OpCreationUtils.h",
        "OpUtils.h",
        "TranslationUtils.h",
        "TypeConversionUtils.h",
    ],
    deps = [
        "///compiler/IR",
        "///schemas",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
    ],
)
