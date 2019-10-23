package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Transforms",
    srcs = [
        "AggressiveOpElimination.cpp",
        "AssignFunctionOrdinals.cpp",
        "ConvertFromTupleCallingConvention.cpp",
        "ConvertToMemRefCallingConvention.cpp",
        "DropUnreachableFunctions.cpp",
        "DropUnusedExecutables.cpp",
        "LegalizeTypeStorage.cpp",
        "LowerStdToIreeDialect.cpp",
        "LowerXLAToIreeDialect.cpp",
    ],
    hdrs = [
        "ConversionUtils.h",
        "Passes.h",
        "Rewrites.h",
    ],
    deps = [
        "///compiler/IR",
        "///compiler/IR/Interpreter",
        "///compiler/IR/Sequencer",
        "///compiler/Utils",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
