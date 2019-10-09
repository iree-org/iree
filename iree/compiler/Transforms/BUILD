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
        "LegalizeTupleElementAccess.cpp",
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
        "//iree/compiler/IR",
        "//iree/compiler/IR/Interpreter",
        "//iree/compiler/IR/Sequencer",
        "//iree/compiler/Utils",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:Analysis",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:Pass",
        "//third_party/llvm/llvm/projects/google_mlir:StandardDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
        "//third_party/llvm/llvm/projects/google_mlir:TransformUtils",
        "//third_party/llvm/llvm/projects/google_mlir:Transforms",
        "//third_party/tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
