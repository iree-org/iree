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
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:Pass",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
        "//third_party/llvm/llvm/projects/google_mlir:TransformUtils",
        "//third_party/llvm/llvm/projects/google_mlir:Transforms",
        "//third_party/tensorflow/compiler/mlir/xla:hlo",
        "//third_party/tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
        "//third_party/tensorflow/compiler/mlir/xla:xla_lower_general_dot",
    ],
    alwayslink = 1,
)
