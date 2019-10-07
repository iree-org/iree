package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Interpreter",
    srcs = ["InterpreterExecutableTranslation.cpp"],
    hdrs = ["InterpreterExecutableTranslation.h"],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/IR/Interpreter",
        "//iree/compiler/Serialization",
        "//iree/compiler/Transforms",
        "//iree/compiler/Transforms/Interpreter",
        "//iree/compiler/Utils",
        "//iree/schemas",
        "//third_party/flatbuffers",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:Pass",
        "//third_party/llvm/llvm/projects/google_mlir:StandardDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
        "//third_party/llvm/llvm/projects/google_mlir:Transforms",
        "//third_party/llvm/llvm/projects/google_mlir:Translation",
        "//third_party/tensorflow/compiler/mlir/xla:hlo",
        "//third_party/tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "//third_party/tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
    ],
    alwayslink = 1,
)
