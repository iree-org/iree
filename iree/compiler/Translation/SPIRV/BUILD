package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "SPIRV",
    srcs = [
        "AffineExprCodegen.h",
        "IREEIndexComputation.cpp",
        "IREEToSPIRV.cpp",
        "IREEToSPIRVPass.cpp",
        "IndexComputation.cpp",
        "SPIRVExecutableTranslation.cpp",
        "SPIRVLowering.cpp",
        "SPIRVLowering.h",
        "XLAIndexPropagation.cpp",
    ],
    hdrs = [
        "IREEIndexComputation.h",
        "IREEToSPIRV.h",
        "IREEToSPIRVPass.h",
        "IndexComputation.h",
        "SPIRVExecutableTranslation.h",
        "XLAIndexPropagation.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/Translation/SPIRV/Kernels",
        "//iree/compiler/Utils",
        "//iree/schemas",
        "//iree/schemas:spirv_executable_def_cc_fbs",
        "//third_party/flatbuffers",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:Pass",
        "//third_party/llvm/llvm/projects/google_mlir:SPIRVDialect",
        "//third_party/llvm/llvm/projects/google_mlir:SPIRVDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:SPIRVSerialization",
        "//third_party/llvm/llvm/projects/google_mlir:StandardDialectRegistration",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
        "//third_party/llvm/llvm/projects/google_mlir:Transforms",
        "//third_party/llvm/llvm/projects/google_mlir:Translation",
        "//third_party/tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
