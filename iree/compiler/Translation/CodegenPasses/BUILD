package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "CodegenPasses",
    srcs = [
        "HLOToLinalgOnBuffers.cpp",
        "HLOToLinalgOnTensors.cpp",
        "LinalgFusion.cpp",
        "Passes.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:map_xla_to_scalar_op",
    ],
    alwayslink = 1,
)
