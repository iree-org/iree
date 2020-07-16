package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ConvertShapeToShapex",
    srcs = ["ConvertShapeToShapex.cpp"],
    deps = [
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "Conversion",
    hdrs = ["Passes.h"],
    deps = [
        ":ConvertShapeToShapex",
        "@llvm-project//mlir:Pass",
    ],
)
