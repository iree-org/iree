package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "StandardToVM",
    srcs = [
        "ConvertStandardToVM.cpp",
        "ConvertStandardToVMTest.cpp",
    ],
    hdrs = [
        "ConvertStandardToVM.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)
