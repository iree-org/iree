package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Analysis",
    srcs = [
        "RegisterAllocation.cpp",
        "RegisterAllocationTest.cpp",
        "ValueLiveness.cpp",
        "ValueLivenessTest.cpp",
    ],
    hdrs = [
        "RegisterAllocation.h",
        "TestPasses.h",
        "ValueLiveness.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
