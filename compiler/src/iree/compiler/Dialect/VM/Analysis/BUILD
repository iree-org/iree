load("//iree:build_defs.oss.bzl", "iree_compiler_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_compiler_cc_library(
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
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
