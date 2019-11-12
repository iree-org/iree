package(
    default_visibility = ["//visibility:public"],
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
        "ValueLiveness.h",
    ],
    deps = [
        "//iree/compiler/Dialect",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)
