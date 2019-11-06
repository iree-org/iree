package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Tools",
    srcs = [
        "VMOpEncoderGen.cpp",
        "VMOpTableGen.cpp",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TableGen",
    ],
    alwayslink = 1,
)
