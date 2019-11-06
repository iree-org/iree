package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["CommonBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "Dialect",
    srcs = [
        "IREEDialect.cpp",
    ],
    hdrs = [
        "IREEDialect.h",
        "Traits.h",
        "Types.h",
    ],
    deps = [
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)
