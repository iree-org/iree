# Testing utilities for IREE.

load("//iree:build_defs.bzl", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gtest",
    testonly = 1,
    hdrs = ["gtest.h"],
    deps = platform_trampoline_deps("gtest", "testing"),
)

cc_library(
    name = "gtest_main",
    testonly = 1,
    hdrs = ["gtest.h"],
    tags = ["keep_dep"],
    deps = platform_trampoline_deps("gtest_main", "testing"),
)
