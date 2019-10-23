# Implementations for iree/testing/

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gtest_internal",
    testonly = 1,
    hdrs = ["gtest_internal.h"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gtest_main_internal",
    testonly = 1,
    srcs = ["gtest_main_internal.cc"],
    hdrs = ["gtest_internal.h"],
    deps = [
        "//iree/base:init",
        "@com_google_googletest//:gtest",
    ],
)
