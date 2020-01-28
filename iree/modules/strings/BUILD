load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "strings_module",
    srcs = ["strings_module.cc"],
    hdrs = ["strings_module.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:module",
        "//iree/vm:module_abi_cc",
        "//iree/vm:ref",
        "//iree/vm:stack",
        "//iree/vm:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "strings_module_test",
    srcs = ["strings_module_test.cc"],
    deps = [
        ":strings_module",
        ":strings_module_test_module_cc",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/testing:gtest_main",
        "//iree/vm:bytecode_module",
        "//iree/vm:context",
        "//iree/vm:instance",
        "//iree/vm:module",
        "//iree/vm:ref",
        "//iree/vm:stack",
        "//iree/vm:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_bytecode_module(
    name = "strings_module_test_module",
    src = "strings_module_test.mlir",
    cc_namespace = "iree::strings_module_test",
    translate_tool = "//iree/modules/strings/dialect:strings-translate",
    translation = "-iree-mlir-to-vm-bytecode-module",
)
