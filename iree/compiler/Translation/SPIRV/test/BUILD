# Tests for common transforms.

load("//iree:build_defs.bzl", "iree_glob_lit_tests")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_glob_lit_tests(
    data = [":test_utilities"],
    driver = "//iree/tools:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//iree/tools:iree-opt",
        "@llvm//:FileCheck",
        "@local_config_mlir//:run_lit.sh",
    ],
)
