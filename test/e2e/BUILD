# Tests for end-to-end IREE support.

load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//iree/tools:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//iree/tools:iree-run-mlir",
        "//third_party/llvm/llvm:FileCheck",
        "//third_party/llvm/llvm/projects/google_mlir:run_lit.sh",
    ],
)
