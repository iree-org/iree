# Tests for end-to-end IREE support starting from the XLA HLO dialect.

load("//:build_defs.google.bzl", "iree_glob_lit_tests", "iree_setup_lit_package")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_setup_lit_package(
    data = [
        "///tools:iree-run-mlir",
    ],
)

iree_glob_lit_tests()
