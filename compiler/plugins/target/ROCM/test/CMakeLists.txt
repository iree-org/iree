# NOTE: Bazel testing of this backend is impossible because there is no way
# for Bazel to bundle the AMD bitcode files that the backend depends on. Users
# of the compiler can pass explicit flags, but we prefer that default tests
# exercise default flags, which cannot be supported properly on Bazel builds.

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "external_function_validation.mlir"
    "smoketest.mlir"
    "target_device_features.mlir"
  TOOLS
    FileCheck
    iree-opt
)
