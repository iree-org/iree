################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Dialect/Flow/Transforms/test/BUILD.bazel          #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "annotate_dispatches.mlir"
    "canonicalize.mlir"
    "capture_dispatch_dynamic_dims.mlir"
    "capture_scf_for_dynamic_dims.mlir"
    "cleanup_tensor_shapes.mlir"
    "deduplicate_executables.mlir"
    "export_benchmark_funcs.mlir"
    "initialize_empty_tensors.mlir"
    "inject_dispatch_tracing.mlir"
    "inject_tensor_tracing.mlir"
    "insert_dispatch_debug_targets.mlir"
    "outline_constants.mlir"
    "outline_dispatch_externs.mlir"
    "outline_dispatch_regions.mlir"
    "pipeline_tests.mlir"
    "top_level_scf_to_cfg.mlir"
    "verify_input_ir.mlir"
  TOOLS
    FileCheck
    iree-opt
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
