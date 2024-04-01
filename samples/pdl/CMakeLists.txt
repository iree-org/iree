iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "preprocessing_pdl_patterns.mlir"
  TOOLS
    FileCheck
    iree-compile
  DATA
    linalg_pdl.mlir
)
