iree_add_all_subdirs()

iree_lit_test_suite(cfg = "//compiler:lit.cfg.py",
  NAME
    lit
  SRCS
    "smoketest.mlir"
  TOOLS
    FileCheck
    iree::tools::iree-opt
)
