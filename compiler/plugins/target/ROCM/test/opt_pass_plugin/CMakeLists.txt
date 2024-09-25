iree_cc_library(
  NAME
    GPUHello
  SRCS
    "GPUHello.cpp"
  DEPS
    LLVMSupport
)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "gpu_hello.mlir"
  TOOLS
    FileCheck
    iree-opt
)
