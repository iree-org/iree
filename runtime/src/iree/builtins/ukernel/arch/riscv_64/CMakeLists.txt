
iree_add_all_subdirs()

iree_compiler_targeting_iree_arch(_IREE_UKERNEL_BITCODE_BUILD_RISCV_64 "riscv_64")
if(_IREE_UKERNEL_BITCODE_BUILD_RISCV_64)

iree_bitcode_library(
  NAME
    ukernel_bitcode_riscv_64_entry_points
  ARCH
    wasm_64
  SRCS
    "softmax_riscv_64_entry_point.c"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_riscv_64_rvv
  ARCH
    riscv_64
  SRCS
    "softmax_riscv_64_rvv.c"
  COPTS
    "-march=rv64gcv_zvl512b"
)

iree_link_bitcode(
  NAME
    ukernel_bitcode_riscv_64
  SRCS
    "ukernel_bitcode_riscv_64_rvv.bc"
)
elseif(IREE_BUILD_COMPILER AND IREE_TARGET_BACKEND_LLVM_CPU)
iree_make_empty_file("${CMAKE_CURRENT_BINARY_DIR}/ukernel_bitcode_riscv_64.bc")
iree_make_empty_file("${CMAKE_CURRENT_BINARY_DIR}/ukernel_bitcode_riscv_64_entry_points.bc")
endif() # _IREE_UKERNEL_BITCODE_BUILD_RISCV_64

if (NOT (IREE_ARCH STREQUAL "riscv_64"))
  return()
endif()

configure_file("config_riscv_64.h.in" "config_riscv_64.h")

iree_cc_library(
  NAME
    common_riscv_64
  HDRS
    "common_riscv_64.h"
  DEPS
    iree::builtins::ukernel::internal_headers
    iree::schemas::cpu_data
)

set(IREE_UK_RISCV_64_DEPS "")

set(IREE_UK_RISCV_64_RVV)
iree_cc_library(
  NAME
    riscv_64
  SRCS
    "softmax_riscv_64_rvv.c"
    "softmax_riscv_64_entry_point.c"
  DEPS
    ::common_riscv_64
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::internal_headers
    ${IREE_UK_RISCV_64_DEPS}
  PUBLIC
)

set(IREE_UK_ARCH_DEPS "iree::builtins::ukernel::arch::riscv_64" PARENT_SCOPE)
