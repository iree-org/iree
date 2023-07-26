iree_add_all_subdirs()

iree_bitcode_library(
  NAME
    ukernel_bitcode_riscv_32_entry_points
  ARCH
    wasm_32
  SRCS
    "mmt4d_riscv_32_entry_point.c"
)

iree_bitcode_library(
  NAME
    ukernel_bitcode_riscv_32_base
  ARCH
    riscv_32
  SRCS
    "mmt4d_riscv_32.c"
)

iree_link_bitcode(
  NAME
    ukernel_bitcode_riscv_32
  SRCS
    "ukernel_bitcode_riscv_32_base.bc"
    "ukernel_bitcode_riscv_32_entry_points.bc"
)
