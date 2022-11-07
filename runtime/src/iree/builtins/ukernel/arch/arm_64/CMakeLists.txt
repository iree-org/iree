iree_cc_library(
  NAME
    assembly
  HDRS
    "assembly.h"
)

iree_cc_library(
  NAME
    mmt4d_tile_arm_64
  SRCS
    "mmt4d_tile_arm_64.S"
  DEPS
    ::assembly
    iree::builtins::ukernel::exported_flag_bits
)
list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64")

check_cxx_compiler_flag("-march=armv8.2-a+dotprod" IREE_UK_BUILD_ARM_64_DOTPROD)
if(IREE_UK_BUILD_ARM_64_DOTPROD)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_dotprod
    SRCS
      "mmt4d_tile_arm_64_dotprod.S"
    COPTS
      "-march=armv8.2-a+dotprod"
    DEPS
      ::assembly
      iree::builtins::ukernel::exported_flag_bits

  )
  list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64_dotprod")
endif()

check_cxx_compiler_flag("-march=armv8.2-a+i8mm" IREE_UK_BUILD_ARM_64_I8MM)
if(IREE_UK_BUILD_ARM_64_I8MM)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_i8mm
    SRCS
      "mmt4d_tile_arm_64_i8mm.S"
    COPTS
      "-march=armv8.2-a+i8mm"
    DEPS
      ::assembly
      iree::builtins::ukernel::exported_flag_bits
  )
  list(APPEND IREE_UK_MMT4D_TILE_ARM_64_DEPS "iree::builtins::ukernel::arch::arm_64::mmt4d_tile_arm_64_i8mm")
endif()

configure_file(config.h.in config.h)

iree_cc_library(
  NAME
    mmt4d_select_tile_arm_64
  HDRS
    "mmt4d_select_tile_arm_64.h"
  SRCS
    "mmt4d_select_tile_arm_64.c"
  DEPS
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::common
    ${IREE_UK_MMT4D_TILE_ARM_64_DEPS}
  PUBLIC
)
