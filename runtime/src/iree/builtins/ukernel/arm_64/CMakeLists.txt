if (CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)

check_cxx_compiler_flag("-march=armv8.2-a+dotprod" HAVE_FLAG_MARCH_ARMV8_2_A_DOTPROD)
if(HAVE_FLAG_MARCH_ARMV8_2_A_DOTPROD)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_dotprod
    SRCS
      "mmt4d_tile_arm_64_dotprod.S"
    COPTS
      "-march=armv8.2-a+dotprod"
  )
  list(APPEND MMT4D_ARM_64_VARIANT_DEPS "iree::builtins::ukernel::arm_64::mmt4d_tile_arm_64_dotprod")
endif()

check_cxx_compiler_flag("-march=armv8.2-a+i8mm" HAVE_FLAG_MARCH_ARMV8_2_A_I8MM)
if(HAVE_FLAG_MARCH_ARMV8_2_A_I8MM)
  iree_cc_library(
    NAME
      mmt4d_tile_arm_64_i8mm
    SRCS
      "mmt4d_tile_arm_64_i8mm.S"
    COPTS
      "-march=armv8.2-a+i8mm"
  )
  list(APPEND MMT4D_ARM_64_VARIANT_DEPS "iree::builtins::ukernel::arm_64::mmt4d_tile_arm_64_i8mm")
endif()

configure_file(config.h.in config.h)

iree_cc_library(
  NAME
    mmt4d_tile_arm_64
  HDRS
    "mmt4d_tile_arm_64.h"
  SRCS
    "mmt4d_tile_arm_64.c"
  DEPS
    iree::base::core_headers
    iree::schemas::cpu_data
    iree::builtins::ukernel::types
    ${MMT4D_ARM_64_VARIANT_DEPS}
  PUBLIC
)

else()  # Not CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64

iree_cc_library(
  NAME
    mmt4d_tile_arm_64
  HDRS
    "mmt4d_tile_arm_64.h"
)

endif()
