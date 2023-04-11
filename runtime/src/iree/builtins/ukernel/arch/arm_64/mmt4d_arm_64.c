// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64.h"

#include "iree/schemas/cpu_data.h"

IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x1_arm_64)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod)
IREE_UK_MMT4D_TILE_FUNC_DECL(iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm)

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x8(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_I8MM
  if (params->cpu_data[0] & IREE_CPU_DATA0_ARM_64_I8MM) {
    return iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm;
  }
#else
  (void)params;
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x4(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_DOTPROD
  if (params->cpu_data[0] & IREE_CPU_DATA0_ARM_64_DOTPROD) {
    return iree_uk_mmt4d_tile_i8i8i32_8x8x4_arm_64_dotprod;
  }
#else
  (void)params;
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_tile_i8i8i32_8x8x1_arm_64;
  }
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 4) {
    return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x4(params);
  }
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 8) {
    return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x8(params);
  }
  return 0;
}

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arm_64(
    const iree_uk_mmt4d_params_t* params) {
  switch (params->type) {
    case iree_uk_mmt4d_type_f32f32f32:
      return iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32(params);
    case iree_uk_mmt4d_type_i8i8i32:
      return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32(params);
    default:
      IREE_UK_ASSUME_UNREACHABLE;
      return 0;
  }
}
