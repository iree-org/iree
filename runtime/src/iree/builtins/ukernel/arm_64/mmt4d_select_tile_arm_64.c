// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arm_64/mmt4d_select_tile_arm_64.h"

#include "iree/builtins/ukernel/arm_64/config.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_UKERNEL_ARCH_ARM_64)

IREE_UKERNEL_MMT4D_TILE_FUNC_DECL(
    iree_ukernel_mmt4d_f32f32f32_tile_8x8x1_arm_64)
IREE_UKERNEL_MMT4D_TILE_FUNC_DECL(iree_ukernel_mmt4d_i8i8i32_tile_8x8x1_arm_64)
IREE_UKERNEL_MMT4D_TILE_FUNC_DECL(
    iree_ukernel_mmt4d_i8i8i32_tile_8x8x4_arm_64_dotprod)
IREE_UKERNEL_MMT4D_TILE_FUNC_DECL(
    iree_ukernel_mmt4d_i8i8i32_tile_8x8x8_arm_64_i8mm)

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_f32f32f32_8x8x1(
    const iree_ukernel_mmt4d_params_t* params) {
  (void)params;
  return iree_ukernel_mmt4d_f32f32f32_tile_8x8x1_arm_64;
}

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x1(
    const iree_ukernel_mmt4d_params_t* params) {
  (void)params;
  return iree_ukernel_mmt4d_i8i8i32_tile_8x8x1_arm_64;
}

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x8(
    const iree_ukernel_mmt4d_params_t* params) {
#ifdef HAVE_FLAG_MARCH_ARMV8_2_A_I8MM
  if (params->cpu_data_field_0 & IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_I8MM) {
    return iree_ukernel_mmt4d_i8i8i32_tile_8x8x8_arm_64_i8mm;
  }
#else
  (void)params;
#endif
  return 0;
}

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x4(
    const iree_ukernel_mmt4d_params_t* params) {
#ifdef HAVE_FLAG_MARCH_ARMV8_2_A_DOTPROD
  if (params->cpu_data_field_0 & IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_DOTPROD) {
    return iree_ukernel_mmt4d_i8i8i32_tile_8x8x4_arm_64_dotprod;
  }
#else
  (void)params;
#endif
  return 0;
}

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_f32f32f32(
    const iree_ukernel_mmt4d_params_t* params) {
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_ukernel_mmt4d_select_tile_func_arm_64_f32f32f32_8x8x1(params);
  }
  return 0;
}

static iree_ukernel_mmt4d_tile_func_t
iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32(
    const iree_ukernel_mmt4d_params_t* params) {
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x1(params);
  }
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 4) {
    return iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x4(params);
  }
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 8) {
    return iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x8(params);
  }
  return 0;
}

iree_ukernel_mmt4d_tile_func_t iree_ukernel_mmt4d_select_tile_func_arm_64(
    const iree_ukernel_mmt4d_params_t* params) {
  switch (params->type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
      return iree_ukernel_mmt4d_select_tile_func_arm_64_f32f32f32(params);
    case iree_ukernel_mmt4d_type_i8i8i32:
      return iree_ukernel_mmt4d_select_tile_func_arm_64_i8i8i32(params);
    default:
      return 0;
  }
}

#endif  // IREE_UKERNEL_ARCH_ARM_64
