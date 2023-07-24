// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64_entry_point.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_8x8x8(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_I8MM
  if (iree_uk_cpu_supports_i8mm(params->cpu_data)) {
    return IREE_UK_SELECT_INLINE_ASM_OR_INTRINSICS(
        iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm_inline_asm,
        iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm_intrinsics,
        params->flags & IREE_UK_FLAG_MMT4D_PREFER_INTRINSICS);
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
  if (iree_uk_cpu_supports_dotprod(params->cpu_data)) {
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

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f16(
    const iree_uk_mmt4d_params_t* params) {
  if ((params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) &&
      params->M0 == 8 && params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16f32(
    const iree_uk_mmt4d_params_t* params) {
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16bf16(
    const iree_uk_mmt4d_params_t* params) {
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

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  switch (iree_uk_mmt4d_type(params->flags)) {
    case iree_uk_mmt4d_type_f32f32f32:
      return iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32(params);
    case iree_uk_mmt4d_type_f16f16f32:
      return iree_uk_mmt4d_select_tile_func_arm_64_f16f16f32(params);
    case iree_uk_mmt4d_type_f16f16f16:
      return iree_uk_mmt4d_select_tile_func_arm_64_f16f16f16(params);
    case iree_uk_mmt4d_type_bf16bf16f32:
      return iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16f32(params);
    case iree_uk_mmt4d_type_bf16bf16bf16:
      return iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16bf16(params);
    case iree_uk_mmt4d_type_i8i8i32:
      return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32(params);
    default:
      IREE_UK_ASSUME_UNREACHABLE;
      return 0;
  }
}
