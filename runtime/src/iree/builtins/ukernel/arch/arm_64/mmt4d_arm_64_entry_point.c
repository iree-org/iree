// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64_entry_point.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32_M0x8x1(
    const iree_uk_mmt4d_params_t* params) {
  switch (params->M0) {
    case 1:
      return iree_uk_mmt4d_tile_f32f32f32_1x8x1_arm_64;
    case 2:
      return iree_uk_mmt4d_tile_f32f32f32_2x8x1_arm_64;
    case 4:
      return iree_uk_mmt4d_tile_f32f32f32_4x8x1_arm_64;
    case 8:
      return iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_select_tile_func_arm_64_f32f32f32_M0x8x1(params);
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f32_M0x8x1(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_FP16FML
  if (iree_uk_cpu_supports_fp16fml(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_f16f16f32_1x8x1_arm_64_fp16fml;
      case 2:
        return iree_uk_mmt4d_tile_f16f16f32_2x8x1_arm_64_fp16fml;
      case 4:
        return iree_uk_mmt4d_tile_f16f16f32_4x8x1_arm_64_fp16fml;
      case 8:
        return iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64_fp16fml;
    }
  }
#endif
  switch (params->M0) {
    case 1:
      return iree_uk_mmt4d_tile_f16f16f32_1x8x1_arm_64;
    case 2:
      return iree_uk_mmt4d_tile_f16f16f32_2x8x1_arm_64;
    case 4:
      return iree_uk_mmt4d_tile_f16f16f32_4x8x1_arm_64;
    case 8:
      return iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_select_tile_func_arm_64_f16f16f32_M0x8x1(params);
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f16_M0x8x1(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_FULLFP16
  if (iree_uk_cpu_supports_fp16(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_f16f16f16_1x8x1_arm_64_fullfp16;
      case 2:
        return iree_uk_mmt4d_tile_f16f16f16_2x8x1_arm_64_fullfp16;
      case 4:
        return iree_uk_mmt4d_tile_f16f16f16_4x8x1_arm_64_fullfp16;
      case 8:
        return iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64_fullfp16;
    }
  }
#endif
  if (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_f16f16f16_1x8x1_arm_64;
      case 2:
        return iree_uk_mmt4d_tile_f16f16f16_2x8x1_arm_64;
      case 4:
        return iree_uk_mmt4d_tile_f16f16f16_4x8x1_arm_64;
      case 8:
        return iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64;
    }
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_f16f16f16(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_select_tile_func_arm_64_f16f16f16_M0x8x1(params);
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16f32_M0x8x4(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_BF16
  if (iree_uk_cpu_supports_bf16(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_bf16bf16f32_1x8x4_arm_64_bf16;
      case 2:
        return iree_uk_mmt4d_tile_bf16bf16f32_2x8x4_arm_64_bf16;
      case 4:
        return iree_uk_mmt4d_tile_bf16bf16f32_4x8x4_arm_64_bf16;
      case 8:
        return iree_uk_mmt4d_tile_bf16bf16f32_8x8x4_arm_64_bf16;
    }
  }
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16f32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 4) {
    return iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16f32_M0x8x4(params);
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16bf16_M0x8x4(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_BF16
  if (iree_uk_cpu_supports_bf16(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_bf16bf16bf16_1x8x4_arm_64_bf16;
      case 2:
        return iree_uk_mmt4d_tile_bf16bf16bf16_2x8x4_arm_64_bf16;
      case 4:
        return iree_uk_mmt4d_tile_bf16bf16bf16_4x8x4_arm_64_bf16;
      case 8:
        return iree_uk_mmt4d_tile_bf16bf16bf16_8x8x4_arm_64_bf16;
    }
  }
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16bf16(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 4) {
    return iree_uk_mmt4d_select_tile_func_arm_64_bf16bf16bf16_M0x8x4(params);
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x1(
    const iree_uk_mmt4d_params_t* params) {
  switch (params->M0) {
    case 1:
      return iree_uk_mmt4d_tile_s8s8s32_1x8x1_arm_64;
    case 2:
      return iree_uk_mmt4d_tile_s8s8s32_2x8x1_arm_64;
    case 4:
      return iree_uk_mmt4d_tile_s8s8s32_4x8x1_arm_64;
    case 8:
      return iree_uk_mmt4d_tile_s8s8s32_8x8x1_arm_64;
  }
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x4(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_DOTPROD
  if (iree_uk_cpu_supports_dotprod(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_s8s8s32_1x8x4_arm_64_dotprod;
      case 2:
        return iree_uk_mmt4d_tile_s8s8s32_2x8x4_arm_64_dotprod;
      case 4:
        return iree_uk_mmt4d_tile_s8s8s32_4x8x4_arm_64_dotprod;
      case 8:
        return iree_uk_mmt4d_tile_s8s8s32_8x8x4_arm_64_dotprod;
    }
  }
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t
iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x8(
    const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_BUILD_ARM_64_I8MM
  if (iree_uk_cpu_supports_i8mm(params->cpu_data)) {
    switch (params->M0) {
      case 1:
        return iree_uk_mmt4d_tile_s8s8s32_1x8x8_arm_64_i8mm;
      case 2:
        return iree_uk_mmt4d_tile_s8s8s32_2x8x8_arm_64_i8mm;
      case 4:
        return iree_uk_mmt4d_tile_s8s8s32_4x8x8_arm_64_i8mm;
      case 8:
        return iree_uk_mmt4d_tile_s8s8s32_8x8x8_arm_64_i8mm;
    }
  }
#endif
  return 0;
}

static iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32(
    const iree_uk_mmt4d_params_t* params) {
  if (params->N0 == 8 && params->K0 == 1) {
    return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x1(params);
  }
  if (params->N0 == 8 && params->K0 == 4) {
    return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x4(params);
  }
  if (params->N0 == 8 && params->K0 == 8) {
    return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32_M0x8x8(params);
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
    case iree_uk_mmt4d_type_s8s8s32:
      return iree_uk_mmt4d_select_tile_func_arm_64_i8i8i32(params);
    case iree_uk_mmt4d_type_s16s16s32:
      return 0;
    case iree_uk_mmt4d_type_s16u4s32:
      return 0;
    case iree_uk_mmt4d_type_s16s8s32:
      return 0;
    default:
      IREE_UK_ASSUME_UNREACHABLE;
      return 0;
  }
}
