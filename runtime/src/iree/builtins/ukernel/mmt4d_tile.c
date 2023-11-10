// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_internal.h"

// Generic implementation of matmul tile, i8*i8->i32 case.
static void iree_uk_mmt4d_tile_s8s8s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int8_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  iree_uk_int32_t acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          iree_uk_int32_t lhs_i32 = lhs_panel[i0 * K0 + k0];
          iree_uk_int32_t rhs_i32 = rhs_panel[j0 * K0 + k0];
          acc[i0 * N0 + j0] += lhs_i32 * rhs_i32;
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

// Generic implementation of matmul tile, f32*f32->f32 case.
static void iree_uk_mmt4d_tile_f32f32f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  float* out_tile = out_tile_untyped;
  const float* lhs_panel = lhs_panel_untyped;
  const float* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  float acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = lhs_panel[i0 * K0 + k0];
          float rhs_f32 = rhs_panel[j0 * K0 + k0];
          acc[i0 * N0 + j0] += lhs_f32 * rhs_f32;
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

// Generic implementation of matmul tile, f16*f16->f32 case.
static void iree_uk_mmt4d_tile_f16f16f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  float* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  float acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = iree_uk_f16_to_f32(lhs_panel[i0 * K0 + k0]);
          float rhs_f32 = iree_uk_f16_to_f32(rhs_panel[j0 * K0 + k0]);
          acc[i0 * N0 + j0] += lhs_f32 * rhs_f32;
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

// Generic implementation of matmul tile, f16*f16->f16 case.
static void iree_uk_mmt4d_tile_f16f16f16_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  iree_uk_int16_t acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = iree_uk_f16_to_f32(lhs_panel[i0 * K0 + k0]);
          float rhs_f32 = iree_uk_f16_to_f32(rhs_panel[j0 * K0 + k0]);
          iree_uk_int16_t* acc_ptr = &acc[i0 * N0 + j0];
          float acc_f32 = iree_uk_f16_to_f32(*acc_ptr);
          *acc_ptr = iree_uk_f32_to_f16(acc_f32 + lhs_f32 * rhs_f32);
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

// Generic implementation of matmul tile, bf16*bf16->f32 case.
static void iree_uk_mmt4d_tile_bf16bf16f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  float* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  float acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = iree_uk_bf16_to_f32(lhs_panel[i0 * K0 + k0]);
          float rhs_f32 = iree_uk_bf16_to_f32(rhs_panel[j0 * K0 + k0]);
          acc[i0 * N0 + j0] += lhs_f32 * rhs_f32;
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

// Generic implementation of matmul tile, bf16*bf16->bf16 case.
static void iree_uk_mmt4d_tile_bf16bf16bf16_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  iree_uk_int16_t acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = iree_uk_bf16_to_f32(lhs_panel[i0 * K0 + k0]);
          float rhs_f32 = iree_uk_bf16_to_f32(rhs_panel[j0 * K0 + k0]);
          iree_uk_int16_t* acc_ptr = &acc[i0 * N0 + j0];
          float acc_f32 = iree_uk_bf16_to_f32(*acc_ptr);
          *acc_ptr = iree_uk_f32_to_bf16(acc_f32 + lhs_f32 * rhs_f32);
        }
      }
    }
    lhs_panel += M0 * K0;
    rhs_panel += N0 * K0;
  }
  // Store the local accumulator tile to the destination.
  for (int i = 0; i < M0 * N0; ++i) out_tile[i] = acc[i];
}

static iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_generic(
    const iree_uk_mmt4d_params_t* params) {
  switch (iree_uk_mmt4d_type(params->flags)) {
    case iree_uk_mmt4d_type_f32f32f32:
      return iree_uk_mmt4d_tile_f32f32f32_generic;
    case iree_uk_mmt4d_type_s8s8s32:
      return iree_uk_mmt4d_tile_s8s8s32_generic;
    case iree_uk_mmt4d_type_f16f16f32:
      return iree_uk_mmt4d_tile_f16f16f32_generic;
    case iree_uk_mmt4d_type_f16f16f16:
      return iree_uk_mmt4d_tile_f16f16f16_generic;
    case iree_uk_mmt4d_type_bf16bf16f32:
      return iree_uk_mmt4d_tile_bf16bf16f32_generic;
    case iree_uk_mmt4d_type_bf16bf16bf16:
      return iree_uk_mmt4d_tile_bf16bf16bf16_generic;
    default:
      // shouldn't happen, validated earlier.
      IREE_UK_ASSUME_UNREACHABLE;
      return 0;
  }
}

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func(
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_tile_func_t arch_tile_func =
      iree_uk_mmt4d_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  return iree_uk_mmt4d_select_tile_func_generic(params);
}
