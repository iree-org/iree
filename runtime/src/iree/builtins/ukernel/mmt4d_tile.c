// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_internal.h"

// Generic implementation of matmul tile, i8*i8->i32 case.
static void iree_uk_mmt4d_tile_i8i8i32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, iree_uk_int32_t K, iree_uk_uint32_t flags,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int8_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  iree_uk_int32_t acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_ssize_t k = 0; k < K; ++k) {
    for (iree_uk_ssize_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_ssize_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_ssize_t k0 = 0; k0 < K0; ++k0) {
          iree_uk_int32_t lhs_val_int32 = lhs_panel[i0 * K0 + k0];
          iree_uk_int32_t rhs_val_int32 = rhs_panel[j0 * K0 + k0];
          acc[i0 * N0 + j0] += lhs_val_int32 * rhs_val_int32;
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
    const void* rhs_panel_untyped, iree_uk_int32_t K, iree_uk_uint32_t flags,
    const iree_uk_mmt4d_params_t* params) {
  float* out_tile = out_tile_untyped;
  const float* lhs_panel = lhs_panel_untyped;
  const float* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  float acc[iree_uk_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_uk_ssize_t k = 0; k < K; ++k) {
    for (iree_uk_ssize_t i0 = 0; i0 < M0; ++i0) {
      for (iree_uk_ssize_t j0 = 0; j0 < N0; ++j0) {
        for (iree_uk_ssize_t k0 = 0; k0 < K0; ++k0) {
          float lhs_val = lhs_panel[i0 * K0 + k0];
          float rhs_val = rhs_panel[j0 * K0 + k0];
          acc[i0 * N0 + j0] += lhs_val * rhs_val;
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
    case iree_uk_mmt4d_type_i8i8i32:
      return iree_uk_mmt4d_tile_i8i8i32_generic;
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
