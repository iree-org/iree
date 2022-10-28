// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_select_tile_generic.h"

// In order to be helpful as a reference for future architecture-specific
// kernels, the generic kernels here are structured like an actual optimized
// kernel, using an "accumulator tile" that in this case is a stack array
// (which would become a group of SIMD registers in an actual optimized kernel).
// The downside of this approach is that we have to set a fixed max size for
// the accumulator tile, but for now all known cases are comfortably far below
// where trouble would happen. For reference:
// - On ARM NEON, the entire register space is 512 bytes, so the accumulator
//   tile is less than that, typically 256 to 384 bytes.
// - On ARM SME, we will be working with an accumulator tile as large as 4096
//   bytes (IIUC).
// - The smallest stack frame size limit that we know we may have to deal with
//   on certain targets is 16 kilobytes.
// The size or architecture-specific tiles is relevant here because this
// generic code is what will be run as a fallback if the device is found not to
// support the CPU feature that the tile sizes were picked to target.
enum { iree_ukernel_mmt4d_tile_generic_max_bytes = 4096 };

// Generic implementation of matmul tile, i8*i8->i32 case.
static void iree_ukernel_mmt4d_tile_i8i8i32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, int32_t K, uint32_t flags,
    const iree_ukernel_mmt4d_params_t* params) {
  int32_t* out_tile = out_tile_untyped;
  const int8_t* lhs_panel = lhs_panel_untyped;
  const int8_t* rhs_panel = rhs_panel_untyped;
  int16_t M0 = params->M0;
  int16_t N0 = params->N0;
  int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  int32_t acc[iree_ukernel_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (flags & IREE_VMVX_MATMUL_FLAG_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_ukernel_ssize_t k = 0; k < K; ++k) {
    for (iree_ukernel_ssize_t i0 = 0; i0 < M0; ++i0) {
      for (iree_ukernel_ssize_t j0 = 0; j0 < N0; ++j0) {
        for (iree_ukernel_ssize_t k0 = 0; k0 < K0; ++k0) {
          int32_t lhs_val_int32 = lhs_panel[i0 * K0 + k0];
          int32_t rhs_val_int32 = rhs_panel[j0 * K0 + k0];
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
static void iree_ukernel_mmt4d_tile_f32f32f32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, int32_t K, uint32_t flags,
    const iree_ukernel_mmt4d_params_t* params) {
  float* out_tile = out_tile_untyped;
  const float* lhs_panel = lhs_panel_untyped;
  const float* rhs_panel = rhs_panel_untyped;
  int16_t M0 = params->M0;
  int16_t N0 = params->N0;
  int16_t K0 = params->K0;
  // Initialize the local accumulator tile.
  float acc[iree_ukernel_mmt4d_tile_generic_max_bytes / sizeof(*out_tile)];
  if (flags & IREE_VMVX_MATMUL_FLAG_ACCUMULATE) {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = out_tile[i];
  } else {
    for (int i = 0; i < M0 * N0; ++i) acc[i] = 0;
  }
  // Accumulation loop.
  for (iree_ukernel_ssize_t k = 0; k < K; ++k) {
    for (iree_ukernel_ssize_t i0 = 0; i0 < M0; ++i0) {
      for (iree_ukernel_ssize_t j0 = 0; j0 < N0; ++j0) {
        for (iree_ukernel_ssize_t k0 = 0; k0 < K0; ++k0) {
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

// Generic implementation of matmul tile
iree_ukernel_mmt4d_status_t iree_ukernel_mmt4d_select_tile_func_generic(
    const iree_ukernel_mmt4d_params_t* params,
    iree_ukernel_mmt4d_tile_func_t* out_tile_func) {
  int tile_elems = params->M0 * params->N0;
  int tile_bytes = tile_elems
                   << iree_ukernel_mmt4d_out_elem_size_log2(params->type);
  if (tile_bytes > iree_ukernel_mmt4d_tile_generic_max_bytes) {
    return iree_ukernel_mmt4d_status_unsupported_generic_tile_size;
  }
  switch (params->type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
      *out_tile_func = iree_ukernel_mmt4d_tile_f32f32f32_generic;
      return iree_ukernel_mmt4d_status_ok;
    case iree_ukernel_mmt4d_type_i8i8i32:
      *out_tile_func = iree_ukernel_mmt4d_tile_i8i8i32_generic;
      return iree_ukernel_mmt4d_status_ok;
    default:
      // shouldn't happen, validated earlier.
      return iree_ukernel_mmt4d_status_bad_type;
  }
}
