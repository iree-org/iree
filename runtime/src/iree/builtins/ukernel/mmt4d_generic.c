// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_generic.h"

#include <stdbool.h>

int iree_ukernel_mmt4d_f32f32f32_generic(
    const iree_ukernel_mmt4d_f32f32f32_params_t* params) {
  bool accumulate = params->flags & IREE_VMVX_MATMUL_FLAG_ACCUMULATE;
  iree_ukernel_size_t lhs_tile_size = params->M0 * params->K0;
  iree_ukernel_size_t rhs_tile_size = params->N0 * params->K0;
  iree_ukernel_size_t out_tile_size = params->M0 * params->N0;
  for (iree_ukernel_size_t i = 0; i < params->M; ++i) {
    for (iree_ukernel_size_t j = 0; j < params->N; ++j) {
      float* out_tile_ptr =
          params->out_buffer + i * params->out_stride + j * out_tile_size;
      const float* lhs_panel_ptr = params->lhs_buffer + i * params->lhs_stride;
      const float* rhs_panel_ptr = params->rhs_buffer + j * params->rhs_stride;
      for (iree_ukernel_size_t i0 = 0; i0 < params->M0; ++i0) {
        for (iree_ukernel_size_t j0 = 0; j0 < params->N0; ++j0) {
          const float* lhs_tile_ptr = lhs_panel_ptr;
          const float* rhs_tile_ptr = rhs_panel_ptr;
          float* out_ptr = out_tile_ptr + i0 * params->N0 + j0;
          float acc = accumulate ? *out_ptr : 0.f;
          for (iree_ukernel_size_t k = 0; k < params->K; ++k) {
            for (iree_ukernel_size_t k0 = 0; k0 < params->K0; ++k0) {
              float lhs_val = lhs_tile_ptr[i0 * params->K0 + k0];
              float rhs_val = rhs_tile_ptr[j0 * params->K0 + k0];
              acc += lhs_val * rhs_val;
            }
            lhs_tile_ptr += lhs_tile_size;
            rhs_tile_ptr += rhs_tile_size;
          }
          *out_ptr = acc;
        }
      }
    }
  }
  return 0;
}

int iree_ukernel_mmt4d_i8i8i32_generic(
    const iree_ukernel_mmt4d_i8i8i32_params_t* params) {
  bool accumulate = params->flags & IREE_VMVX_MATMUL_FLAG_ACCUMULATE;
  iree_ukernel_size_t lhs_tile_size = params->M0 * params->K0;
  iree_ukernel_size_t rhs_tile_size = params->N0 * params->K0;
  iree_ukernel_size_t out_tile_size = params->M0 * params->N0;
  for (iree_ukernel_size_t i = 0; i < params->M; ++i) {
    for (iree_ukernel_size_t j = 0; j < params->N; ++j) {
      int32_t* out_tile_ptr =
          params->out_buffer + i * params->out_stride + j * out_tile_size;
      const int8_t* lhs_panel_ptr = params->lhs_buffer + i * params->lhs_stride;
      const int8_t* rhs_panel_ptr = params->rhs_buffer + j * params->rhs_stride;
      for (iree_ukernel_size_t i0 = 0; i0 < params->M0; ++i0) {
        for (iree_ukernel_size_t j0 = 0; j0 < params->N0; ++j0) {
          const int8_t* lhs_tile_ptr = lhs_panel_ptr;
          const int8_t* rhs_tile_ptr = rhs_panel_ptr;
          int32_t* out_ptr = out_tile_ptr + i0 * params->N0 + j0;
          int32_t acc = accumulate ? *out_ptr : 0;
          for (iree_ukernel_size_t k = 0; k < params->K; ++k) {
            for (iree_ukernel_size_t k0 = 0; k0 < params->K0; ++k0) {
              // C's implicit promotion to int saves skin, but let's be explicit
              int32_t lhs_val_int32 = lhs_tile_ptr[i0 * params->K0 + k0];
              int32_t rhs_val_int32 = rhs_tile_ptr[j0 * params->K0 + k0];
              acc += lhs_val_int32 * rhs_val_int32;
            }
            lhs_tile_ptr += lhs_tile_size;
            rhs_tile_ptr += rhs_tile_size;
          }
          *out_ptr = acc;
        }
      }
    }
  }
  return 0;
}
