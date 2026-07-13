// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/conv_nchwc_internal.h"
#include "iree/builtins/ukernel/exported_bits.h"

// Generic scalar implementation of the conv_nchwc tile function for f32.
//
// Computes one output panel of shape [ow_count, k0] by iterating over
// (ic_outer, fh, fw, c_inner) and accumulating. Used as a reference kernel
// for correctness checking and as a portable fallback when no ISA-specific
// tile function is available.
static void iree_uk_conv_nchwc_tile_f32f32f32_generic(
    void* IREE_UK_RESTRICT output_panel,
    const void* IREE_UK_RESTRICT input_panel,
    const void* IREE_UK_RESTRICT filter_panel,
    const iree_uk_conv_nchwc_params_t* params) {
  float* out_ptr = (float*)output_panel;
  const float* in_ptr = (const float*)input_panel;
  const float* f_ptr = (const float*)filter_panel;

  const iree_uk_index_t ow_count = params->ow_count;
  const iree_uk_index_t IC_outer = params->IC_outer;
  const iree_uk_index_t FH = params->FH;
  const iree_uk_index_t FW = params->FW;
  const iree_uk_int32_t k0 = params->k0;
  const iree_uk_int32_t c0 = params->c0;
  const iree_uk_int32_t stride_w = params->stride_w;

  const iree_uk_index_t in_stride_w = c0;
  const iree_uk_index_t in_stride_h = params->input_stride_h;
  const iree_uk_index_t in_stride_ic_outer = params->input_stride_ic_outer;

  const iree_uk_index_t f_stride_fw = params->filter_stride_fw;
  const iree_uk_index_t f_stride_fh = params->filter_stride_fh;
  const iree_uk_index_t f_stride_ic_outer = params->filter_stride_ic_outer;

  for (iree_uk_index_t i = 0; i < ow_count; ++i) {
    for (iree_uk_int32_t ko = 0; ko < k0; ++ko) {
      float acc = (params->flags & IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE)
                      ? out_ptr[i * k0 + ko]
                      : 0.f;
      for (iree_uk_index_t ic_o = 0; ic_o < IC_outer; ++ic_o) {
        for (iree_uk_index_t fh = 0; fh < FH; ++fh) {
          for (iree_uk_index_t fw = 0; fw < FW; ++fw) {
            for (iree_uk_int32_t ci = 0; ci < c0; ++ci) {
              float in_val =
                  in_ptr[ic_o * in_stride_ic_outer + fh * in_stride_h +
                         (i * stride_w + fw) * in_stride_w + ci];
              float f_val = f_ptr[ic_o * f_stride_ic_outer + fh * f_stride_fh +
                                  fw * f_stride_fw + ci * k0 + ko];
              acc += in_val * f_val;
            }
          }
        }
      }
      out_ptr[i * k0 + ko] = acc;
    }
  }
}

iree_uk_conv_nchwc_tile_selection_t iree_uk_conv_nchwc_select_tile_func_generic(
    const iree_uk_conv_nchwc_params_t* params) {
  iree_uk_conv_nchwc_tile_selection_t selection = {0};
  switch (iree_uk_conv_nchwc_type(params->flags)) {
    case iree_uk_conv_nchwc_type_f32f32f32:
      selection.tile_func = iree_uk_conv_nchwc_tile_f32f32f32_generic;
      selection.ow_tile = 16;
      break;
    default:
      break;
  }
  return selection;
}
