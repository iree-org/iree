// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/x86_64/common_x86_64.h"
#include "iree/builtins/ukernel/arch/x86_64/conv_nchwc_x86_64_internal.h"

// AVX-512 tile kernel for data-tiled NCHWc conv (f32 x f32 -> f32).
//
// Operand layouts:
//   input  [N, IC/c0, H, W, c0]
//   filter [OC/k0, IC/c0, FH, FW, c0, k0]
//   output [N, OC/k0, OH, OW, k0]
//
// Computes one output panel of shape [OW_TILE=16, k0=16] per call.
//
// Strategy:
//   - 16 ZMM accumulators, one per output-width position, each holding a
//   k0-wide OC block.
//   - For each (ic_outer, fh, fw, ci):
//       load one ZMM containing k0 output-channel values for fixed (ic_outer,
//       fh, fw, ci), broadcast scalar input value at (ow, fw, ci) and FMA into
//       each OW accumulator.
//
// Tile shapes: OW_TILE=16, k0=16, c0=16.
#define OW_TILE 16
#define K0 16
#define C0 16

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_conv_nchwc_tile_f32f32f32_16x16x16_x86_64_avx512_base_impl(
    float* IREE_UK_RESTRICT out_ptr, const float* IREE_UK_RESTRICT in_ptr,
    const float* IREE_UK_RESTRICT f_ptr,
    const iree_uk_conv_nchwc_params_t* params, iree_uk_index_t ow_count) {
  const iree_uk_index_t IC_outer = params->IC_outer;
  const iree_uk_index_t FH = params->FH;
  const iree_uk_index_t FW = params->FW;
  const iree_uk_index_t in_pos_stride = (iree_uk_index_t)params->stride_w * C0;
  const iree_uk_index_t in_stride_h = params->input_stride_h;
  const iree_uk_index_t in_stride_ic_outer = params->input_stride_ic_outer;
  const iree_uk_index_t f_stride_fw = params->filter_stride_fw;
  const iree_uk_index_t f_stride_fh = params->filter_stride_fh;
  const iree_uk_index_t f_stride_ic_outer = params->filter_stride_ic_outer;

  __m512 acc[OW_TILE];
  if (params->flags & IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
      if (i < ow_count) {
        acc[i] = _mm512_loadu_ps(out_ptr + i * K0);
      } else {
        acc[i] = _mm512_setzero_ps();
      }
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
      acc[i] = _mm512_setzero_ps();
    }
  }

  for (iree_uk_index_t ic_o = 0; ic_o < IC_outer; ++ic_o) {
    const float* in_ic = in_ptr + ic_o * in_stride_ic_outer;
    const float* f_ic = f_ptr + ic_o * f_stride_ic_outer;

    for (iree_uk_index_t fh = 0; fh < FH; ++fh) {
      const float* in_h = in_ic + fh * in_stride_h;
      const float* f_h = f_ic + fh * f_stride_fh;

      for (iree_uk_index_t fw = 0; fw < FW; ++fw) {
        const float* f_fw = f_h + fw * f_stride_fw;
        const float* in_fw = in_h + fw * C0;
        for (int ci = 0; ci < C0; ++ci) {
          // Load filter[ci, 0:k0].
          __m512 f_vec = _mm512_loadu_ps(f_fw + ci * K0);

          // Broadcast input[..., ci] for each output position and FMA.
          IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
            if (i < ow_count) {
              __m512 in_bc = _mm512_set1_ps(in_fw[i * in_pos_stride + ci]);
              acc[i] = _mm512_fmadd_ps(in_bc, f_vec, acc[i]);
            }
          }
        }
      }
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
    if (i < ow_count) {
      _mm512_storeu_ps(out_ptr + i * K0, acc[i]);
    }
  }
}

void iree_uk_conv_nchwc_tile_f32f32f32_16x16x16_x86_64_avx512_base(
    void* IREE_UK_RESTRICT output_panel,
    const void* IREE_UK_RESTRICT input_panel,
    const void* IREE_UK_RESTRICT filter_panel,
    const iree_uk_conv_nchwc_params_t* params) {
  IREE_UK_ASSERT(params->ow_count > 0 && params->ow_count <= OW_TILE);
  float* IREE_UK_RESTRICT out_ptr = (float*)output_panel;
  const float* IREE_UK_RESTRICT in_ptr = (const float*)input_panel;
  const float* IREE_UK_RESTRICT f_ptr = (const float*)filter_panel;
  // TODO(phemashekar): adopt the "always pad" rule (ow_count == OW_TILE always)
  // to delete the tail path entirely.
  if (params->ow_count == OW_TILE) {
    iree_uk_conv_nchwc_tile_f32f32f32_16x16x16_x86_64_avx512_base_impl(
        out_ptr, in_ptr, f_ptr, params, OW_TILE);
  } else {
    iree_uk_conv_nchwc_tile_f32f32f32_16x16x16_x86_64_avx512_base_impl(
        out_ptr, in_ptr, f_ptr, params, params->ow_count);
  }
}

#undef OW_TILE
#undef K0
#undef C0
