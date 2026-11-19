// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/conv_nchwc_arm_64_internal.h"

// Baseline NEON tile kernel for data-tiled NCHWc convolution (f32).
//
// Operand layouts:
//   input  [N, IC/c0, H, W, c0]
//   filter [OC/k0, IC/c0, FH, FW, c0, k0]
//   output [N, OC/k0, OH, OW, k0]
//
// Computes an output tile of [OW_TILE=8, k0=8].
//
// Uses 16 Q-register accumulators (two per output-width position).
// Processes c0 in chunks of 4 to use lane-based FMLAs, avoiding
// scalar broadcasts.
//
// Tile sizes: OW_TILE=8, k0=8, c0=8.
#define OW_TILE 8
#define K0 8
#define C0 8

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_conv_nchwc_tile_f32f32f32_8x8x8_arm_64_impl(
    float* IREE_UK_RESTRICT out_ptr, const float* IREE_UK_RESTRICT in_ptr,
    const float* IREE_UK_RESTRICT f_ptr,
    const iree_uk_conv_nchwc_params_t* params, iree_uk_index_t ow_count) {
  const iree_uk_index_t IC_outer = params->IC_outer;
  const iree_uk_index_t FH = params->FH;
  const iree_uk_index_t FW = params->FW;
  const iree_uk_index_t in_pos_stride = (iree_uk_index_t)params->stride_w * C0;

  const iree_uk_index_t in_stride_h = params->input_stride_h;
  const iree_uk_index_t in_stride_ic_outer = params->input_stride_ic_outer;
  const iree_uk_index_t f_stride_fw = (iree_uk_index_t)C0 * K0;
  const iree_uk_index_t f_stride_fh = params->filter_stride_fh;
  const iree_uk_index_t f_stride_ic_outer = params->filter_stride_ic_outer;

  float32x4_t acc[OW_TILE][2];

  if (params->flags & IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE) {
    IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
      if (i < ow_count) {
        acc[i][0] = vld1q_f32(out_ptr + i * K0 + 0);
        acc[i][1] = vld1q_f32(out_ptr + i * K0 + 4);
      } else {
        acc[i][0] = vdupq_n_f32(0.0f);
        acc[i][1] = vdupq_n_f32(0.0f);
      }
    }
  } else {
    IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
      acc[i][0] = vdupq_n_f32(0.0f);
      acc[i][1] = vdupq_n_f32(0.0f);
    }
  }

  for (iree_uk_index_t ic_o = 0; ic_o < IC_outer; ++ic_o) {
    const float* in_ic = in_ptr + ic_o * in_stride_ic_outer;
    const float* f_ic = f_ptr + ic_o * f_stride_ic_outer;

    for (iree_uk_index_t fh = 0; fh < FH; ++fh) {
      const float* in_h = in_ic + fh * in_stride_h;
      const float* f_h = f_ic + fh * f_stride_fh;

      for (iree_uk_index_t fw = 0; fw < FW; ++fw) {
        const float* in_fw = in_h + fw * C0;
        const float* f_fw = f_h + fw * f_stride_fw;

        // Keep this loop rolled: unrolling it doubles the live filter
        // registers and makes the compiler spill the accumulators.
        IREE_UK_UNROLL_DISABLE for (int ci_chunk = 0; ci_chunk < C0 / 4;
                                    ++ci_chunk) {
          const int ci_base = ci_chunk * 4;

          float32x4_t f0_lo = vld1q_f32(f_fw + (ci_base + 0) * K0 + 0);
          float32x4_t f0_hi = vld1q_f32(f_fw + (ci_base + 0) * K0 + 4);
          float32x4_t f1_lo = vld1q_f32(f_fw + (ci_base + 1) * K0 + 0);
          float32x4_t f1_hi = vld1q_f32(f_fw + (ci_base + 1) * K0 + 4);
          float32x4_t f2_lo = vld1q_f32(f_fw + (ci_base + 2) * K0 + 0);
          float32x4_t f2_hi = vld1q_f32(f_fw + (ci_base + 2) * K0 + 4);
          float32x4_t f3_lo = vld1q_f32(f_fw + (ci_base + 3) * K0 + 0);
          float32x4_t f3_hi = vld1q_f32(f_fw + (ci_base + 3) * K0 + 4);

          IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
            if (i < ow_count) {
              float32x4_t in_vec =
                  vld1q_f32(in_fw + i * in_pos_stride + ci_base);

              acc[i][0] = vfmaq_laneq_f32(acc[i][0], f0_lo, in_vec, 0);
              acc[i][1] = vfmaq_laneq_f32(acc[i][1], f0_hi, in_vec, 0);
              acc[i][0] = vfmaq_laneq_f32(acc[i][0], f1_lo, in_vec, 1);
              acc[i][1] = vfmaq_laneq_f32(acc[i][1], f1_hi, in_vec, 1);
              acc[i][0] = vfmaq_laneq_f32(acc[i][0], f2_lo, in_vec, 2);
              acc[i][1] = vfmaq_laneq_f32(acc[i][1], f2_hi, in_vec, 2);
              acc[i][0] = vfmaq_laneq_f32(acc[i][0], f3_lo, in_vec, 3);
              acc[i][1] = vfmaq_laneq_f32(acc[i][1], f3_hi, in_vec, 3);
            }
          }
        }
      }
    }
  }

  IREE_UK_UNROLL for (int i = 0; i < OW_TILE; ++i) {
    if (i < ow_count) {
      vst1q_f32(out_ptr + i * K0 + 0, acc[i][0]);
      vst1q_f32(out_ptr + i * K0 + 4, acc[i][1]);
    }
  }
}

void iree_uk_conv_nchwc_tile_f32f32f32_8x8x8_arm_64(
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
    iree_uk_conv_nchwc_tile_f32f32f32_8x8x8_arm_64_impl(out_ptr, in_ptr, f_ptr,
                                                        params, OW_TILE);
  } else {
    iree_uk_conv_nchwc_tile_f32f32f32_8x8x8_arm_64_impl(
        out_ptr, in_ptr, f_ptr, params, params->ow_count);
  }
}

#undef OW_TILE
#undef K0
#undef C0
