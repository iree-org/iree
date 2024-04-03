// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/mmt4d_internal.h"

// Generic implementation of matmul tile, s8*s4->s32 case.
static void iree_uk_mmt4d_tile_s8s4s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int8_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // K0 must be even.
  IREE_UK_ASSERT(!(K0 % 2));
  iree_uk_int16_t K0half = K0 / 2;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int32_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        // As K0 must be even, we 2x-unroll the K0 loop, writing a 2D dot
        // product.
        for (iree_uk_index_t k0h = 0; k0h < K0half; ++k0h) {
          iree_uk_int32_t lhs_0 = lhs_panel[k * M0 * K0 + i0 * K0 + 2 * k0h];
          iree_uk_int32_t lhs_1 =
              lhs_panel[k * M0 * K0 + i0 * K0 + 2 * k0h + 1];
          iree_uk_int8_t rhs_byte =
              rhs_panel[k * N0 * K0half + j0 * K0half + k0h];
          iree_uk_int8_t rhs_0 = rhs_byte & 0x0F;
          if (rhs_0 & 0x08) {
            rhs_0 |= 0xF0;
          }
          iree_uk_int8_t rhs_1 = (rhs_byte >> 4) & 0x0F;
          if (rhs_1 & 0x08) {
            rhs_1 |= 0xF0;
          }
          acc += lhs_0 * rhs_0 + lhs_1 * rhs_1;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, s8*s8->s32 case.
static void iree_uk_mmt4d_tile_s8s8s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int8_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int32_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          iree_uk_int32_t lhs_i32 = lhs_panel[k * M0 * K0 + i0 * K0 + k0];
          iree_uk_int32_t rhs_i32 = rhs_panel[k * N0 * K0 + j0 * K0 + k0];
          acc += lhs_i32 * rhs_i32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, s16*s16->s32 case.
static void iree_uk_mmt4d_tile_s16s16s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int32_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          iree_uk_int32_t lhs_i32 = lhs_panel[k * M0 * K0 + i0 * K0 + k0];
          iree_uk_int32_t rhs_i32 = rhs_panel[k * N0 * K0 + j0 * K0 + k0];
          acc += lhs_i32 * rhs_i32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, s16*u4->s32 case.
static void iree_uk_mmt4d_tile_s16u4s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  // K0 must be even.
  IREE_UK_ASSERT(!(K0 % 2));
  iree_uk_int16_t K0half = K0 / 2;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int32_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        // As K0 must be even, we 2x-unroll the K0 loop, writing a 2D dot
        // product.
        for (iree_uk_index_t k0h = 0; k0h < K0half; ++k0h) {
          iree_uk_int32_t lhs_0 = lhs_panel[k * M0 * K0 + i0 * K0 + 2 * k0h];
          iree_uk_int32_t lhs_1 =
              lhs_panel[k * M0 * K0 + i0 * K0 + 2 * k0h + 1];
          iree_uk_uint8_t rhs_byte =
              rhs_panel[k * N0 * K0half + j0 * K0half + k0h];
          iree_uk_int32_t rhs_0 = rhs_byte & 0xf;
          iree_uk_int32_t rhs_1 = rhs_byte >> 4;
          acc += lhs_0 * rhs_0 + lhs_1 * rhs_1;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, s16*s8->s32 case.
static void iree_uk_mmt4d_tile_s16s8s32_generic(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* out_tile = out_tile_untyped;
  const iree_uk_int16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_int8_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int32_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          iree_uk_int32_t lhs_i32 = lhs_panel[k * M0 * K0 + i0 * K0 + k0];
          iree_uk_int32_t rhs_i32 = rhs_panel[k * N0 * K0 + j0 * K0 + k0];
          acc += lhs_i32 * rhs_i32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
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
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      float acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                      ? out_tile[i0 * N0 + j0]
                      : 0.f;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 = lhs_panel[k * M0 * K0 + i0 * K0 + k0];
          float rhs_f32 = rhs_panel[k * N0 * K0 + j0 * K0 + k0];
          acc += lhs_f32 * rhs_f32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
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
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      float acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                      ? out_tile[i0 * N0 + j0]
                      : 0.f;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_f16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_f16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          acc += lhs_f32 * rhs_f32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, f16*f16->f16 case.
// Not skipping intermediate roundings.
static void iree_uk_mmt4d_tile_f16f16f16_generic_noskipround(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int16_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_f16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_f16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          float acc_f32 = iree_uk_f16_to_f32(acc);
          acc = iree_uk_f32_to_f16(acc_f32 + lhs_f32 * rhs_f32);
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, f16*f16->f16 case.
// Skipping intermediate roundings.
static void iree_uk_mmt4d_tile_f16f16f16_generic_skipround(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      float acc_f32 = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                          ? iree_uk_f16_to_f32(out_tile[i0 * N0 + j0])
                          : 0.f;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_f16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_f16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          acc_f32 += lhs_f32 * rhs_f32;
        }
      }
      out_tile[i0 * N0 + j0] = iree_uk_f32_to_f16(acc_f32);
    }
  }
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
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      float acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                      ? out_tile[i0 * N0 + j0]
                      : 0.f;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_bf16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_bf16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          acc += lhs_f32 * rhs_f32;
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, bf16*bf16->bf16 case.
// Not skipping intermediate roundings.
static void iree_uk_mmt4d_tile_bf16bf16bf16_generic_noskipround(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      iree_uk_int16_t acc = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                                ? out_tile[i0 * N0 + j0]
                                : 0;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_bf16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_bf16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          float acc_f32 = iree_uk_bf16_to_f32(acc);
          acc = iree_uk_f32_to_bf16(acc_f32 + lhs_f32 * rhs_f32);
        }
      }
      out_tile[i0 * N0 + j0] = acc;
    }
  }
}

// Generic implementation of matmul tile, bf16*bf16->bf16 case.
// Skipping intermediate roundings.
static void iree_uk_mmt4d_tile_bf16bf16bf16_generic_skipround(
    void* out_tile_untyped, const void* lhs_panel_untyped,
    const void* rhs_panel_untyped, const iree_uk_mmt4d_params_t* params) {
  iree_uk_int16_t* out_tile = out_tile_untyped;
  const iree_uk_uint16_t* lhs_panel = lhs_panel_untyped;
  const iree_uk_uint16_t* rhs_panel = rhs_panel_untyped;
  iree_uk_int16_t M0 = params->M0;
  iree_uk_int16_t N0 = params->N0;
  iree_uk_int16_t K0 = params->K0;
  for (iree_uk_index_t i0 = 0; i0 < M0; ++i0) {
    for (iree_uk_index_t j0 = 0; j0 < N0; ++j0) {
      float acc_f32 = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)
                          ? iree_uk_bf16_to_f32(out_tile[i0 * N0 + j0])
                          : 0.f;
      for (iree_uk_index_t k = 0; k < params->K; ++k) {
        for (iree_uk_index_t k0 = 0; k0 < K0; ++k0) {
          float lhs_f32 =
              iree_uk_bf16_to_f32(lhs_panel[k * M0 * K0 + i0 * K0 + k0]);
          float rhs_f32 =
              iree_uk_bf16_to_f32(rhs_panel[k * N0 * K0 + j0 * K0 + k0]);
          acc_f32 += lhs_f32 * rhs_f32;
        }
      }
      out_tile[i0 * N0 + j0] = iree_uk_f32_to_bf16(acc_f32);
    }
  }
}

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_generic(
    const iree_uk_mmt4d_params_t* params) {
  switch (iree_uk_mmt4d_type(params->flags)) {
    case iree_uk_mmt4d_type_f32f32f32:
      return iree_uk_mmt4d_tile_f32f32f32_generic;
    case iree_uk_mmt4d_type_s8s8s32:
      return iree_uk_mmt4d_tile_s8s8s32_generic;
    case iree_uk_mmt4d_type_s8s4s32:
      return iree_uk_mmt4d_tile_s8s4s32_generic;
    case iree_uk_mmt4d_type_s16s16s32:
      return iree_uk_mmt4d_tile_s16s16s32_generic;
    case iree_uk_mmt4d_type_s16u4s32:
      return iree_uk_mmt4d_tile_s16u4s32_generic;
    case iree_uk_mmt4d_type_s16s8s32:
      return iree_uk_mmt4d_tile_s16s8s32_generic;
    case iree_uk_mmt4d_type_f16f16f32:
      return iree_uk_mmt4d_tile_f16f16f32_generic;
    case iree_uk_mmt4d_type_f16f16f16:
      return (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS)
                 ? iree_uk_mmt4d_tile_f16f16f16_generic_skipround
                 : iree_uk_mmt4d_tile_f16f16f16_generic_noskipround;
    case iree_uk_mmt4d_type_bf16bf16f32:
      return iree_uk_mmt4d_tile_bf16bf16f32_generic;
    case iree_uk_mmt4d_type_bf16bf16bf16:
      return (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS)
                 ? iree_uk_mmt4d_tile_bf16bf16bf16_generic_skipround
                 : iree_uk_mmt4d_tile_bf16bf16bf16_generic_noskipround;
    default:
      // Shouldn't happen, validated earlier.
      return 0;
  }
}
