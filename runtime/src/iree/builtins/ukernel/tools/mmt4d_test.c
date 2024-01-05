// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iree/builtins/ukernel/common.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/mmt4d_internal.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"

static void iree_mmt4d_reference_innerloop_f32f32f32(
    float* out_ptr, const float* lhs_ptr, const float* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  float acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0.f;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 = lhs_ptr[k * params->M0 * params->K0 + k0];
      float rhs_f32 = rhs_ptr[k * params->N0 * params->K0 + k0];
      acc += lhs_f32 * rhs_f32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_f16f16f32(
    float* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  float acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0.f;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_f16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_f16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      acc += lhs_f32 * rhs_f32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_f16f16f16_noskipround(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  uint16_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_f16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_f16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      float acc_f32 = iree_math_f16_to_f32(acc);
      acc = iree_math_f32_to_f16(acc_f32 + lhs_f32 * rhs_f32);
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_f16f16f16_skipround(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  float acc_f32 = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE
                      ? iree_math_f16_to_f32(*out_ptr)
                      : 0.f;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_f16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_f16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      acc_f32 += lhs_f32 * rhs_f32;
    }
  }
  *out_ptr = iree_math_f32_to_f16(acc_f32);
}

static void iree_mmt4d_reference_innerloop_f16f16f16(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  if (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) {
    iree_mmt4d_reference_innerloop_f16f16f16_skipround(out_ptr, lhs_ptr,
                                                       rhs_ptr, params);
  } else {
    iree_mmt4d_reference_innerloop_f16f16f16_noskipround(out_ptr, lhs_ptr,
                                                         rhs_ptr, params);
  }
}

static void iree_mmt4d_reference_innerloop_bf16bf16f32(
    float* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  float acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0.f;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_bf16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_bf16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      acc += lhs_f32 * rhs_f32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_bf16bf16bf16_noskipround(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  uint16_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_bf16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_bf16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      float acc_f32 = iree_math_bf16_to_f32(acc);
      acc = iree_math_f32_to_bf16(acc_f32 + lhs_f32 * rhs_f32);
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_bf16bf16bf16_skipround(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  float acc_f32 = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE
                      ? iree_math_bf16_to_f32(*out_ptr)
                      : 0.f;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      float lhs_f32 =
          iree_math_bf16_to_f32(lhs_ptr[k * params->M0 * params->K0 + k0]);
      float rhs_f32 =
          iree_math_bf16_to_f32(rhs_ptr[k * params->N0 * params->K0 + k0]);
      acc_f32 += lhs_f32 * rhs_f32;
    }
  }
  *out_ptr = iree_math_f32_to_bf16(acc_f32);
}

static void iree_mmt4d_reference_innerloop_bf16bf16bf16(
    uint16_t* out_ptr, const uint16_t* lhs_ptr, const uint16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  if (params->flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) {
    iree_mmt4d_reference_innerloop_bf16bf16bf16_skipround(out_ptr, lhs_ptr,
                                                          rhs_ptr, params);
  } else {
    iree_mmt4d_reference_innerloop_bf16bf16bf16_noskipround(out_ptr, lhs_ptr,
                                                            rhs_ptr, params);
  }
}

static void iree_mmt4d_reference_innerloop_s8s8s32(
    int32_t* out_ptr, const int8_t* lhs_ptr, const int8_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  int32_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      int32_t lhs_i32 = lhs_ptr[k * params->M0 * params->K0 + k0];
      int32_t rhs_i32 = rhs_ptr[k * params->N0 * params->K0 + k0];
      acc += lhs_i32 * rhs_i32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_s16s16s32(
    int32_t* out_ptr, const int16_t* lhs_ptr, const int16_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  int32_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      int32_t lhs_i32 = lhs_ptr[k * params->M0 * params->K0 + k0];
      int32_t rhs_i32 = rhs_ptr[k * params->N0 * params->K0 + k0];
      acc += lhs_i32 * rhs_i32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_s16u4s32(
    int32_t* out_ptr, const int16_t* lhs_ptr, const uint8_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  // K0 must be even.
  IREE_UK_ASSERT(!(params->K0 % 2));
  iree_uk_int16_t K0half = params->K0 / 2;
  int32_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    // As K0 must be even, we 2x-unroll the K0 loop, writing a 2D dot product.
    for (iree_uk_index_t k0h = 0; k0h < K0half; ++k0h) {
      int32_t lhs_0 = lhs_ptr[k * params->M0 * params->K0 + 2 * k0h];
      int32_t lhs_1 = lhs_ptr[k * params->M0 * params->K0 + 2 * k0h + 1];
      uint8_t rhs_byte = rhs_ptr[k * params->N0 * K0half + k0h];
      int32_t rhs_0 = rhs_byte & 0xf;
      int32_t rhs_1 = rhs_byte >> 4;
      acc += lhs_0 * rhs_0 + lhs_1 * rhs_1;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference_innerloop_s16s8s32(
    int32_t* out_ptr, const int16_t* lhs_ptr, const int8_t* rhs_ptr,
    const iree_uk_mmt4d_params_t* params) {
  int32_t acc = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE ? *out_ptr : 0;
  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    for (iree_uk_index_t k0 = 0; k0 < params->K0; ++k0) {
      int32_t lhs_i32 = lhs_ptr[k * params->M0 * params->K0 + k0];
      int32_t rhs_i32 = rhs_ptr[k * params->N0 * params->K0 + k0];
      acc += lhs_i32 * rhs_i32;
    }
  }
  *out_ptr = acc;
}

static void iree_mmt4d_reference(const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->flags);
  iree_uk_index_t lhs_elem_bits =
      iree_uk_type_bit_count(iree_uk_mmt4d_lhs_type(mmt4d_type));
  iree_uk_index_t rhs_elem_bits =
      iree_uk_type_bit_count(iree_uk_mmt4d_rhs_type(mmt4d_type));
  iree_uk_index_t out_elem_size =
      iree_uk_type_size(iree_uk_mmt4d_out_type(mmt4d_type));
  for (iree_uk_index_t i = 0; i < params->M; ++i) {
    for (iree_uk_index_t j = 0; j < params->N; ++j) {
      void* out_tile_ptr = ((char*)params->out_buffer) +
                           (params->out_offset + i * params->out_stride0 +
                            j * params->M0 * params->N0) *
                               out_elem_size;
      const void* lhs_panel_ptr =
          ((const char*)params->lhs_buffer) +
          iree_uk_bits_to_bytes_exact(
              (params->lhs_offset + i * params->lhs_stride0) * lhs_elem_bits);
      const void* rhs_panel_ptr =
          ((const char*)params->rhs_buffer) +
          iree_uk_bits_to_bytes_exact(
              (params->rhs_offset + j * params->rhs_stride0) * rhs_elem_bits);
      for (iree_uk_index_t i0 = 0; i0 < params->M0; ++i0) {
        for (iree_uk_index_t j0 = 0; j0 < params->N0; ++j0) {
          void* out_ptr =
              ((char*)out_tile_ptr) + (i0 * params->N0 + j0) * out_elem_size;
          const void* lhs_ptr =
              ((char*)lhs_panel_ptr) +
              iree_uk_bits_to_bytes_exact(i0 * params->K0 * lhs_elem_bits);
          const void* rhs_ptr =
              ((char*)rhs_panel_ptr) +
              iree_uk_bits_to_bytes_exact(j0 * params->K0 * rhs_elem_bits);
          switch (params->flags & IREE_UK_FLAG_MMT4D_TYPE_MASK) {
            case IREE_UK_FLAG_MMT4D_TYPE_F32F32F32:
              iree_mmt4d_reference_innerloop_f32f32f32(
                  (float*)out_ptr, (const float*)lhs_ptr, (const float*)rhs_ptr,
                  params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_F16F16F32:
              iree_mmt4d_reference_innerloop_f16f16f32(
                  (float*)out_ptr, (const uint16_t*)lhs_ptr,
                  (const uint16_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_F16F16F16:
              iree_mmt4d_reference_innerloop_f16f16f16(
                  (uint16_t*)out_ptr, (const uint16_t*)lhs_ptr,
                  (const uint16_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32:
              iree_mmt4d_reference_innerloop_bf16bf16f32(
                  (float*)out_ptr, (const uint16_t*)lhs_ptr,
                  (const uint16_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16:
              iree_mmt4d_reference_innerloop_bf16bf16bf16(
                  (uint16_t*)out_ptr, (const uint16_t*)lhs_ptr,
                  (const uint16_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_S8S8S32:
              iree_mmt4d_reference_innerloop_s8s8s32(
                  (int32_t*)out_ptr, (const int8_t*)lhs_ptr,
                  (const int8_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_S16S16S32:
              iree_mmt4d_reference_innerloop_s16s16s32(
                  (int32_t*)out_ptr, (const int16_t*)lhs_ptr,
                  (const int16_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_S16U4S32:
              iree_mmt4d_reference_innerloop_s16u4s32(
                  (int32_t*)out_ptr, (const int16_t*)lhs_ptr,
                  (const uint8_t*)rhs_ptr, params);
              break;
            case IREE_UK_FLAG_MMT4D_TYPE_S16S8S32:
              iree_mmt4d_reference_innerloop_s16s8s32(
                  (int32_t*)out_ptr, (const int16_t*)lhs_ptr,
                  (const int8_t*)rhs_ptr, params);
              break;
            default:
              IREE_UK_ASSERT(false && "unhandled type");
          }
          out_ptr = ((char*)out_ptr) + out_elem_size;
        }
      }
    }
  }
}

static iree_uk_index_t iree_uk_test_round_up_to_ensure_multiple_of_8_bits(
    iree_uk_index_t index, iree_uk_type_t type) {
  // Honor the requirement that strides should be multiples of 8 bits.
  while ((index << iree_uk_type_bit_count_log2(type)) & 7) {
    ++index;
  }
  return index;
}

static iree_uk_index_t iree_uk_test_random_stride(
    iree_uk_index_t min_stride, iree_uk_type_t type,
    iree_uk_random_engine_t* engine) {
  // Randomly make strides either tight or not to exercise all cases.
  iree_uk_index_t stride = min_stride + iree_uk_random_engine_get_0_1(engine);
  return iree_uk_test_round_up_to_ensure_multiple_of_8_bits(stride, type);
}

static iree_uk_index_t iree_uk_test_random_offset(
    iree_uk_type_t type, iree_uk_random_engine_t* engine) {
  // Randomly make strides either tight or not to exercise all cases.
  iree_uk_index_t stride = iree_uk_random_engine_get_0_1(engine);
  return iree_uk_test_round_up_to_ensure_multiple_of_8_bits(stride, type);
}

static void iree_uk_test_mmt4d_for_shape_params(
    iree_uk_test_t* test, const iree_uk_mmt4d_params_t* src_params) {
  iree_uk_mmt4d_params_t params;
  memcpy(&params, src_params, sizeof params);
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params.flags);
  iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(mmt4d_type);
  iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(mmt4d_type);
  iree_uk_type_t out_type = iree_uk_mmt4d_out_type(mmt4d_type);
  // Populate strides first - we need them below to compute buffer lengths.
  // Randomly make strides either tight or not to exercise all cases.
  iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
  params.lhs_stride0 = iree_uk_test_random_stride(
      params.K * params.M0 * params.K0, lhs_type, engine);
  params.rhs_stride0 = iree_uk_test_random_stride(
      params.K * params.N0 * params.K0, rhs_type, engine);
  params.out_stride0 = iree_uk_test_random_stride(
      params.N * params.M0 * params.N0, out_type, engine);
  iree_uk_index_t lhs_buffer_size =
      iree_uk_2d_buffer_length(lhs_type, params.M, params.lhs_stride0);
  iree_uk_index_t rhs_buffer_size =
      iree_uk_2d_buffer_length(rhs_type, params.N, params.rhs_stride0);
  void* lhs_buffer = malloc(lhs_buffer_size);
  void* rhs_buffer = malloc(rhs_buffer_size);
  iree_uk_write_random_buffer(lhs_buffer, lhs_buffer_size, lhs_type, engine);
  iree_uk_write_random_buffer(rhs_buffer, rhs_buffer_size, rhs_type, engine);
  params.lhs_offset = iree_uk_test_random_offset(lhs_type, engine);
  params.rhs_offset = iree_uk_test_random_offset(rhs_type, engine);
  params.out_offset = iree_uk_test_random_offset(out_type, engine);
  params.lhs_buffer =
      (const char*)lhs_buffer -
      iree_uk_bits_to_bytes_exact(params.lhs_offset
                                  << iree_uk_type_bit_count_log2(lhs_type));
  params.rhs_buffer =
      (const char*)rhs_buffer -
      iree_uk_bits_to_bytes_exact(params.rhs_offset
                                  << iree_uk_type_bit_count_log2(rhs_type));

  iree_uk_mmt4d_params_t reference_params;
  memcpy(&reference_params, &params, sizeof params);
  iree_uk_index_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.M, params.out_stride0);
  void* init_out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(init_out_buffer, out_buffer_size, out_type,
                              engine);
  void* reference_out_buffer = malloc(out_buffer_size);
  memcpy(reference_out_buffer, init_out_buffer, out_buffer_size);
  reference_params.out_buffer =
      (char*)reference_out_buffer -
      iree_uk_bits_to_bytes_exact(params.out_offset
                                  << iree_uk_type_bit_count_log2(out_type));

  iree_uk_mmt4d_params_t actual_params;
  memcpy(&actual_params, &params, sizeof params);
  void* actual_out_buffer = malloc(out_buffer_size);
  memcpy(actual_out_buffer, init_out_buffer, out_buffer_size);
  actual_params.out_buffer =
      (char*)actual_out_buffer -
      iree_uk_bits_to_bytes_exact(params.out_offset
                                  << iree_uk_type_bit_count_log2(out_type));

  iree_mmt4d_reference(&reference_params);
  iree_uk_mmt4d(&actual_params);

  // For now we use exact comparisons, even for float, even though the reference
  // code accumulates in a different order compared to the actual code. This
  // relies on picking input test matrix elements so that all intermediate
  // values are exactly representable - i.e. small integer numerators.
  // This also relies on honoring IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS
  // consistently between actual tile functions (including generic fallback
  // ones) and the reference code in this test.
  bool fail = memcmp(actual_out_buffer, reference_out_buffer, out_buffer_size);

  if (fail) {
    IREE_UK_TEST_FAIL(test);
  }

  free(init_out_buffer);
  free(reference_out_buffer);
  free(actual_out_buffer);
  free(lhs_buffer);
  free(rhs_buffer);
}

static void iree_uk_test_mmt4d_for_tile_params(iree_uk_test_t* test,
                                               const void* src_params) {
  typedef struct shape_mnk_t {
    int m, n, k;
  } shape_mnk_t;
  const iree_uk_mmt4d_type_t mmt4d_type =
      iree_uk_mmt4d_type(((const iree_uk_mmt4d_params_t*)src_params)->flags);
  const iree_uk_type_t out_type = iree_uk_mmt4d_out_type(mmt4d_type);
  const int max_reduction_size =
      (out_type == IREE_UK_TYPE_BFLOAT_16) ? 100 : 1000;
  const shape_mnk_t shapes[] = {
      // Degenerate case M==0. Vacuous.
      {0, 1, 1},
      {0, 5, 7},
      // Degenerate case N==0. Vacuous.
      {1, 0, 1},
      {5, 0, 7},
      // Degenerate case K==0. Vacuous if flags have ACCUMULATE. Zeroing the
      // output buffer otherwise.
      {1, 1, 0},
      {5, 7, 0},
      // Non-degenerate cases.
      {1, 1, 1},
      {1, 1, 2},
      {1, 1, 10},
      {1, 1, max_reduction_size},
      {2, 1, 1},
      {1, 2, 1},
      {2, 2, 2},
      {5, 7, 13},
  };
  for (int i = 0; i < IREE_ARRAYSIZE(shapes); ++i) {
    iree_uk_mmt4d_params_t params;
    memcpy(&params, src_params, sizeof params);
    params.cpu_data = iree_uk_test_cpu_data(test);
    shape_mnk_t shape = shapes[i];
    params.M = shape.m;
    params.N = shape.n;
    params.K = shape.k;
    for (int accumulate = 0; accumulate <= 1; ++accumulate) {
      if (accumulate) params.flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
      iree_uk_test_mmt4d_for_shape_params(test, &params);
    }
  }
}

static void iree_uk_test_mmt4d_impl(iree_uk_uint32_t flags, int M0, int N0,
                                    int K0, const char* cpu_features) {
  const char* code_path_suffix = "";
  if (flags & IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS) {
    code_path_suffix = " skipround";
  }
  char types_str[32];
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(flags);
  iree_uk_type_triple_str(types_str, sizeof types_str, mmt4d_type);
  iree_uk_mmt4d_params_t params = {
      .flags = flags, .M0 = M0, .N0 = N0, .K0 = K0};
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%dx%d%s",
           types_str, M0, N0, K0, code_path_suffix);
  iree_uk_test(test_label_str, iree_uk_test_mmt4d_for_tile_params, &params,
               cpu_features);
}

static void iree_uk_test_mmt4d(iree_uk_uint32_t flags, int M0, int N0, int K0,
                               const char* cpu_features) {
  // Test narrowed, power-of-two values of M0, as mmt4d kernels tend to have
  // narrow variants for handling these cases.
  for (int narrowM0 = 1; narrowM0 < M0; narrowM0 *= 2) {
    iree_uk_test_mmt4d_impl(flags, narrowM0, N0, K0, cpu_features);
  }
  iree_uk_test_mmt4d_impl(flags, M0, N0, K0, cpu_features);
}

static void iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
    iree_uk_uint32_t flags, int M0, int N0, int K0, const char* cpu_features) {
  iree_uk_test_mmt4d(flags, M0, N0, K0, cpu_features);
  iree_uk_test_mmt4d(flags | IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS, M0,
                     N0, K0, cpu_features);
}

int main(int argc, char** argv) {
  // Generic tests, not matching any particular CPU feature. This is the place
  // to test weird M0, N0, K0 to ensure e.g. that we haven't unwittingly baked
  // in a power-of-two assumption
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 3, 5, 7, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 9, 6, 3, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 7, 3, 6, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16U4S32, 5, 3, 2, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S8S32, 7, 5, 6, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 4, 6, 5, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 3, 5, 8, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32, 11, 4, 1, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16, 2, 9, 3, "");

#if defined(IREE_ARCH_ARM_64)
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 8, 1, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1, "fp16fml");
  iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
      IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1, "");
  iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
      IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1, "fullfp16");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16, 8, 8, 4, "bf16");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32, 8, 8, 4, "bf16");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 1, "");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 4, "dotprod");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 8, "i8mm");
#elif defined(IREE_ARCH_X86_64)
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 4, 1, "");  // SSE
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 8, 8, 1, "avx2_fma");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F32F32F32, 16, 16, 1,
                     "avx512_base");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 8, 8, 1, "avx2_fma");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_F16F16F32, 16, 16, 1,
                     "avx512_base");
  iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
      IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 8, 8, 1, "avx2_fma");
  iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
      IREE_UK_FLAG_MMT4D_TYPE_F16F16F16, 16, 16, 1, "avx512_base");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32, 16, 16, 2,
                     "avx512_bf16");
  iree_uk_test_mmt4d_default_and_skip_intermediate_roundings(
      IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16, 16, 16, 2, "avx512_bf16");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 4, 2, "");  // SSE2
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 8, 8, 2, "avx2_fma");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 16, 16, 2, "avx512_base");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S8S8S32, 16, 16, 2, "avx512_vnni");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 16, 16, 2, "avx2_fma");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 16, 16, 2,
                     "avx512_base");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16S16S32, 16, 16, 2,
                     "avx512_vnni");
  iree_uk_test_mmt4d(IREE_UK_FLAG_MMT4D_TYPE_S16U4S32, 1, 32, 8, "avx512_vnni");
#endif  // defined(IREE_ARCH_ARM_64)

  return iree_uk_test_exit_status();
}
