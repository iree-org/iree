// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/conv_nchwc_internal.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"

// Inner reduction (ic_o, fh, fw, ci) for one output element. The outer driver
// below computes `in_window` (pointer to input[n, 0, oh*sH, ow*sW, 0]) and
// `f_slice` (pointer to filter[oc_o, 0, 0, 0, 0, ko]) and dispatches into here.
static void iree_uk_conv_nchwc_reference_innerloop_f32f32f32(
    float* out_ptr, const float* in_window, const float* f_slice,
    const iree_uk_conv_nchwc_params_t* params) {
  float acc =
      params->flags & IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE ? *out_ptr : 0.f;
  const iree_uk_index_t c0 = params->c0;
  const iree_uk_index_t k0 = params->k0;
  for (iree_uk_index_t ic_o = 0; ic_o < params->IC_outer; ++ic_o) {
    const float* in_ic = in_window + ic_o * params->input_stride_ic_outer;
    const float* f_ic = f_slice + ic_o * params->filter_stride_ic_outer;
    for (iree_uk_index_t fh = 0; fh < params->FH; ++fh) {
      const float* in_fh = in_ic + fh * params->input_stride_h;
      const float* f_fh = f_ic + fh * params->filter_stride_fh;
      for (iree_uk_index_t fw = 0; fw < params->FW; ++fw) {
        const float* f_fw = f_fh + fw * params->filter_stride_fw;
        for (iree_uk_index_t ci = 0; ci < c0; ++ci) {
          acc += in_fh[fw * c0 + ci] * f_fw[ci * k0];
        }
      }
    }
  }
  *out_ptr = acc;
}

static void iree_uk_conv_nchwc_reference(
    const iree_uk_conv_nchwc_params_t* params) {
  iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(params->flags);
  iree_uk_index_t in_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_input_type(type));
  iree_uk_index_t f_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_filter_type(type));
  iree_uk_index_t out_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_output_type(type));
  const char* in_base =
      (const char*)params->input_buffer + params->input_offset * in_elem_size;
  const char* f_base =
      (const char*)params->filter_buffer + params->filter_offset * f_elem_size;
  char* out_base =
      (char*)params->output_buffer + params->output_offset * out_elem_size;

  for (iree_uk_index_t n = 0; n < params->N; ++n) {
    for (iree_uk_index_t oc_o = 0; oc_o < params->OC_outer; ++oc_o) {
      for (iree_uk_index_t oh = 0; oh < params->OH; ++oh) {
        for (iree_uk_index_t ow = 0; ow < params->OW; ++ow) {
          const char* in_window =
              in_base + (n * params->input_stride_n +
                         oh * params->stride_h * params->input_stride_h +
                         ow * params->stride_w * params->c0) *
                            in_elem_size;
          char* out_pixel =
              out_base + (n * params->output_stride_n +
                          oc_o * params->output_stride_oc_outer +
                          oh * params->output_stride_oh + ow * params->k0) *
                             out_elem_size;
          for (iree_uk_index_t ko = 0; ko < params->k0; ++ko) {
            const char* f_slice =
                f_base +
                (oc_o * params->filter_stride_oc_outer + ko) * f_elem_size;
            void* out_ptr = out_pixel + ko * out_elem_size;
            switch (params->flags & IREE_UK_FLAG_CONV_NCHWC_TYPE_MASK) {
              case IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32:
                iree_uk_conv_nchwc_reference_innerloop_f32f32f32(
                    (float*)out_ptr, (const float*)in_window,
                    (const float*)f_slice, params);
                break;
              default:
                IREE_UK_ASSERT(false && "unhandled type");
            }
          }
        }
      }
    }
  }
}

// Strides/Offsets are element counts. For a sub-byte type (e.g., i4), an
// odd count is a fractional number of bytes which isn't byte addressable.
// Round up to the next whole-byte count.
static iree_uk_index_t iree_uk_test_round_up_to_ensure_multiple_of_8_bits(
    iree_uk_index_t index, iree_uk_type_t type) {
  while ((index << iree_uk_type_bit_count_log2(type)) & 7) ++index;
  return index;
}

// Randomly make the strides either tight or not to exercise all cases.
static iree_uk_index_t iree_uk_test_random_stride(
    iree_uk_index_t min_stride, iree_uk_type_t type,
    iree_uk_random_engine_t* engine) {
  iree_uk_index_t stride = min_stride + iree_uk_random_engine_get_0_1(engine);
  return iree_uk_test_round_up_to_ensure_multiple_of_8_bits(stride, type);
}

// Randomly make the offsets either tight or not to exercise all cases.
static iree_uk_index_t iree_uk_test_random_offset(
    iree_uk_type_t type, iree_uk_random_engine_t* engine) {
  iree_uk_index_t offset = iree_uk_random_engine_get_0_1(engine);
  return iree_uk_test_round_up_to_ensure_multiple_of_8_bits(offset, type);
}

static void iree_uk_test_conv_nchwc_for_shape_params(
    iree_uk_test_t* test, const iree_uk_conv_nchwc_params_t* src_params) {
  iree_uk_conv_nchwc_params_t params;
  memcpy(&params, src_params, sizeof params);
  iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(params.flags);
  iree_uk_type_t in_type = iree_uk_conv_nchwc_input_type(type);
  iree_uk_type_t f_type = iree_uk_conv_nchwc_filter_type(type);
  iree_uk_type_t out_type = iree_uk_conv_nchwc_output_type(type);
  iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);

  // Input H/W are derived from output shape, filter shape, and stride.
  iree_uk_index_t H = (params.OH - 1) * params.stride_h + params.FH;
  iree_uk_index_t W = (params.OW - 1) * params.stride_w + params.FW;

  // Populate the outer strides; innermost dims (W, OW, c0, k0) stay
  // pack-contiguous. We need these before sizing buffers below.
  params.input_stride_h =
      iree_uk_test_random_stride(W * params.c0, in_type, engine);
  params.input_stride_ic_outer =
      iree_uk_test_random_stride(H * params.input_stride_h, in_type, engine);
  params.input_stride_n = iree_uk_test_random_stride(
      params.IC_outer * params.input_stride_ic_outer, in_type, engine);

  params.filter_stride_fw = iree_uk_test_random_stride(
      (iree_uk_index_t)params.c0 * params.k0, f_type, engine);
  params.filter_stride_fh = iree_uk_test_random_stride(
      params.FW * params.filter_stride_fw, f_type, engine);
  params.filter_stride_ic_outer = iree_uk_test_random_stride(
      params.FH * params.filter_stride_fh, f_type, engine);
  params.filter_stride_oc_outer = iree_uk_test_random_stride(
      params.IC_outer * params.filter_stride_ic_outer, f_type, engine);

  params.output_stride_oh =
      iree_uk_test_random_stride(params.OW * params.k0, out_type, engine);
  params.output_stride_oc_outer = iree_uk_test_random_stride(
      params.OH * params.output_stride_oh, out_type, engine);
  params.output_stride_n = iree_uk_test_random_stride(
      params.OC_outer * params.output_stride_oc_outer, out_type, engine);

  iree_uk_index_t in_buffer_size =
      iree_uk_2d_buffer_length(in_type, params.N, params.input_stride_n);
  iree_uk_index_t f_buffer_size = iree_uk_2d_buffer_length(
      f_type, params.OC_outer, params.filter_stride_oc_outer);
  iree_uk_index_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.N, params.output_stride_n);
  void* in_buffer = malloc(in_buffer_size);
  void* f_buffer = malloc(f_buffer_size);
  iree_uk_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  iree_uk_write_random_buffer(f_buffer, f_buffer_size, f_type, engine);

  // Random offsets + pointer-shift trick: shift `buffer` back by `offset`, so
  // the kernel's `buffer + offset` lands at the actual allocation. Exercises
  // the kernel's offset arithmetic without inflating the allocation.
  params.input_offset = iree_uk_test_random_offset(in_type, engine);
  params.filter_offset = iree_uk_test_random_offset(f_type, engine);
  params.output_offset = iree_uk_test_random_offset(out_type, engine);
  params.input_buffer =
      (const char*)in_buffer -
      iree_uk_bits_to_bytes_exact(params.input_offset
                                  << iree_uk_type_bit_count_log2(in_type));
  params.filter_buffer =
      (const char*)f_buffer -
      iree_uk_bits_to_bytes_exact(params.filter_offset
                                  << iree_uk_type_bit_count_log2(f_type));

  void* init_out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(init_out_buffer, out_buffer_size, out_type,
                              engine);
  void* reference_out_buffer = malloc(out_buffer_size);
  void* actual_out_buffer = malloc(out_buffer_size);
  memcpy(reference_out_buffer, init_out_buffer, out_buffer_size);
  memcpy(actual_out_buffer, init_out_buffer, out_buffer_size);

  iree_uk_conv_nchwc_params_t reference_params;
  memcpy(&reference_params, &params, sizeof params);
  reference_params.output_buffer =
      (char*)reference_out_buffer -
      iree_uk_bits_to_bytes_exact(params.output_offset
                                  << iree_uk_type_bit_count_log2(out_type));

  iree_uk_conv_nchwc_params_t actual_params;
  memcpy(&actual_params, &params, sizeof params);
  actual_params.output_buffer =
      (char*)actual_out_buffer -
      iree_uk_bits_to_bytes_exact(params.output_offset
                                  << iree_uk_type_bit_count_log2(out_type));

  iree_uk_conv_nchwc_reference(&reference_params);
  iree_uk_conv_nchwc_p(&actual_params);

  // Exact compare: small-integer random fills keep every intermediate FMA
  // exactly representable, so reduction order is immaterial.
  if (memcmp(actual_out_buffer, reference_out_buffer, out_buffer_size)) {
    IREE_UK_TEST_FAIL(test);
  }

  free(in_buffer);
  free(f_buffer);
  free(init_out_buffer);
  free(reference_out_buffer);
  free(actual_out_buffer);
}

static void iree_uk_test_conv_nchwc_for_tile_params(iree_uk_test_t* test,
                                                    const void* src_params) {
  typedef struct conv_shape_t {
    int N, OC_outer, OH, OW, IC_outer, FH, FW, sH, sW;
  } conv_shape_t;
  const conv_shape_t shapes[] = {
      // An output extent is 0, so exit early.
      {0, 1, 3, 3, 1, 3, 3, 1, 1},  // N = 0
      {1, 0, 3, 3, 1, 3, 3, 1, 1},  // OC_outer = 0
      {1, 1, 0, 3, 1, 3, 3, 1, 1},  // OH = 0
      {1, 1, 3, 0, 1, 3, 3, 1, 1},  // OW = 0
      // Zero reduction: non-accumulate must zero the output panel; accumulate
      // must leave it intact.
      {1, 1, 2, 2, 0, 3, 3, 1, 1},  // IC_outer = 0
      {1, 1, 2, 2, 1, 0, 3, 1, 1},  // FH = 0
      {1, 1, 2, 2, 1, 3, 0, 1, 1},  // FW = 0
      // Regular cases.
      {1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 15, 1, 1, 1, 1, 1},
      {1, 1, 1, 16, 1, 1, 1, 1, 1},
      {1, 1, 1, 17, 1, 1, 1, 1, 1},
      {1, 2, 3, 5, 2, 3, 3, 1, 1},
      {2, 2, 5, 5, 2, 3, 3, 1, 1},
      {1, 1, 3, 3, 1, 1, 1, 1, 1},
      {1, 1, 5, 5, 1, 3, 3, 2, 2},
      {1, 1, 1, 33, 1, 3, 3, 1, 1},
  };
  for (int i = 0; i < (int)IREE_ARRAYSIZE(shapes); ++i) {
    iree_uk_conv_nchwc_params_t params;
    memcpy(&params, src_params, sizeof params);
    params.cpu_data = iree_uk_test_cpu_data(test);
    if (!(params.flags &
          IREE_UK_FLAG_CONV_NCHWC_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION)) {
      if (!(iree_uk_conv_nchwc_info_p(&params) &
            IREE_UK_FLAG_CONV_NCHWC_INFO_HAVE_ARCHITECTURE_SPECIFIC_TILE_FUNCTION)) {
        IREE_UK_ASSERT(0 &&
                       "No architecture-specific tile function available "
                       "for this case or missing CPU features, which should "
                       "have been handled earlier.");
        IREE_UK_TEST_FAIL(test);
      }
    }
    conv_shape_t s = shapes[i];
    params.N = s.N;
    params.OC_outer = s.OC_outer;
    params.OH = s.OH;
    params.OW = s.OW;
    params.IC_outer = s.IC_outer;
    params.FH = s.FH;
    params.FW = s.FW;
    params.stride_h = s.sH;
    params.stride_w = s.sW;
    const iree_uk_uint32_t base_flags = params.flags;
    for (int accumulate = 0; accumulate <= 1; ++accumulate) {
      params.flags = base_flags;
      if (accumulate) params.flags |= IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE;
      iree_uk_test_conv_nchwc_for_shape_params(test, &params);
    }
  }
}

static void iree_uk_test_conv_nchwc(iree_uk_uint32_t flags, int k0, int c0,
                                    const char* cpu_features) {
  // Always allow the fallback in this test. The problem with trying to enforce
  // that no fallback is accidentally used, is that it's not easy to tell when
  // a fallback is legitimate. It depends on the build system used (CMake or
  // Bazel) and within the CMake case, it depends on the native toolchain. The
  // ground truth is given by the IREE_UK_BUILD_* defines, but they are
  // currently an implementation detail within each arch/ subdirectory.
  flags |= IREE_UK_FLAG_CONV_NCHWC_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION;
  char types_str[32];
  iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(flags);
  iree_uk_type_triple_str(types_str, sizeof types_str, type);
  iree_uk_conv_nchwc_params_t params = {.flags = flags, .k0 = k0, .c0 = c0};
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%d",
           types_str, k0, c0);
  iree_uk_test(test_label_str, iree_uk_test_conv_nchwc_for_tile_params, &params,
               cpu_features);
}

int main(int argc, char** argv) {
  // Generic (no CPU features). Exercises the generic fallback tile.
  iree_uk_test_conv_nchwc(IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32, 16, 16, "");
  // A tile shape no arch dispatch table matches — exercises the generic
  // fallback even when arch-specific tiles are compiled in.
  iree_uk_test_conv_nchwc(IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32, 4, 4, "");

#if defined(IREE_ARCH_X86_64)
  iree_uk_test_conv_nchwc(IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32, 16, 16,
                          "avx512_base");
#endif

  return iree_uk_test_exit_status();
}
