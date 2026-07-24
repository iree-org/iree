// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/conv_nchwc.h"

#include "iree/builtins/ukernel/conv_nchwc_internal.h"
#include "iree/builtins/ukernel/exported_bits.h"

static void iree_uk_conv_nchwc_validate(
    const iree_uk_conv_nchwc_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_uint32_t allflags =
      IREE_UK_FLAG_CONV_NCHWC_TYPE_MASK | IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE |
      IREE_UK_FLAG_CONV_NCHWC_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  const iree_uk_uint32_t type_bits =
      params->flags & IREE_UK_FLAG_CONV_NCHWC_TYPE_MASK;
  IREE_UK_ASSERT(type_bits > 0 && type_bits < IREE_UK_FLAG_CONV_NCHWC_TYPE_END);
  // Bound shape dims to a narrower int32 range here. This can be
  // relaxed later if needed.
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->N, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->OC_outer, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->OH, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->OW, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->IC_outer, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->FH, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->FW, 31));
  // int32 is overkill for tile sizes; enforce int16 range for now.
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->k0, 15));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->c0, 15));
  IREE_UK_ASSERT(params->k0 > 0 && params->c0 > 0);
  IREE_UK_ASSERT(params->stride_h > 0 && params->stride_w > 0);
#endif
}

static bool iree_uk_conv_nchwc_early(
    const iree_uk_conv_nchwc_params_t* params) {
  if (params->N == 0 || params->OC_outer == 0 || params->OH == 0 ||
      params->OW == 0) {
    return true;
  }
  if ((params->IC_outer == 0 || params->FH == 0 || params->FW == 0) &&
      (params->flags & IREE_UK_FLAG_CONV_NCHWC_ACCUMULATE)) {
    return true;
  }
  return false;
}

// Shared outer driver for conv2d tile implementation.
//
// The tile function is called once per output panel of shape [OW_TILE, k0].
// It receives:
//   - output_panel: pointer to output[n, oc_outer, oh, ow, 0]  (k0 inner)
//   - input_panel:  pointer to input[n, 0, oh x sH, ow x sW, 0]
//   - filter_panel: pointer to filter[oc_outer, 0, 0, 0, 0, 0]
//   - params:      strides + shape info
//
// The tile function internally handles the reduction over (ic_outer, fh, fw,
// c_inner) and honors the ACCUMULATE flag for the whole call.
static void iree_uk_conv_nchwc_using_tile_func(
    const iree_uk_conv_nchwc_params_t* params,
    iree_uk_conv_nchwc_tile_func_t tile_func, iree_uk_int32_t OW_TILE) {
  IREE_UK_ASSERT(OW_TILE > 0);
  const iree_uk_index_t N = params->N;
  const iree_uk_index_t OC_outer = params->OC_outer;
  const iree_uk_index_t OH = params->OH;
  const iree_uk_index_t OW = params->OW;
  const iree_uk_int32_t k0 = params->k0;
  const iree_uk_int32_t c0 = params->c0;
  const iree_uk_int32_t stride_h = params->stride_h;
  const iree_uk_int32_t stride_w = params->stride_w;
  const iree_uk_index_t input_stride_n = params->input_stride_n;
  const iree_uk_index_t input_stride_h = params->input_stride_h;
  const iree_uk_index_t filter_stride_oc_outer = params->filter_stride_oc_outer;
  const iree_uk_index_t output_stride_n = params->output_stride_n;
  const iree_uk_index_t output_stride_oc_outer = params->output_stride_oc_outer;
  const iree_uk_index_t output_stride_oh = params->output_stride_oh;

  const iree_uk_conv_nchwc_type_t type = iree_uk_conv_nchwc_type(params->flags);
  const iree_uk_index_t input_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_input_type(type));
  const iree_uk_index_t filter_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_filter_type(type));
  const iree_uk_index_t out_elem_size =
      iree_uk_type_size(iree_uk_conv_nchwc_output_type(type));

  const char* input_base = (const char*)params->input_buffer +
                           params->input_offset * input_elem_size;
  const char* filter_base = (const char*)params->filter_buffer +
                            params->filter_offset * filter_elem_size;
  char* output_base =
      (char*)params->output_buffer + params->output_offset * out_elem_size;

  iree_uk_conv_nchwc_params_t call_params = *params;

  for (iree_uk_index_t n = 0; n < N; ++n) {
    const char* in_n = input_base + n * input_stride_n * input_elem_size;
    char* out_n = output_base + n * output_stride_n * out_elem_size;

    for (iree_uk_index_t oc_o = 0; oc_o < OC_outer; ++oc_o) {
      const char* f_panel =
          filter_base + oc_o * filter_stride_oc_outer * filter_elem_size;
      char* out_oc = out_n + oc_o * output_stride_oc_outer * out_elem_size;

      for (iree_uk_index_t oh = 0; oh < OH; ++oh) {
        char* out_oh = out_oc + oh * output_stride_oh * out_elem_size;
        iree_uk_index_t ih = oh * stride_h;
        const char* in_oh = in_n + ih * input_stride_h * input_elem_size;

        for (iree_uk_index_t ow = 0; ow < OW; ow += OW_TILE) {
          iree_uk_index_t ow_count = OW_TILE;
          if (ow + OW_TILE > OW) {
            ow_count = OW - ow;
          }
          char* out_panel = out_oh + ow * k0 * out_elem_size;
          iree_uk_index_t iw = ow * stride_w;
          const char* in_panel = in_oh + iw * c0 * input_elem_size;

          // Active output width for this call; equals OW_TILE except on the
          // right-edge tail.
          call_params.ow_count = ow_count;
          tile_func(out_panel, in_panel, f_panel, &call_params);
        }
      }
    }
  }
}

void iree_uk_conv_nchwc_p(const iree_uk_conv_nchwc_params_t* params) {
  iree_uk_conv_nchwc_validate(params);

  // Return early without selecting a tile function in trivial cases.
  if (iree_uk_conv_nchwc_early(params)) return;

  // Select a target specific tile function.
  iree_uk_conv_nchwc_tile_selection_t sel =
      iree_uk_conv_nchwc_select_tile_func_arch(params);

  // If no target specific tile function is available, fall back to a generic
  // one if allowed by the flags.
  if (!sel.tile_func) {
    if (params->flags &
        IREE_UK_FLAG_CONV_NCHWC_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION) {
      sel = iree_uk_conv_nchwc_select_tile_func_generic(params);
    } else {
      IREE_UK_ASSERT(
          0 && "no target-specific tile function, and fallback not enabled.");
    }
  }
  iree_uk_conv_nchwc_using_tile_func(params, sel.tile_func, sel.ow_tile);
}

IREE_UK_EXPORT void iree_uk_conv_nchwc(
    const void* input_buffer, iree_uk_index_t input_offset,
    iree_uk_index_t input_stride_n, iree_uk_index_t input_stride_ic_outer,
    iree_uk_index_t input_stride_h, const void* filter_buffer,
    iree_uk_index_t filter_offset, iree_uk_index_t filter_stride_oc_outer,
    iree_uk_index_t filter_stride_ic_outer, iree_uk_index_t filter_stride_fh,
    iree_uk_index_t filter_stride_fw, void* output_buffer,
    iree_uk_index_t output_offset, iree_uk_index_t output_stride_n,
    iree_uk_index_t output_stride_oc_outer, iree_uk_index_t output_stride_oh,
    iree_uk_index_t N, iree_uk_index_t OC_outer, iree_uk_index_t OH,
    iree_uk_index_t OW, iree_uk_index_t IC_outer, iree_uk_index_t FH,
    iree_uk_index_t FW, iree_uk_int32_t k0, iree_uk_int32_t c0,
    iree_uk_int32_t stride_h, iree_uk_int32_t stride_w, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_nchwc_params_t params = {
      .input_buffer = input_buffer,
      .input_offset = input_offset,
      .input_stride_n = input_stride_n,
      .input_stride_ic_outer = input_stride_ic_outer,
      .input_stride_h = input_stride_h,
      .filter_buffer = filter_buffer,
      .filter_offset = filter_offset,
      .filter_stride_oc_outer = filter_stride_oc_outer,
      .filter_stride_ic_outer = filter_stride_ic_outer,
      .filter_stride_fh = filter_stride_fh,
      .filter_stride_fw = filter_stride_fw,
      .output_buffer = output_buffer,
      .output_offset = output_offset,
      .output_stride_n = output_stride_n,
      .output_stride_oc_outer = output_stride_oc_outer,
      .output_stride_oh = output_stride_oh,
      .N = N,
      .OC_outer = OC_outer,
      .OH = OH,
      .OW = OW,
      .IC_outer = IC_outer,
      .FH = FH,
      .FW = FW,
      .k0 = k0,
      .c0 = c0,
      .stride_h = stride_h,
      .stride_w = stride_w,
      .flags = flags,
      .cpu_data = cpu_data,
  };
  iree_uk_conv_nchwc_p(&params);
}

iree_uk_uint32_t iree_uk_conv_nchwc_info_p(
    const iree_uk_conv_nchwc_params_t* params) {
  iree_uk_uint32_t result = 0;
  if (iree_uk_conv_nchwc_select_tile_func_arch(params).tile_func) {
    result |=
        IREE_UK_FLAG_CONV_NCHWC_INFO_HAVE_ARCHITECTURE_SPECIFIC_TILE_FUNCTION;
  }
  return result;
}

IREE_UK_EXPORT iree_uk_uint32_t iree_uk_conv_nchwc_info(
    iree_uk_int32_t k0, iree_uk_int32_t c0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data) {
  iree_uk_conv_nchwc_params_t params = {
      .k0 = k0, .c0 = c0, .flags = flags, .cpu_data = cpu_data};
  return iree_uk_conv_nchwc_info_p(&params);
}
