// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_CONV_NCHWC_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_CONV_NCHWC_INTERNAL_H_

#include "iree/builtins/ukernel/conv_nchwc.h"

// Internal params struct passed between implementation functions.
typedef struct iree_uk_conv_nchwc_params_t {
  const void* input_buffer;
  iree_uk_index_t input_offset;
  iree_uk_index_t input_stride_n;
  iree_uk_index_t input_stride_ic_outer;
  iree_uk_index_t input_stride_h;
  const void* filter_buffer;
  iree_uk_index_t filter_offset;
  iree_uk_index_t filter_stride_oc_outer;
  iree_uk_index_t filter_stride_ic_outer;
  iree_uk_index_t filter_stride_fh;
  iree_uk_index_t filter_stride_fw;
  void* output_buffer;
  iree_uk_index_t output_offset;
  iree_uk_index_t output_stride_n;
  iree_uk_index_t output_stride_oc_outer;
  iree_uk_index_t output_stride_oh;
  iree_uk_index_t N;
  iree_uk_index_t OC_outer;
  iree_uk_index_t OH;
  iree_uk_index_t OW;
  iree_uk_index_t IC_outer;
  iree_uk_index_t FH;
  iree_uk_index_t FW;
  iree_uk_int32_t k0;  // inner OC block
  iree_uk_int32_t c0;  // inner IC block
  iree_uk_int32_t stride_h;
  iree_uk_int32_t stride_w;
  iree_uk_uint32_t flags;
  // ow_count: how many output columns this particular tile call should produce.
  // Equals the selected tile's ow_tile for full tiles; smaller only on the
  // partial tail, as set by the driver.
  iree_uk_index_t ow_count;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_conv_nchwc_params_t;

// Equivalent to the public iree_uk_conv_nchwc entry point but taking the params
// struct.
void iree_uk_conv_nchwc_p(const iree_uk_conv_nchwc_params_t* params);

// Equivalent to the public iree_uk_conv_nchwc_info but operating on the params
// struct. Only the struct fields corresponding to iree_uk_conv_nchwc_info
// params are used.
iree_uk_uint32_t iree_uk_conv_nchwc_info_p(
    const iree_uk_conv_nchwc_params_t* params);

// TODO: extend to match mmt4d's coverage (S8S8S32, S16/U4 variants, F16,
// BF16) once those paths are exercised by the compiler-side encoding
// resolver. For now this ukernel is F32-only.
typedef enum iree_uk_conv_nchwc_type_t {
  iree_uk_conv_nchwc_type_f32f32f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_32, FLOAT_32, FLOAT_32),
} iree_uk_conv_nchwc_type_t;

static inline iree_uk_conv_nchwc_type_t iree_uk_conv_nchwc_type(
    iree_uk_uint32_t flags) {
  switch (flags & IREE_UK_FLAG_CONV_NCHWC_TYPE_MASK) {
    case IREE_UK_FLAG_CONV_NCHWC_TYPE_F32F32F32:
      return iree_uk_conv_nchwc_type_f32f32f32;
    default:
      // Work around a LLVM/riscv32 miscompile. Without the unreachable here,
      // returning (iree_uk_conv_nchwc_type_t)0 causes this whole switch
      // statement to be miscompiled by LLVM/riscv32 as if it were UB, as the 0
      // was passed to `iree_uk_type_bit_count(x)`, which evaluates to
      // `1<<(x - 3)`, which is UB if x<3.
#if defined(IREE_UK_COMPILER_CLANG) && defined(IREE_UK_ARCH_RISCV_32)
      __builtin_unreachable();
#endif
      // Shouldn't happen, validated earlier.
      return (iree_uk_conv_nchwc_type_t)0;
  }
}

static inline iree_uk_type_t iree_uk_conv_nchwc_input_type(
    iree_uk_conv_nchwc_type_t type) {
  return iree_uk_untie_type(0, type);
}

static inline iree_uk_type_t iree_uk_conv_nchwc_filter_type(
    iree_uk_conv_nchwc_type_t type) {
  return iree_uk_untie_type(1, type);
}

static inline iree_uk_type_t iree_uk_conv_nchwc_output_type(
    iree_uk_conv_nchwc_type_t type) {
  return iree_uk_untie_type(2, type);
}

// Function pointer that computes one (ow_count x k0) output panel.
// The tile function performs the full reduction over (ic_outer, fh, fw,
// c_inner) internally. The driver calls it once per (n, oc_outer, oh, ow-tile)
// output panel. When the reduction range is empty (IC_outer/FH/FW == 0), the
// driver still calls the tile function in the non-accumulate case, and the tile
// function must write zeros to the output panel.
typedef void (*iree_uk_conv_nchwc_tile_func_t)(
    void* IREE_UK_RESTRICT output_panel,
    const void* IREE_UK_RESTRICT input_panel,
    const void* IREE_UK_RESTRICT filter_panel,
    const iree_uk_conv_nchwc_params_t* params);

// A selected tile function plus the output-width tile size it was compiled
// for.
typedef struct iree_uk_conv_nchwc_tile_selection_t {
  iree_uk_conv_nchwc_tile_func_t tile_func;  // null if unavailable
  iree_uk_int32_t ow_tile;
} iree_uk_conv_nchwc_tile_selection_t;

// Architecture-specific implementation, or generic fallback returning null
iree_uk_conv_nchwc_tile_selection_t iree_uk_conv_nchwc_select_tile_func_arch(
    const iree_uk_conv_nchwc_params_t* params);

// Generic fallback.
iree_uk_conv_nchwc_tile_selection_t iree_uk_conv_nchwc_select_tile_func_generic(
    const iree_uk_conv_nchwc_params_t* params);

#endif  // IREE_BUILTINS_UKERNEL_CONV_NCHWC_INTERNAL_H_
