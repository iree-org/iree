// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d.h"

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/mmt4d_internal.h"

static void iree_uk_mmt4d_validate(const iree_uk_mmt4d_params_t* params) {
#ifdef IREE_UK_ENABLE_ASSERTS
  const iree_uk_uint32_t allflags =
      IREE_UK_FLAG_MMT4D_TYPE_MASK | IREE_UK_FLAG_MMT4D_ACCUMULATE |
      IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS |
      IREE_UK_FLAG_MMT4D_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION;
  IREE_UK_ASSERT(!(params->flags & ~allflags));
  iree_uk_uint32_t flags_type = params->flags & IREE_UK_FLAG_MMT4D_TYPE_MASK;
  IREE_UK_ASSERT(flags_type < IREE_UK_FLAG_MMT4D_TYPE_END);
  // Some implementations may wish to avoid supporting absurdly wide types. For
  // instance, K is the innermost (i.e. hottest) loop bound, so some 32bit
  // targets may benefit from K being int32, not int64. We still let K be of
  // type int64 to be future-proof, as types are hard to change later. But we
  // enforce a narrower range here, as we can always relax that later as needed.
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->M, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->N, 31));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->K, 31));
  // int32 is overkill for the inner tile sizes. Enforce int16 range for now.
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->M0, 15));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->N0, 15));
  IREE_UK_ASSERT(IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(params->K0, 15));

  // Requirements on sub-byte element type cases
  // - Ensure that the output type is not sub-byte.
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->flags);
  IREE_UK_ASSERT(iree_uk_type_bit_count(iree_uk_mmt4d_out_type(mmt4d_type)) >=
                 8);
  // - Ensure that (K0 * {LHS,RHS} element bits) is a multiple of 8 bits.
  int lhs_bits = iree_uk_type_bit_count(iree_uk_mmt4d_lhs_type(mmt4d_type));
  int rhs_bits = iree_uk_type_bit_count(iree_uk_mmt4d_rhs_type(mmt4d_type));
  IREE_UK_ASSERT(!((params->K0 * lhs_bits) % 8));
  IREE_UK_ASSERT(!((params->K0 * rhs_bits) % 8));
  // - Ensure that {LHS,RHS} strides are multiples of 8 bits.
  IREE_UK_ASSERT(!((params->lhs_stride0 * lhs_bits) % 8));
  IREE_UK_ASSERT(!((params->rhs_stride0 * rhs_bits) % 8));
#endif  // IREE_UK_ENABLE_ASSERTS
}

// General mmt4d implementation, shared among all cases. The idea is that the
// only really performance-critical part is the inner-most loop, and that's
// handled by the tile_func passed as argument here. Sharing the outer loops
// across all cases is a roughly 2x code shrink compared to if we were
// emitting the whole loop nest for each case.
static void iree_uk_mmt4d_using_tile_func(const iree_uk_mmt4d_params_t* params,
                                          iree_uk_mmt4d_tile_func_t tile_func) {
  const iree_uk_int32_t M = params->M;
  const iree_uk_int32_t N = params->N;
  const iree_uk_int16_t M0 = params->M0;
  const iree_uk_int16_t N0 = params->N0;
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->flags);
  const iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(mmt4d_type);
  const iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(mmt4d_type);
  const iree_uk_type_t out_type = iree_uk_mmt4d_out_type(mmt4d_type);
  const iree_uk_int16_t lhs_elem_bits_log2 =
      iree_uk_type_bit_count_log2(lhs_type);
  const iree_uk_int16_t rhs_elem_bits_log2 =
      iree_uk_type_bit_count_log2(rhs_type);
  const iree_uk_int16_t out_elem_size_log2 = iree_uk_type_size_log2(out_type);
  char* out_tile_row =
      (char*)params->out_buffer + (params->out_offset << out_elem_size_log2);
  const char* lhs_panel =
      (const char*)params->lhs_buffer +
      iree_uk_bits_to_bytes_exact(params->lhs_offset << lhs_elem_bits_log2);
  const char* rhs_panel_start =
      (const char*)params->rhs_buffer +
      iree_uk_bits_to_bytes_exact(params->rhs_offset << rhs_elem_bits_log2);
  iree_uk_int32_t out_tile_size = (M0 * N0) << out_elem_size_log2;
  iree_uk_index_t lhs_panel_stride =
      iree_uk_bits_to_bytes_exact(params->lhs_stride0 << lhs_elem_bits_log2);
  iree_uk_index_t rhs_panel_stride =
      iree_uk_bits_to_bytes_exact(params->rhs_stride0 << rhs_elem_bits_log2);
  iree_uk_index_t out_stride = params->out_stride0 << out_elem_size_log2;
  for (iree_uk_int32_t i = 0; i < M; ++i) {
    char* out_tile = out_tile_row;
    const char* rhs_panel = rhs_panel_start;
    // Prefetches needed on ARM Cortex-X2, Issue #13332.
    IREE_UK_PREFETCH_RW(out_tile_row, IREE_UK_PREFETCH_LOCALITY_L3);
    IREE_UK_PREFETCH_RO(lhs_panel, IREE_UK_PREFETCH_LOCALITY_L1);
    IREE_UK_PREFETCH_RO(rhs_panel, IREE_UK_PREFETCH_LOCALITY_L1);
    for (iree_uk_int32_t j = 0; j < N; ++j) {
      tile_func(out_tile, lhs_panel, rhs_panel, params);
      out_tile += out_tile_size;
      rhs_panel += rhs_panel_stride;
    }
    out_tile_row += out_stride;
    lhs_panel += lhs_panel_stride;
  }
}

// Early-return code paths, including trivial or near-trivial cases (when one
// of the dimensions is 0) and in the future, hardware ports that specialize
// the entire loop nest.
// Returns true if already done.
static bool iree_uk_mmt4d_early(const iree_uk_mmt4d_params_t* params) {
  // Trivial cases
  if (params->M == 0 || params->N == 0 ||
      (params->K == 0 && params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)) {
    return true;
  }
  // Targets that want to specialize the entire loop nest can do so here.
  return false;
}

void iree_uk_mmt4d_p(const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_validate(params);

  // Maybe handle this mmt4d "early", without needing to select a tile_func.
  // Typical cases include trivial cases (e.g. when params->K == 0) and hardware
  // targets that want to handle the entire loop nest in target-specific code.
  if (iree_uk_mmt4d_early(params)) return;

  // Select a target-specific tile_func (inner loop on K, computing one M0xN0
  // tile) and use that with generic outer loops.
  iree_uk_mmt4d_tile_func_t tile_func =
      iree_uk_mmt4d_select_tile_func_arch(params);

  // If no target-specific tile_func is available, fall back to a generic one if
  // allowed by the flags.
  if (!tile_func) {
    if (params->flags &
        IREE_UK_FLAG_MMT4D_ALLOW_GENERIC_FALLBACK_TILE_FUNCTION) {
      tile_func = iree_uk_mmt4d_select_tile_func_generic(params);
    } else {
      IREE_UK_ASSERT(
          0 && "no target-specific tile function, and fallback not enabled.");
    }
  }

  iree_uk_mmt4d_using_tile_func(params, tile_func);
}

iree_uk_uint32_t iree_uk_mmt4d_info_p(const iree_uk_mmt4d_params_t* params) {
  iree_uk_uint32_t result = 0;
  if (iree_uk_mmt4d_select_tile_func_arch(params)) {
    result |= IREE_UK_FLAG_MMT4D_INFO_HAVE_ARCHITECTURE_SPECIFIC_TILE_FUNCTION;
  }
  return result;
}

IREE_UK_EXPORT void iree_uk_mmt4d(
    const void* lhs_buffer, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, const void* rhs_buffer,
    iree_uk_index_t rhs_offset, iree_uk_index_t rhs_stride0, void* out_buffer,
    iree_uk_index_t out_offset, iree_uk_index_t out_stride0, iree_uk_index_t M,
    iree_uk_index_t N, iree_uk_index_t K, iree_uk_int32_t M0,
    iree_uk_int32_t N0, iree_uk_int32_t K0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data) {
  iree_uk_mmt4d_params_t params = {.lhs_buffer = lhs_buffer,
                                   .lhs_offset = lhs_offset,
                                   .lhs_stride0 = lhs_stride0,
                                   .rhs_buffer = rhs_buffer,
                                   .rhs_offset = rhs_offset,
                                   .rhs_stride0 = rhs_stride0,
                                   .out_buffer = out_buffer,
                                   .out_offset = out_offset,
                                   .out_stride0 = out_stride0,
                                   .M = M,
                                   .N = N,
                                   .K = K,
                                   .M0 = M0,
                                   .N0 = N0,
                                   .K0 = K0,
                                   .flags = flags,
                                   .cpu_data = cpu_data};
  iree_uk_mmt4d_p(&params);
}

IREE_UK_EXPORT iree_uk_uint32_t
iree_uk_mmt4d_info(iree_uk_int32_t M0, iree_uk_int32_t N0, iree_uk_int32_t K0,
                   iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data) {
  iree_uk_mmt4d_params_t params = {
      .M0 = M0, .N0 = N0, .K0 = K0, .flags = flags, .cpu_data = cpu_data};
  return iree_uk_mmt4d_info_p(&params);
}
