// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SpaceMiT IME (XSMTVDot) s8s8s32 mmt4d.
//
// One `smt.vmadot` at SEW=8 is a square MAC atom selected by vl (not VLMAX):
//   atom=4  (vl*SEW=256)  -> 4x4x8   -> tile 12x16x8
//   atom=8  (vl*SEW=1024) -> 8x8x16  -> tile 24x32x16
//   atom=16 (vl*SEW=4096) -> 16x16x32 -> tile 48x64x32
// The primary macro-tile is a fixed 3x4 atom grid (12 vmadot per K panel).
// Narrow truncations of the same grid (2x4 and 1x4 atom rows) are also
// implemented below, sharing the same N0 = 4*atom and K0 = 2*atom as their
// bucket's primary tile and only shrinking the number of atom-rows (MT).
// Each {vd,vd+1} int32 accumulator group holds one atom x atom output block,
// row-major and contiguous within the group, and stays resident across K.
//
// Panels:
//   lhs : int8 [K1][M0][K0]   K0-contiguous
//   rhs : int8 [K1][N0][K0]   transposed (N0,K0), K0-contiguous
//   out : int32 [M0][N0]

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"

enum { IME_NT = 4 };

// Byte-offset index vector mapping accumulator-group element i (row-major
// atom x atom, i = r*atom + c) to its position in the strided (M0,N0) output
// block: out offset (in elements) = r*N0 + c = i + (N0 - atom) * (i / atom).
static inline vuint32m2_t ime_out_index(int N0, int atom, int log2_atom,
                                        size_t vl32) {
  vuint32m2_t vi = __riscv_vid_v_u32m2(vl32);
  vuint32m2_t row = __riscv_vsrl_vx_u32m2(vi, (size_t)log2_atom, vl32);
  vuint32m2_t off =
      __riscv_vmacc_vx_u32m2(vi, (uint32_t)(N0 - atom), row, vl32);
  return __riscv_vsll_vx_u32m2(off, 2, vl32);
}

// Shared 3x4-atom (primary) xsmtvdot body. The macro-tile is a fixed 3x4
// grid of the SEW=8 MAC atom, so M0 alone determines the atom edge (4 / 8 /
// 16) and hence N0 = 4*atom and K0 = 2*atom.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_12xXXx8_to_48xXXx32_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  enum { MT = 3 };
  const int atom = M0 / MT;
  const int N0 = IME_NT * atom;
  const int K0 = 2 * atom;
  IREE_UK_ASSERT(atom == 4 || atom == 8 || atom == 16);
  IREE_UK_ASSERT(params->M0 == M0);
  IREE_UK_ASSERT(params->N0 == N0);
  IREE_UK_ASSERT(params->K0 == K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  const int K1 = (int)(params->K);
  const int accumulate = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) != 0;
  const int atom_elems = atom * atom;
  const int panel_stride = atom * K0;
  const int log2_atom = __builtin_ctz((unsigned int)atom);
  const size_t vl8 = (size_t)panel_stride;
  const size_t vl32 = (size_t)atom_elems;

  // out_tile element address of accumulator group (mt,nt), where
  // acc<mt*IME_NT+nt> holds A[mt] . B[nt].
#define IME_OUT(mt, nt) (out + ((mt) * atom) * N0 + (nt) * atom)

  vint32m2_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10,
      acc11;
  if (accumulate) {
    vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
    acc0 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 0), idx, vl32);
    acc1 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 1), idx, vl32);
    acc2 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 2), idx, vl32);
    acc3 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 3), idx, vl32);
    acc4 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 0), idx, vl32);
    acc5 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 1), idx, vl32);
    acc6 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 2), idx, vl32);
    acc7 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 3), idx, vl32);
    acc8 = __riscv_vluxei32_v_i32m2(IME_OUT(2, 0), idx, vl32);
    acc9 = __riscv_vluxei32_v_i32m2(IME_OUT(2, 1), idx, vl32);
    acc10 = __riscv_vluxei32_v_i32m2(IME_OUT(2, 2), idx, vl32);
    acc11 = __riscv_vluxei32_v_i32m2(IME_OUT(2, 3), idx, vl32);
  } else {
    acc0 = __riscv_vmv_v_x_i32m2(0, vl32);
    acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = acc8 = acc9 = acc10 =
        acc11 = acc0;
  }

  for (int k = 0; k < K1; ++k) {
    vint8m1_t a0 = __riscv_vle8_v_i8m1(lhs + 0 * panel_stride, vl8);
    vint8m1_t a1 = __riscv_vle8_v_i8m1(lhs + 1 * panel_stride, vl8);
    vint8m1_t a2 = __riscv_vle8_v_i8m1(lhs + 2 * panel_stride, vl8);
    vint8m1_t b0 = __riscv_vle8_v_i8m1(rhs + 0 * panel_stride, vl8);
    vint8m1_t b1 = __riscv_vle8_v_i8m1(rhs + 1 * panel_stride, vl8);
    vint8m1_t b2 = __riscv_vle8_v_i8m1(rhs + 2 * panel_stride, vl8);
    vint8m1_t b3 = __riscv_vle8_v_i8m1(rhs + 3 * panel_stride, vl8);
    lhs += MT * panel_stride;
    rhs += IME_NT * panel_stride;

    __asm__ volatile(
        "    smt.vmadot  %[c0],  %[a0], %[b0]       \n\t"
        "    smt.vmadot  %[c1],  %[a0], %[b1]       \n\t"
        "    smt.vmadot  %[c2],  %[a0], %[b2]       \n\t"
        "    smt.vmadot  %[c3],  %[a0], %[b3]       \n\t"
        "    smt.vmadot  %[c4],  %[a1], %[b0]       \n\t"
        "    smt.vmadot  %[c5],  %[a1], %[b1]       \n\t"
        "    smt.vmadot  %[c6],  %[a1], %[b2]       \n\t"
        "    smt.vmadot  %[c7],  %[a1], %[b3]       \n\t"
        "    smt.vmadot  %[c8],  %[a2], %[b0]       \n\t"
        "    smt.vmadot  %[c9],  %[a2], %[b1]       \n\t"
        "    smt.vmadot  %[c10], %[a2], %[b2]       \n\t"
        "    smt.vmadot  %[c11], %[a2], %[b3]       \n\t"
        : [c0] "+vr"(acc0), [c1] "+vr"(acc1), [c2] "+vr"(acc2),
          [c3] "+vr"(acc3), [c4] "+vr"(acc4), [c5] "+vr"(acc5),
          [c6] "+vr"(acc6), [c7] "+vr"(acc7), [c8] "+vr"(acc8),
          [c9] "+vr"(acc9), [c10] "+vr"(acc10), [c11] "+vr"(acc11)
        : [a0] "vr"(a0), [a1] "vr"(a1), [a2] "vr"(a2), [b0] "vr"(b0),
          [b1] "vr"(b1), [b2] "vr"(b2), [b3] "vr"(b3)
        :);
  }

  vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 0), idx, acc0, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 1), idx, acc1, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 2), idx, acc2, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 3), idx, acc3, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 0), idx, acc4, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 1), idx, acc5, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 2), idx, acc6, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 3), idx, acc7, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(2, 0), idx, acc8, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(2, 1), idx, acc9, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(2, 2), idx, acc10, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(2, 3), idx, acc11, vl32);

#undef IME_OUT
}

// Shared 2x4-atom (narrow) xsmtvdot body: truncation of the 3x4 primary grid
// down to 2 atom-rows. Same N0 = 4*atom and K0 = 2*atom as the matching
// primary tile for a given atom (VLEN bucket); only M0 = 2*atom shrinks.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_8xXXx8_to_32xXXx32_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  enum { MT = 2 };
  const int atom = M0 / MT;
  const int N0 = IME_NT * atom;
  const int K0 = 2 * atom;
  IREE_UK_ASSERT(atom == 4 || atom == 8 || atom == 16);
  IREE_UK_ASSERT(params->M0 == M0);
  IREE_UK_ASSERT(params->N0 == N0);
  IREE_UK_ASSERT(params->K0 == K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  const int K1 = (int)(params->K);
  const int accumulate = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) != 0;
  const int atom_elems = atom * atom;
  const int panel_stride = atom * K0;
  const int log2_atom = __builtin_ctz((unsigned int)atom);
  const size_t vl8 = (size_t)panel_stride;
  const size_t vl32 = (size_t)atom_elems;

#define IME_OUT(mt, nt) (out + ((mt) * atom) * N0 + (nt) * atom)

  vint32m2_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  if (accumulate) {
    vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
    acc0 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 0), idx, vl32);
    acc1 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 1), idx, vl32);
    acc2 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 2), idx, vl32);
    acc3 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 3), idx, vl32);
    acc4 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 0), idx, vl32);
    acc5 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 1), idx, vl32);
    acc6 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 2), idx, vl32);
    acc7 = __riscv_vluxei32_v_i32m2(IME_OUT(1, 3), idx, vl32);
  } else {
    acc0 = __riscv_vmv_v_x_i32m2(0, vl32);
    acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = acc0;
  }

  for (int k = 0; k < K1; ++k) {
    vint8m1_t a0 = __riscv_vle8_v_i8m1(lhs + 0 * panel_stride, vl8);
    vint8m1_t a1 = __riscv_vle8_v_i8m1(lhs + 1 * panel_stride, vl8);
    vint8m1_t b0 = __riscv_vle8_v_i8m1(rhs + 0 * panel_stride, vl8);
    vint8m1_t b1 = __riscv_vle8_v_i8m1(rhs + 1 * panel_stride, vl8);
    vint8m1_t b2 = __riscv_vle8_v_i8m1(rhs + 2 * panel_stride, vl8);
    vint8m1_t b3 = __riscv_vle8_v_i8m1(rhs + 3 * panel_stride, vl8);
    lhs += MT * panel_stride;
    rhs += IME_NT * panel_stride;

    __asm__ volatile(
        "    smt.vmadot  %[c0], %[a0], %[b0]        \n\t"
        "    smt.vmadot  %[c1], %[a0], %[b1]        \n\t"
        "    smt.vmadot  %[c2], %[a0], %[b2]        \n\t"
        "    smt.vmadot  %[c3], %[a0], %[b3]        \n\t"
        "    smt.vmadot  %[c4], %[a1], %[b0]        \n\t"
        "    smt.vmadot  %[c5], %[a1], %[b1]        \n\t"
        "    smt.vmadot  %[c6], %[a1], %[b2]        \n\t"
        "    smt.vmadot  %[c7], %[a1], %[b3]        \n\t"
        : [c0] "+vr"(acc0), [c1] "+vr"(acc1), [c2] "+vr"(acc2),
          [c3] "+vr"(acc3), [c4] "+vr"(acc4), [c5] "+vr"(acc5),
          [c6] "+vr"(acc6), [c7] "+vr"(acc7)
        : [a0] "vr"(a0), [a1] "vr"(a1), [b0] "vr"(b0), [b1] "vr"(b1),
          [b2] "vr"(b2), [b3] "vr"(b3)
        :);
  }

  vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 0), idx, acc0, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 1), idx, acc1, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 2), idx, acc2, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 3), idx, acc3, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 0), idx, acc4, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 1), idx, acc5, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 2), idx, acc6, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(1, 3), idx, acc7, vl32);

#undef IME_OUT
}

// Shared 1x4-atom (narrow) xsmtvdot body: truncation of the 3x4 primary grid
// down to a single atom-row.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_s8s8s32_4xXXx8_to_16xXXx32_riscv_64_xsmtvdot(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  enum { MT = 1 };
  const int atom = M0 / MT;
  const int N0 = IME_NT * atom;
  const int K0 = 2 * atom;
  IREE_UK_ASSERT(atom == 4 || atom == 8 || atom == 16);
  IREE_UK_ASSERT(params->M0 == M0);
  IREE_UK_ASSERT(params->N0 == N0);
  IREE_UK_ASSERT(params->K0 == K0);

  const iree_uk_int8_t* IREE_UK_RESTRICT lhs = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out = out_tile;

  const int K1 = (int)(params->K);
  const int accumulate = (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) != 0;
  const int atom_elems = atom * atom;
  const int panel_stride = atom * K0;
  const int log2_atom = __builtin_ctz((unsigned int)atom);
  const size_t vl8 = (size_t)panel_stride;
  const size_t vl32 = (size_t)atom_elems;

#define IME_OUT(mt, nt) (out + ((mt) * atom) * N0 + (nt) * atom)

  vint32m2_t acc0, acc1, acc2, acc3;
  if (accumulate) {
    vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
    acc0 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 0), idx, vl32);
    acc1 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 1), idx, vl32);
    acc2 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 2), idx, vl32);
    acc3 = __riscv_vluxei32_v_i32m2(IME_OUT(0, 3), idx, vl32);
  } else {
    acc0 = __riscv_vmv_v_x_i32m2(0, vl32);
    acc1 = acc2 = acc3 = acc0;
  }

  for (int k = 0; k < K1; ++k) {
    vint8m1_t a0 = __riscv_vle8_v_i8m1(lhs, vl8);
    vint8m1_t b0 = __riscv_vle8_v_i8m1(rhs + 0 * panel_stride, vl8);
    vint8m1_t b1 = __riscv_vle8_v_i8m1(rhs + 1 * panel_stride, vl8);
    vint8m1_t b2 = __riscv_vle8_v_i8m1(rhs + 2 * panel_stride, vl8);
    vint8m1_t b3 = __riscv_vle8_v_i8m1(rhs + 3 * panel_stride, vl8);
    lhs += MT * panel_stride;
    rhs += IME_NT * panel_stride;

    __asm__ volatile(
        "    smt.vmadot  %[c0], %[a0], %[b0]        \n\t"
        "    smt.vmadot  %[c1], %[a0], %[b1]        \n\t"
        "    smt.vmadot  %[c2], %[a0], %[b2]        \n\t"
        "    smt.vmadot  %[c3], %[a0], %[b3]        \n\t"
        : [c0] "+vr"(acc0), [c1] "+vr"(acc1), [c2] "+vr"(acc2), [c3] "+vr"(acc3)
        : [a0] "vr"(a0), [b0] "vr"(b0), [b1] "vr"(b1), [b2] "vr"(b2),
          [b3] "vr"(b3)
        :);
  }

  vuint32m2_t idx = ime_out_index(N0, atom, log2_atom, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 0), idx, acc0, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 1), idx, acc1, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 2), idx, acc2, vl32);
  __riscv_vsuxei32_v_i32m2(IME_OUT(0, 3), idx, acc3, vl32);

#undef IME_OUT
}

// Primary 3x4-atom tiles.
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_to_48xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_riscv_64_xsmtvdot_zvl256b, 12)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_to_48xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_24xXXx16_riscv_64_xsmtvdot_zvl1024b, 24)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_12xXXx8_to_48xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_48xXXx32_riscv_64_xsmtvdot_zvl4096b, 48)

// Narrow 2x4-atom tiles.
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_8xXXx8_to_32xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx8_riscv_64_xsmtvdot_zvl256b, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_8xXXx8_to_32xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_16xXXx16_riscv_64_xsmtvdot_zvl1024b, 16)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_8xXXx8_to_32xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_32xXXx32_riscv_64_xsmtvdot_zvl4096b, 32)

// Narrow 1x4-atom tiles.
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4xXXx8_to_16xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_4xXXx8_riscv_64_xsmtvdot_zvl256b, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4xXXx8_to_16xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_8xXXx16_riscv_64_xsmtvdot_zvl1024b, 8)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_4xXXx8_to_16xXXx32_riscv_64_xsmtvdot,
    iree_uk_mmt4d_tile_s8s8s32_16xXXx32_riscv_64_xsmtvdot_zvl4096b, 16)
