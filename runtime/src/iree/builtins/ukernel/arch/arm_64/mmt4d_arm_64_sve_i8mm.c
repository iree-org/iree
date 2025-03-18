#include <stdarg.h>
#include <stdint.h>

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"

// A microkernel, which uses VSCALE to cover for multiple iterations of the
// K-loop. Outputs a single 2x2 tile. Not used as other microkernels have better
// performance.
void iree_uk_mmt4d_tile_s8s8s32_2x2x8_arm_64_sve_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;

  const svbool_t mask_4xi32 = svwhilelt_b32(0, 4);
  svint32_t acc0 = params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE
                       ? svld1_s32(mask_4xi32, out_ptr)
                       : svdup_s32(0);

  svint32_t acc1 = svdup_s32(0);
  svint32_t acc2 = svdup_s32(0);
  svint32_t acc3 = svdup_s32(0);

  const iree_uk_uint64_t vscale = svcntb() / 16;
  iree_uk_index_t k = 0;

  while (k + 4 * vscale <= params->K) {
    svint8_t lhs0 = svld1_s8(svptrue_b8(), lhs_ptr);
    svint8_t rhs0 = svld1_s8(svptrue_b8(), rhs_ptr);
    lhs_ptr += 16 * vscale;
    rhs_ptr += 16 * vscale;

    svint8_t lhs1 = svld1_s8(svptrue_b8(), lhs_ptr);
    svint8_t rhs1 = svld1_s8(svptrue_b8(), rhs_ptr);
    acc0 = svmmla_s32(acc0, lhs0, rhs0);
    lhs_ptr += 16 * vscale;
    rhs_ptr += 16 * vscale;
    acc1 = svmmla_s32(acc1, lhs1, rhs1);

    svint8_t lhs2 = svld1_s8(svptrue_b8(), lhs_ptr);
    svint8_t rhs2 = svld1_s8(svptrue_b8(), rhs_ptr);
    acc2 = svmmla_s32(acc2, lhs2, rhs2);
    lhs_ptr += 16 * vscale;
    rhs_ptr += 16 * vscale;

    svint8_t lhs3 = svld1_s8(svptrue_b8(), lhs_ptr);
    svint8_t rhs3 = svld1_s8(svptrue_b8(), rhs_ptr);
    lhs_ptr += 16 * vscale;
    rhs_ptr += 16 * vscale;

    acc3 = svmmla_s32(acc3, lhs3, rhs3);

    k += 4 * vscale;
  }

  while (k + vscale <= params->K) {
    svint8_t lhs = svld1_s8(svptrue_b8(), lhs_ptr);
    svint8_t rhs = svld1_s8(svptrue_b8(), rhs_ptr);
    lhs_ptr += 16 * vscale;
    rhs_ptr += 16 * vscale;

    acc0 = svmmla_s32(acc0, lhs, rhs);

    k += vscale;
  }

  if (k < params->K) {
    const svbool_t mask_i8 = svwhilelt_b8_s64(0, 16 * (params->K - k));
    svint8_t lhs = svld1_s8(mask_i8, lhs_ptr);
    svint8_t rhs = svld1_s8(mask_i8, rhs_ptr);
    acc0 = svmmla_s32(acc0, lhs, rhs);
  }

  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc1);
  acc2 = svadd_s32_x(svptrue_b32(), acc2, acc3);
  acc0 = svadd_s32_x(svptrue_b32(), acc0, acc2);

  if (vscale > 1) {
    int32_t tmp[16];
    svst1(svptrue_b32(), tmp, acc0);
    for (iree_uk_index_t i = 1; i < vscale; ++i) {
      svint32_t t = svld1(mask_4xi32, tmp + 4 * i);
      acc0 = svadd_x(mask_4xi32, acc0, t);
    }
  }

  svst1(mask_4xi32, out_ptr, acc0);
}

// Helper functions to unpack a number of 2x2 tiles in a couple of SVE registers
// (representing tiles in the same row) and store them into the output tile rows
// pointed to by at `r0` and `r1`. There are versions with and without
// accumulation, as well as versions where the registers contain a single 1x2
// row.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_smmla_2x(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0,
    int32_t* IREE_UK_RESTRICT r1) {
  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);
  svuint64_t u = svuzp1(x, y);
  svuint64_t v = svuzp2(x, y);

  svint32_t uu = svreinterpret_s32(u);
  svint32_t vv = svreinterpret_s32(v);

  svst1(svptrue_b32(), r0, uu);
  svst1(svptrue_b32(), r1, vv);
}

// Variant of the above when the two registers represent two full rows.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_smmla_2x_rows(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0,
    int32_t* IREE_UK_RESTRICT r1) {
  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);

  svuint64_t u = svtrn1(x, y);
  svuint64_t v = svtrn2(x, y);

  svuint64_t uu = svuzp1(u, v);
  svuint64_t vv = svuzp2(u, v);

  a = svreinterpret_s32(uu);
  b = svreinterpret_s32(vv);

  svst1(svptrue_b32(), r0, a);
  svst1(svptrue_b32(), r1, b);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_smmla_1x(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0) {
  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);
  svuint64_t u = svuzp1(x, y);

  svint32_t uu = svreinterpret_s32(u);

  svst1(svptrue_b32(), r0, uu);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_accumulate_smmla_2x_rows(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0,
    int32_t* IREE_UK_RESTRICT r1) {
  svint32_t a0 = svld1(svptrue_b32(), r0);
  svint32_t b0 = svld1(svptrue_b32(), r1);

  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);

  svuint64_t u = svtrn1(x, y);
  svuint64_t v = svtrn2(x, y);

  svuint64_t uu = svuzp1(u, v);
  svuint64_t vv = svuzp2(u, v);

  a = svreinterpret_s32(uu);
  b = svreinterpret_s32(vv);

  a = svadd_x(svptrue_b32(), a0, a);
  b = svadd_x(svptrue_b32(), b0, b);

  svst1(svptrue_b32(), r0, a);
  svst1(svptrue_b32(), r1, b);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_accumulate_smmla_2x(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0,
    int32_t* IREE_UK_RESTRICT r1) {
  svint32_t a0 = svld1(svptrue_b32(), r0);
  svint32_t b0 = svld1(svptrue_b32(), r1);

  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);
  svuint64_t u = svuzp1(x, y);
  svuint64_t v = svuzp2(x, y);

  svint32_t uu = svreinterpret_s32(u);
  svint32_t vv = svreinterpret_s32(v);

  uu = svadd_x(svptrue_b32(), a0, uu);
  vv = svadd_x(svptrue_b32(), b0, vv);

  svst1(svptrue_b32(), r0, uu);
  svst1(svptrue_b32(), r1, vv);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void unpack_accumulate_smmla_1x(
    svint32_t a, svint32_t b, int32_t* IREE_UK_RESTRICT r0) {
  svint32_t a0 = svld1(svptrue_b32(), r0);

  svuint64_t x = svreinterpret_u64(a);
  svuint64_t y = svreinterpret_u64(b);
  svuint64_t u = svuzp1(x, y);

  svint32_t uu = svreinterpret_s32(u);

  uu = svadd_x(svptrue_b32(), a0, uu);

  svst1(svptrue_b32(), r0, uu);
}

// The next three microkernels work by replicating a single 2x8 LHS tile
// VSCALE times to fill a SVE register and multiplying it by several
// RHS SVE registers to compute two full rows of the output tile.
// The tile size and VSCALE are dependent on each other - the VSCALE must be
// such that the RHS registers (`rh0`, `rhs`, etc) cover the entire RHS tile.
// For example the "4x2VSx8" kernel should work correctly for
// 4x2x8/VSCALE == 1, 4x4x8/VSCALE == 2, 4x8x8/VSCALE == 4 (tested only with
// VSCALE == 2)
void iree_uk_mmt4d_tile_s8s8s32_4x4x8_arm_64_sve_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const iree_uk_int32_t M0 = params->M0;
  const iree_uk_int32_t N0 = params->N0;
  const iree_uk_int32_t K0 = params->K0;

  const svbool_t mask_2x8xi8 = svwhilelt_b8(0, 16);

  svint8_t rhs0;
  svint8_t lhs0, lhs1;
  svint32_t acc00, acc10;

  acc00 = acc10 = svdup_s32(0);

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    lhs0 = svld1(mask_2x8xi8, &lhs_ptr[k * M0 * K0]);
    lhs0 = svdupq_lane_s8(lhs0, 0);

    lhs1 = svld1(mask_2x8xi8, &lhs_ptr[k * M0 * K0 + 2 * K0]);
    lhs1 = svdupq_lane_s8(lhs1, 0);

    rhs0 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0]);

    acc00 = svmmla_s32(acc00, lhs0, rhs0);
    acc10 = svmmla_s32(acc10, lhs1, rhs0);
  }

  if (IREE_UK_UNLIKELY(params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE))
    unpack_accumulate_smmla_2x_rows(acc00, acc10, out_ptr, out_ptr + 2 * N0);
  else
    unpack_smmla_2x_rows(acc00, acc10, out_ptr, out_ptr + 2 * N0);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void
iree_uk_mmt4d_tile_s8s8s32_1x4VSx8_to_8x4VSx8_arm_64_sve_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_int32_t M0) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const iree_uk_int32_t N0 = params->N0;
  const iree_uk_int32_t K0 = params->K0;

  const svbool_t lhs_mask = M0 == 1 ? svwhilelt_b8(0, 8) : svwhilelt_b8(0, 16);

  svint8_t rhs0, rhs1;
  svint8_t lhs0, lhs1, lhs2, lhs3;
  svint32_t acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;

  acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = acc30 = acc31 = svdup_s32(0);

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    rhs0 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0]);
    rhs1 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0 + 2 * 16]);

    lhs0 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0]);
    lhs0 = svdupq_lane_s8(lhs0, 0);
    acc00 = svmmla_s32(acc00, lhs0, rhs0);
    acc01 = svmmla_s32(acc01, lhs0, rhs1);

    if (M0 > 2) {
      lhs1 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 2 * K0]);
      lhs1 = svdupq_lane_s8(lhs1, 0);
      acc10 = svmmla_s32(acc10, lhs1, rhs0);
      acc11 = svmmla_s32(acc11, lhs1, rhs1);
    }

    if (M0 > 4) {
      lhs2 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 4 * K0]);
      lhs2 = svdupq_lane_s8(lhs2, 0);
      acc20 = svmmla_s32(acc20, lhs2, rhs0);
      acc21 = svmmla_s32(acc21, lhs2, rhs1);
    }

    if (M0 > 6) {
      lhs3 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 6 * K0]);
      lhs3 = svdupq_lane_s8(lhs3, 0);
      acc30 = svmmla_s32(acc30, lhs3, rhs0);
      acc31 = svmmla_s32(acc31, lhs3, rhs1);
    }
  }

  if (IREE_UK_UNLIKELY(params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)) {
    if (M0 == 1)
      unpack_accumulate_smmla_1x(acc00, acc01, out_ptr);
    else
      unpack_accumulate_smmla_2x(acc00, acc01, out_ptr, out_ptr + N0);

    if (M0 > 2)
      unpack_accumulate_smmla_2x(acc10, acc11, out_ptr + 2 * N0,
                                 out_ptr + 3 * N0);
    if (M0 > 4)
      unpack_accumulate_smmla_2x(acc20, acc21, out_ptr + 4 * N0,
                                 out_ptr + 5 * N0);
    if (M0 > 6)
      unpack_accumulate_smmla_2x(acc30, acc31, out_ptr + 6 * N0,
                                 out_ptr + 7 * N0);
  } else {
    if (M0 == 1)
      unpack_smmla_1x(acc00, acc01, out_ptr);
    else
      unpack_smmla_2x(acc00, acc01, out_ptr, out_ptr + N0);

    if (M0 > 2)
      unpack_smmla_2x(acc10, acc11, out_ptr + 2 * N0, out_ptr + 3 * N0);

    if (M0 > 4)
      unpack_smmla_2x(acc20, acc21, out_ptr + 4 * N0, out_ptr + 5 * N0);

    if (M0 > 6)
      unpack_smmla_2x(acc30, acc31, out_ptr + 6 * N0, out_ptr + 7 * N0);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x4VSx8_to_8x4VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_1x8x8_arm_64_sve_i8mm, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x4VSx8_to_8x4VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_2x8x8_arm_64_sve_i8mm, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x4VSx8_to_8x4VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_4x8x8_arm_64_sve_i8mm, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x4VSx8_to_8x4VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_8x8x8_arm_64_sve_i8mm, 8)

// This microkernel does not curently work when invoked by IREE since IREE
// passes it sliced tiles.
IREE_UK_ATTRIBUTE_ALWAYS_INLINE static void
iree_uk_mmt4d_tile_s8s8s32_1x8VSx8_to_8x8VSx8_arm_64_sve_i8mm(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_int32_t M0) {
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

  const iree_uk_int32_t N0 = params->N0;
  const iree_uk_int32_t K0 = params->K0;

  const svbool_t lhs_mask = M0 > 1 ? svwhilelt_b8(0, 16) : svwhilelt_b8(0, 8);

  svint32_t acc00, acc01, acc02, acc03;
  svint32_t acc10, acc11, acc12, acc13;
  svint32_t acc20, acc21, acc22, acc23;
  svint32_t acc30, acc31, acc32, acc33;

  acc00 = acc01 = acc02 = acc03 = svdup_s32(0);
  acc10 = acc11 = acc12 = acc13 = svdup_s32(0);
  acc20 = acc21 = acc22 = acc23 = svdup_s32(0);
  acc30 = acc31 = acc32 = acc33 = svdup_s32(0);

  for (iree_uk_index_t k = 0; k < params->K; ++k) {
    svint8_t rhs0 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0]);
    svint8_t rhs1 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0 + 2 * 16]);
    svint8_t rhs2 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0 + 4 * 16]);
    svint8_t rhs3 = svld1(svptrue_b8(), &rhs_ptr[k * N0 * K0 + 6 * 16]);

    svint8_t lhs0 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0]);
    lhs0 = svdupq_lane_s8(lhs0, 0);
    acc00 = svmmla_s32(acc00, lhs0, rhs0);
    acc01 = svmmla_s32(acc01, lhs0, rhs1);
    acc02 = svmmla_s32(acc02, lhs0, rhs2);
    acc03 = svmmla_s32(acc03, lhs0, rhs3);

    if (M0 > 2) {
      svint8_t lhs1 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 2 * K0]);
      lhs1 = svdupq_lane_s8(lhs1, 0);
      acc10 = svmmla_s32(acc10, lhs1, rhs0);
      acc11 = svmmla_s32(acc11, lhs1, rhs1);
      acc12 = svmmla_s32(acc12, lhs1, rhs2);
      acc13 = svmmla_s32(acc13, lhs1, rhs3);
    }

    if (M0 > 4) {
      svint8_t lhs2 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 4 * K0]);
      lhs2 = svdupq_lane_s8(lhs2, 0);
      acc20 = svmmla_s32(acc20, lhs2, rhs0);
      acc21 = svmmla_s32(acc21, lhs2, rhs1);
      acc22 = svmmla_s32(acc22, lhs2, rhs2);
      acc23 = svmmla_s32(acc23, lhs2, rhs3);
    }

    if (M0 > 6) {
      svint8_t lhs3 = svld1(lhs_mask, &lhs_ptr[k * M0 * K0 + 6 * K0]);
      lhs3 = svdupq_lane_s8(lhs3, 0);
      acc30 = svmmla_s32(acc30, lhs3, rhs0);
      acc31 = svmmla_s32(acc31, lhs3, rhs1);
      acc32 = svmmla_s32(acc32, lhs3, rhs2);
      acc33 = svmmla_s32(acc33, lhs3, rhs3);
    }
  }

  iree_uk_index_t i, j;
  iree_uk_index_t vscale = svcntd() / 2;
  if (IREE_UK_UNLIKELY(params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE)) {
    i = 0;
    if (M0 == 1) {
      j = 0;
      unpack_accumulate_smmla_1x(acc00, acc01, out_ptr + i * N0 + j);
      j = 2 * 2 * vscale;
      unpack_accumulate_smmla_1x(acc02, acc03, out_ptr + i * N0 + j);
    } else {
      j = 0;
      unpack_accumulate_smmla_2x(acc00, acc01, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_accumulate_smmla_2x(acc02, acc03, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
    }

    if (M0 > 2) {
      i = 2;
      j = 0;
      unpack_accumulate_smmla_2x(acc10, acc11, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_accumulate_smmla_2x(acc12, acc13, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
    }

    if (M0 > 4) {
      i = 4;
      j = 0;
      unpack_accumulate_smmla_2x(acc20, acc21, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_accumulate_smmla_2x(acc22, acc23, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
    }

    if (M0 > 6) {
      i = 6;
      j = 0;
      unpack_accumulate_smmla_2x(acc30, acc31, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_accumulate_smmla_2x(acc32, acc33, out_ptr + i * N0 + j,
                                 out_ptr + (i + 1) * N0 + j);
    }
  } else {
    i = 0;
    if (M0 == 1) {
      j = 0;
      unpack_smmla_1x(acc00, acc01, out_ptr + i * N0 + j);
      j = 2 * 2 * vscale;
      unpack_smmla_1x(acc02, acc03, out_ptr + i * N0 + j);
    } else {
      j = 0;
      unpack_smmla_2x(acc00, acc01, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_smmla_2x(acc02, acc03, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
    }

    if (M0 > 2) {
      i = 2;
      j = 0;
      unpack_smmla_2x(acc10, acc11, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_smmla_2x(acc12, acc13, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
    }

    if (M0 > 4) {
      i = 4;
      j = 0;
      unpack_smmla_2x(acc20, acc21, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_smmla_2x(acc22, acc23, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
    }
    if (M0 > 6) {
      i = 6;
      j = 0;
      unpack_smmla_2x(acc30, acc31, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
      j = 2 * 2 * vscale;
      unpack_smmla_2x(acc32, acc33, out_ptr + i * N0 + j,
                      out_ptr + (i + 1) * N0 + j);
    }
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8VSx8_to_8x8VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_1x16x8_arm_64_sve_i8mm, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8VSx8_to_8x8VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_2x16x8_arm_64_sve_i8mm, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8VSx8_to_8x8VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_4x16x8_arm_64_sve_i8mm, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_s8s8s32_1x8VSx8_to_8x8VSx8_arm_64_sve_i8mm,
    iree_uk_mmt4d_tile_s8s8s32_8x16x8_arm_64_sve_i8mm, 8)
