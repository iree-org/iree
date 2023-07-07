// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/elementwise.h"

// TODO: We should only be including/using this in standalone builds. In others,
// we have to emulate or use other mechanisms. Since this file only contains
// fallback implementations, we don't care about the quality *that* much but
// would still like to avoid the libc dep for compatibility with the bitcode
// path.
#include <math.h>

//===----------------------------------------------------------------------===//
// Helpers for defining generic implementations of elementwise functions.
// Since it affords the best code size tradeoff options, the entrypoint
// is dispatched based on an opcode.
//===----------------------------------------------------------------------===//

// Opcodes for generic functions operating on 32-bit operands and result.
// Since the outer dispatcher only differentiates based on width, all other
// type specificity is carried by the opcode.
// Binary opcodes are named "X32B" and unary opcodes "X32U".
// The initial list was sorted, and it is encouraged to sort extensions, but
// each opcode must be numerically stable, so the list is not expected to
// be sorted over time.
typedef enum {
  IREE_UK_X32B_ADDF = 0,
  IREE_UK_X32B_ADDI = 1,
  IREE_UK_X32B_ANDI = 2,
  IREE_UK_X32B_DIVF = 3,
  IREE_UK_X32B_DIVSI = 4,
  IREE_UK_X32B_DIVUI = 5,
  IREE_UK_X32B_MULF = 6,
  IREE_UK_X32B_MULI = 7,
  IREE_UK_X32B_ORI = 8,
  IREE_UK_X32B_SHLI = 9,
  IREE_UK_X32B_SHRSI = 10,
  IREE_UK_X32B_SHRUI = 11,
  IREE_UK_X32B_SUBF = 12,
  IREE_UK_X32B_SUBI = 13,
  IREE_UKENREL_X32B_XORI = 14,
} iree_uk_x32b_opcode_t;

typedef enum {
  IREE_UK_X32U_ABSF,
  IREE_UK_X32U_CEILF,
  IREE_UK_X32U_CTLZ,
  IREE_UK_X32U_EXPF,
  IREE_UK_X32U_FLOORF,
  IREE_UK_X32U_LOGF,
  IREE_UK_X32U_NEGF,
  IREE_UK_X32U_RSQRTF,
} iree_uk_x32u_opcode_t;

// Macros to access various typed, dereferenced pointers.
#define ASF32(ptr) *((float*)ptr)
#define ASUI32(ptr) *((iree_uk_uint32_t*)ptr)
#define ASSI32(ptr) *((iree_uk_int32_t*)ptr)

//===----------------------------------------------------------------------===//
// Implementation macros.
//===----------------------------------------------------------------------===//

// Defines a generic "dispatched" implementation via opcode_t by invoking
// the function iree_uk_generic_{category}_2d.
// Corresponds to the header macro DECLARE_UKERNEL_BINARY_2D.
#define DISPATCH_UKERNEL_BINARY_2D(opcode, opcode_t, dtype, category)         \
  IREE_UK_EXPORT int iree_uk_##category##_##opcode##_2d(                      \
      const dtype* lhs, iree_uk_index_t lhs_offset,                           \
      iree_uk_index_t lhs_stride0, iree_uk_index_t lhs_stride1,               \
      const dtype* rhs, iree_uk_index_t rhs_offset,                           \
      iree_uk_index_t rhs_stride0, iree_uk_index_t rhs_stride1,               \
      dtype* IREE_UK_RESTRICT out, iree_uk_index_t out_offset,                \
      iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,               \
      iree_uk_index_t size0, iree_uk_index_t size1) {                         \
    return iree_uk_generic_##category##_2d(                                   \
        opcode_t, lhs, lhs_offset, lhs_stride0, lhs_stride1, rhs, rhs_offset, \
        rhs_stride0, rhs_stride1, out, out_offset, out_stride0, out_stride1,  \
        size0, size1);                                                        \
  }

// Defines a generic "dispatched" implementation via opcode_t by invoking
// the function iree_uk_generic_{category}_2d.
// Corresponds to the header macro DECLARE_UKERNEL_BINARY_2D.
#define DISPATCH_UKERNEL_UNARY_2D(opcode, opcode_t, dtype, category)          \
  IREE_UK_EXPORT int iree_uk_##category##_##opcode##_2d(                      \
      const dtype* in, iree_uk_index_t in_offset, iree_uk_index_t in_stride0, \
      iree_uk_index_t in_stride1, dtype* IREE_UK_RESTRICT out,                \
      iree_uk_index_t out_offset, iree_uk_index_t out_stride0,                \
      iree_uk_index_t out_stride1, iree_uk_index_t size0,                     \
      iree_uk_index_t size1) {                                                \
    return iree_uk_generic_##category##_2d(                                   \
        opcode_t, in, in_offset, in_stride0, in_stride1, out, out_offset,     \
        out_stride0, out_stride1, size0, size1);                              \
  }

//===----------------------------------------------------------------------===//
// Internal helpers.
//===----------------------------------------------------------------------===//

// Computes a single element of an x32b opcode. On error, should set
// |*result_code| to a non-zero value (but should not touch it otherwise).
static void iree_uk_generic_x32b_op(iree_uk_x32b_opcode_t opcode,
                                    int* result_code,
                                    const iree_uk_uint32_t* lhs,
                                    const iree_uk_uint32_t* rhs,
                                    iree_uk_uint32_t* out) {
  switch (opcode) {
    case IREE_UK_X32B_ADDF:
      ASF32(out) = ASF32(lhs) + ASF32(rhs);
      return;
    case IREE_UK_X32B_ADDI:
      ASUI32(out) = ASUI32(lhs) + ASUI32(rhs);
      return;
    case IREE_UK_X32B_ANDI:
      ASUI32(out) = ASUI32(lhs) & ASUI32(rhs);
      return;
    case IREE_UK_X32B_DIVF:
      ASF32(out) = ASF32(lhs) / ASF32(rhs);
      return;
    case IREE_UK_X32B_DIVSI:
      ASSI32(out) = ASSI32(lhs) / ASSI32(rhs);
      return;
    case IREE_UK_X32B_DIVUI:
      ASUI32(out) = ASUI32(lhs) / ASUI32(rhs);
      return;
    case IREE_UK_X32B_MULF:
      ASF32(out) = ASF32(lhs) * ASF32(rhs);
      return;
    case IREE_UK_X32B_MULI:
      ASUI32(out) = ASUI32(lhs) * ASUI32(rhs);
      return;
    case IREE_UK_X32B_ORI:
      ASUI32(out) = ASUI32(lhs) | ASUI32(rhs);
      return;
    case IREE_UK_X32B_SHLI:
      ASUI32(out) = ASUI32(lhs) << ASUI32(rhs);
      return;
    case IREE_UK_X32B_SHRSI:
      ASSI32(out) = ASSI32(lhs) >> ASSI32(rhs);
      return;
    case IREE_UK_X32B_SHRUI:
      ASUI32(out) = ASUI32(lhs) >> ASUI32(rhs);
      return;
    case IREE_UKENREL_X32B_XORI:
      ASUI32(out) = ASUI32(lhs) ^ ASUI32(rhs);
      return;
    case IREE_UK_X32B_SUBF:
      ASF32(out) = ASF32(lhs) - ASF32(rhs);
      return;
    case IREE_UK_X32B_SUBI:
      ASSI32(out) = ASUI32(lhs) - ASUI32(rhs);
      return;
    default:
      *result_code = 1;
  }
}

// Computes a single element of an x32u opcode. On error, should set
// |*result_code| to a non-zero value (but should not touch it otherwise).
static void iree_uk_generic_x32u_op(iree_uk_x32u_opcode_t opcode,
                                    int* result_code,
                                    const iree_uk_uint32_t* in,
                                    iree_uk_uint32_t* out) {
  switch (opcode) {
    case IREE_UK_X32U_ABSF:
      ASF32(out) = fabsf(ASF32(in));
      return;
    case IREE_UK_X32U_CEILF:
      ASF32(out) = ceilf(ASF32(in));
      return;
    case IREE_UK_X32U_CTLZ:
      ASUI32(out) = iree_uk_count_leading_zeros_u32(ASUI32(in));
      return;
    case IREE_UK_X32U_EXPF:
      ASF32(out) = expf(ASF32(in));
      return;
    case IREE_UK_X32U_FLOORF:
      ASF32(out) = floorf(ASF32(in));
      return;
    case IREE_UK_X32U_LOGF:
      ASF32(out) = logf(ASF32(in));
      return;
    case IREE_UK_X32U_NEGF:
      ASF32(out) = -ASF32(in);
      return;
    case IREE_UK_X32U_RSQRTF:
      ASF32(out) = 1.0f / sqrtf(ASF32(in));
      return;
    default:
      *result_code = 1;
  }
}

//===----------------------------------------------------------------------===//
// Opcode dispatch entry points.
//===----------------------------------------------------------------------===//

// Generic 32bit binary kernels.
IREE_UK_ATTRIBUTE_NOINLINE static int iree_uk_generic_x32b_2d(
    iree_uk_x32b_opcode_t opcode,
    // LHS.
    const iree_uk_uint32_t* lhs, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, iree_uk_index_t lhs_stride1,
    // RHS
    const iree_uk_uint32_t* rhs, iree_uk_index_t rhs_offset,
    iree_uk_index_t rhs_stride0, iree_uk_index_t rhs_stride1,
    // OUT.
    iree_uk_uint32_t* IREE_UK_RESTRICT out, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    // Sizes.
    iree_uk_index_t size0, iree_uk_index_t size1) {
  int result_code = 0;
  // TODO: Manually unroll to x4 to trigger vectorization.
  for (iree_uk_index_t i = 0; i < size0; ++i) {
    for (iree_uk_index_t j = 0; j < size1; ++j) {
      iree_uk_generic_x32b_op(opcode, &result_code,
                              &lhs[i * lhs_stride0 + j * lhs_stride1],
                              &rhs[i * rhs_stride0 + j * rhs_stride1],
                              &out[i * out_stride0 + j * out_stride1]);
    }
  }
  return result_code;
}

// Generic 32bit unary kernels.
IREE_UK_ATTRIBUTE_NOINLINE static int iree_uk_generic_x32u_2d(
    iree_uk_x32u_opcode_t opcode,
    // IN.
    const iree_uk_uint32_t* in, iree_uk_index_t in_offset,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1,
    // OUT.
    iree_uk_uint32_t* IREE_UK_RESTRICT out, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    // Sizes.
    iree_uk_index_t size0, iree_uk_index_t size1) {
  int result_code = 0;
  // TODO: Manually unroll to x4 to trigger vectorization.
  for (iree_uk_index_t i = 0; i < size0; ++i) {
    for (iree_uk_index_t j = 0; j < size1; ++j) {
      iree_uk_generic_x32u_op(opcode, &result_code,
                              &in[i * in_stride0 + j * in_stride1],
                              &out[i * out_stride0 + j * out_stride1]);
    }
  }
  return result_code;
}

DISPATCH_UKERNEL_BINARY_2D(addf, IREE_UK_X32B_ADDF, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(addi, IREE_UK_X32B_ADDI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(andi, IREE_UK_X32B_ANDI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(divf, IREE_UK_X32B_DIVF, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(divsi, IREE_UK_X32B_DIVSI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(divui, IREE_UK_X32B_DIVUI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(mulf, IREE_UK_X32B_MULF, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(muli, IREE_UK_X32B_MULI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(ori, IREE_UK_X32B_ORI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(shli, IREE_UK_X32B_SHLI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(shrsi, IREE_UK_X32B_SHRSI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(shrui, IREE_UK_X32B_SHRUI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(subf, IREE_UK_X32B_SUBF, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(subi, IREE_UK_X32B_SUBI, iree_uk_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(xori, IREE_UKENREL_X32B_XORI, iree_uk_uint32_t,
                           x32b);

DISPATCH_UKERNEL_UNARY_2D(absf, IREE_UK_X32U_ABSF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(ceilf, IREE_UK_X32U_CEILF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(ctlz, IREE_UK_X32U_CTLZ, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(expf, IREE_UK_X32U_EXPF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(floorf, IREE_UK_X32U_FLOORF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(logf, IREE_UK_X32U_LOGF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(negf, IREE_UK_X32U_NEGF, iree_uk_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(rsqrtf, IREE_UK_X32U_RSQRTF, iree_uk_uint32_t, x32u);
