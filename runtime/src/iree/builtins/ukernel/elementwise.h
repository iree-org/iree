// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ELEMENTWISE_H_
#define IREE_BUILTINS_UKERNEL_ELEMENTWISE_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Public API - Binary kernels.
//===----------------------------------------------------------------------===//

// Binary ukernel func 2d, x32.
// It takes lhs, rhs, out buffers and size, returning 0 on success and !0 on
// error.
typedef int (*iree_uk_x32b_2d_func_t)(
    const iree_uk_uint32_t* lhs, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, iree_uk_index_t lhs_stride1,
    const iree_uk_uint32_t* rhs, iree_uk_index_t rhs_offset,
    iree_uk_index_t rhs_stride0, iree_uk_index_t rhs_stride1,
    iree_uk_uint32_t* out, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t size0, iree_uk_index_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_uk_{category}_{opcode}_2d(...)
// of function type iree_uk_{category}_2d_func_t.
#define DECLARE_UKERNEL_BINARY_2D(opcode, dtype, category)      \
  IREE_UK_EXPORT int iree_uk_##category##_##opcode##_2d(        \
      const dtype* lhs, iree_uk_index_t lhs_offset,             \
      iree_uk_index_t lhs_stride0, iree_uk_index_t lhs_stride1, \
      const dtype* rhs, iree_uk_index_t rhs_offset,             \
      iree_uk_index_t rhs_stride0, iree_uk_index_t rhs_stride1, \
      dtype* IREE_UK_RESTRICT out, iree_uk_index_t out_offset,  \
      iree_uk_index_t out_stride0, iree_uk_index_t out_stride1, \
      iree_uk_index_t size0, iree_uk_index_t size1)

DECLARE_UKERNEL_BINARY_2D(addf, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(addi, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(andi, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divf, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divsi, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divui, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(mulf, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(muli, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(ori, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shli, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrsi, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrui, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subf, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subi, iree_uk_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(xori, iree_uk_uint32_t, x32b);

//===----------------------------------------------------------------------===//
// Public API - Unary kernels.
//===----------------------------------------------------------------------===//

// Unary ukernel func 2d, x32.
// It takes in, out buffers and size, returning 0 on success and !0 on
// error.
typedef int (*iree_uk_x32u_2d_func_t)(
    const iree_uk_uint32_t* in, iree_uk_index_t in_offset,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1,
    iree_uk_uint32_t* out, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t size0, iree_uk_index_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_uk_{category}_{opcode}_2d(...)
// It takes lhs, rhs, out buffers and size, returning 0 on success and !0 on
// error.
#define DECLARE_UKERNEL_UNARY_2D(opcode, dtype, category)                     \
  IREE_UK_EXPORT int iree_uk_##category##_##opcode##_2d(                      \
      const dtype* in, iree_uk_index_t in_offset, iree_uk_index_t in_stride0, \
      iree_uk_index_t in_stride1, dtype* IREE_UK_RESTRICT out,                \
      iree_uk_index_t out_offset, iree_uk_index_t out_stride0,                \
      iree_uk_index_t out_stride1, iree_uk_index_t size0,                     \
      iree_uk_index_t size1)

DECLARE_UKERNEL_UNARY_2D(absf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ceilf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ctlz, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(expf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(floorf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(logf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(negf, iree_uk_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(rsqrtf, iree_uk_uint32_t, x32u);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_ELEMENTWISE_H_
