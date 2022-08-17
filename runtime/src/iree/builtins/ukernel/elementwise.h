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
typedef int (*iree_ukernel_x32b_2d_func_t)(
    const uint32_t* lhs, iree_ukernel_size_t lhs_offset,
    iree_ukernel_size_t lhs_stride0, iree_ukernel_size_t lhs_stride1,
    const uint32_t* rhs, iree_ukernel_size_t rhs_offset,
    iree_ukernel_size_t rhs_stride0, iree_ukernel_size_t rhs_stride1,
    uint32_t* out, iree_ukernel_size_t out_offset,
    iree_ukernel_size_t out_stride0, iree_ukernel_size_t out_stride1,
    iree_ukernel_size_t size0, iree_ukernel_size_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_ukernel_{category}_{opcode}_2d(...)
// of function type iree_ukernel_{category}_2d_func_t.
#define DECLARE_UKERNEL_BINARY_2D(opcode, dtype, category)              \
  IREE_UKERNEL_EXPORT int iree_ukernel_##category##_##opcode##_2d(      \
      const dtype* lhs, iree_ukernel_size_t lhs_offset,                 \
      iree_ukernel_size_t lhs_stride0, iree_ukernel_size_t lhs_stride1, \
      const dtype* rhs, iree_ukernel_size_t rhs_offset,                 \
      iree_ukernel_size_t rhs_stride0, iree_ukernel_size_t rhs_stride1, \
      dtype* IREE_RESTRICT out, iree_ukernel_size_t out_offset,         \
      iree_ukernel_size_t out_stride0, iree_ukernel_size_t out_stride1, \
      iree_ukernel_size_t size0, iree_ukernel_size_t size1)

DECLARE_UKERNEL_BINARY_2D(addf, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(addi, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(andi, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divf, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divsi, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divui, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(mulf, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(muli, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(ori, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shli, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrsi, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrui, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subf, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subi, uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(xori, uint32_t, x32b);

//===----------------------------------------------------------------------===//
// Public API - Unary kernels.
//===----------------------------------------------------------------------===//

// Unary ukernel func 2d, x32.
// It takes in, out buffers and size, returning 0 on success and !0 on
// error.
typedef int (*iree_ukernel_x32u_2d_func_t)(
    const uint32_t* in, iree_ukernel_size_t in_offset,
    iree_ukernel_size_t in_stride0, iree_ukernel_size_t in_stride1,
    uint32_t* out, iree_ukernel_size_t out_offset,
    iree_ukernel_size_t out_stride0, iree_ukernel_size_t out_stride1,
    iree_ukernel_size_t size0, iree_ukernel_size_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_ukernel_{category}_{opcode}_2d(...)
// It takes lhs, rhs, out buffers and size, returning 0 on success and !0 on
// error.
#define DECLARE_UKERNEL_UNARY_2D(opcode, dtype, category)               \
  IREE_UKERNEL_EXPORT int iree_ukernel_##category##_##opcode##_2d(      \
      const dtype* in, iree_ukernel_size_t in_offset,                   \
      iree_ukernel_size_t in_stride0, iree_ukernel_size_t in_stride1,   \
      dtype* IREE_RESTRICT out, iree_ukernel_size_t out_offset,         \
      iree_ukernel_size_t out_stride0, iree_ukernel_size_t out_stride1, \
      iree_ukernel_size_t size0, iree_ukernel_size_t size1)

DECLARE_UKERNEL_UNARY_2D(absf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ceilf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ctlz, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(expf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(floorf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(logf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(negf, uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(rsqrtf, uint32_t, x32u);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_ELEMENTWISE_H_
