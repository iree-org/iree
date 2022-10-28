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
    const iree_ukernel_uint32_t* lhs, iree_ukernel_ssize_t lhs_offset,
    iree_ukernel_ssize_t lhs_stride0, iree_ukernel_ssize_t lhs_stride1,
    const iree_ukernel_uint32_t* rhs, iree_ukernel_ssize_t rhs_offset,
    iree_ukernel_ssize_t rhs_stride0, iree_ukernel_ssize_t rhs_stride1,
    iree_ukernel_uint32_t* out, iree_ukernel_ssize_t out_offset,
    iree_ukernel_ssize_t out_stride0, iree_ukernel_ssize_t out_stride1,
    iree_ukernel_ssize_t size0, iree_ukernel_ssize_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_ukernel_{category}_{opcode}_2d(...)
// of function type iree_ukernel_{category}_2d_func_t.
#define DECLARE_UKERNEL_BINARY_2D(opcode, dtype, category)                \
  IREE_UKERNEL_EXPORT int iree_ukernel_##category##_##opcode##_2d(        \
      const dtype* lhs, iree_ukernel_ssize_t lhs_offset,                  \
      iree_ukernel_ssize_t lhs_stride0, iree_ukernel_ssize_t lhs_stride1, \
      const dtype* rhs, iree_ukernel_ssize_t rhs_offset,                  \
      iree_ukernel_ssize_t rhs_stride0, iree_ukernel_ssize_t rhs_stride1, \
      dtype* IREE_UKERNEL_RESTRICT out, iree_ukernel_ssize_t out_offset,  \
      iree_ukernel_ssize_t out_stride0, iree_ukernel_ssize_t out_stride1, \
      iree_ukernel_ssize_t size0, iree_ukernel_ssize_t size1)

DECLARE_UKERNEL_BINARY_2D(addf, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(addi, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(andi, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divf, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divsi, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(divui, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(mulf, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(muli, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(ori, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shli, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrsi, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(shrui, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subf, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(subi, iree_ukernel_uint32_t, x32b);
DECLARE_UKERNEL_BINARY_2D(xori, iree_ukernel_uint32_t, x32b);

//===----------------------------------------------------------------------===//
// Public API - Unary kernels.
//===----------------------------------------------------------------------===//

// Unary ukernel func 2d, x32.
// It takes in, out buffers and size, returning 0 on success and !0 on
// error.
typedef int (*iree_ukernel_x32u_2d_func_t)(
    const iree_ukernel_uint32_t* in, iree_ukernel_ssize_t in_offset,
    iree_ukernel_ssize_t in_stride0, iree_ukernel_ssize_t in_stride1,
    iree_ukernel_uint32_t* out, iree_ukernel_ssize_t out_offset,
    iree_ukernel_ssize_t out_stride0, iree_ukernel_ssize_t out_stride1,
    iree_ukernel_ssize_t size0, iree_ukernel_ssize_t size1);

// Declares a binary 2d microkernel with the following signature:
//   int iree_ukernel_{category}_{opcode}_2d(...)
// It takes lhs, rhs, out buffers and size, returning 0 on success and !0 on
// error.
#define DECLARE_UKERNEL_UNARY_2D(opcode, dtype, category)                 \
  IREE_UKERNEL_EXPORT int iree_ukernel_##category##_##opcode##_2d(        \
      const dtype* in, iree_ukernel_ssize_t in_offset,                    \
      iree_ukernel_ssize_t in_stride0, iree_ukernel_ssize_t in_stride1,   \
      dtype* IREE_UKERNEL_RESTRICT out, iree_ukernel_ssize_t out_offset,  \
      iree_ukernel_ssize_t out_stride0, iree_ukernel_ssize_t out_stride1, \
      iree_ukernel_ssize_t size0, iree_ukernel_ssize_t size1)

DECLARE_UKERNEL_UNARY_2D(absf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ceilf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(ctlz, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(expf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(floorf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(logf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(negf, iree_ukernel_uint32_t, x32u);
DECLARE_UKERNEL_UNARY_2D(rsqrtf, iree_ukernel_uint32_t, x32u);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_ELEMENTWISE_H_
