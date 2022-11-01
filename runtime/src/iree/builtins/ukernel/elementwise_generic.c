// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "elementwise.h"

// Include the generic implementation helpers.
#include "elementwise_impl.c.inc"

DISPATCH_UKERNEL_BINARY_2D(addf, IREE_UKERNEL_X32B_ADDF, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(addi, IREE_UKERNEL_X32B_ADDI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(andi, IREE_UKERNEL_X32B_ANDI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(divf, IREE_UKERNEL_X32B_DIVF, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(divsi, IREE_UKERNEL_X32B_DIVSI,
                           iree_ukernel_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(divui, IREE_UKERNEL_X32B_DIVUI,
                           iree_ukernel_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(mulf, IREE_UKERNEL_X32B_MULF, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(muli, IREE_UKERNEL_X32B_MULI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(ori, IREE_UKERNEL_X32B_ORI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(shli, IREE_UKERNEL_X32B_SHLI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(shrsi, IREE_UKERNEL_X32B_SHRSI,
                           iree_ukernel_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(shrui, IREE_UKERNEL_X32B_SHRUI,
                           iree_ukernel_uint32_t, x32b);
DISPATCH_UKERNEL_BINARY_2D(subf, IREE_UKERNEL_X32B_SUBF, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(subi, IREE_UKERNEL_X32B_SUBI, iree_ukernel_uint32_t,
                           x32b);
DISPATCH_UKERNEL_BINARY_2D(xori, IREE_UKENREL_X32B_XORI, iree_ukernel_uint32_t,
                           x32b);

DISPATCH_UKERNEL_UNARY_2D(absf, IREE_UKERNEL_X32U_ABSF, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(ceilf, IREE_UKERNEL_X32U_CEILF, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(ctlz, IREE_UKERNEL_X32U_CTLZ, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(expf, IREE_UKERNEL_X32U_EXPF, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(floorf, IREE_UKERNEL_X32U_FLOORF,
                          iree_ukernel_uint32_t, x32u);
DISPATCH_UKERNEL_UNARY_2D(logf, IREE_UKERNEL_X32U_LOGF, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(negf, IREE_UKERNEL_X32U_NEGF, iree_ukernel_uint32_t,
                          x32u);
DISPATCH_UKERNEL_UNARY_2D(rsqrtf, IREE_UKERNEL_X32U_RSQRTF,
                          iree_ukernel_uint32_t, x32u);
