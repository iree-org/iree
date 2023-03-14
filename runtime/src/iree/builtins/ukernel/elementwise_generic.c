// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "elementwise.h"

// Include the generic implementation helpers.
#include "elementwise_impl.c.inc"

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
