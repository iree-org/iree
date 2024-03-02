// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_H_

#include "iree/builtins/ukernel/common.h"

// `mmt4d` microkernel. Used on LLVMCPU (as well as VMVX), due to difficulty of
// code generation of matrix multiplications kernels.
IREE_UK_EXPORT void iree_uk_mmt4d(
    const void* lhs_buffer, iree_uk_index_t lhs_offset,
    iree_uk_index_t lhs_stride0, const void* rhs_buffer,
    iree_uk_index_t rhs_offset, iree_uk_index_t rhs_stride0, void* out_buffer,
    iree_uk_index_t out_offset, iree_uk_index_t out_stride0, iree_uk_index_t M,
    iree_uk_index_t N, iree_uk_index_t K, iree_uk_int32_t M0,
    iree_uk_int32_t N0, iree_uk_int32_t K0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data);

// Returns a bit-field of information about how a mmt4d with the given
// parameters would run.
IREE_UK_EXPORT iree_uk_uint32_t
iree_uk_mmt4d_info(iree_uk_int32_t M0, iree_uk_int32_t N0, iree_uk_int32_t K0,
                   iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data);

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
