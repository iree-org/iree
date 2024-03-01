// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_PACK_H_
#define IREE_BUILTINS_UKERNEL_PACK_H_

#include "iree/builtins/ukernel/common.h"

// `pack` microkernel. Currently only used in the VMVX backend, not used in the
// LLVMCPU backend, because codegen is thought to be good enough and because
// fusions really matter here.

// The least significant bits of `padding_value`, up to element size, are used
// for padding. As this is based solely on bit-significance and not on byte
// addresses, this is independent of endianness.
//
// If the element size is less than 64 bits then the most significant bits
// (above element size) are unused.
//
// If the element size is more than 64 bits then only repeating 64-bit
// patterns are supported for padding. This covers most cases as floating
// point types encode zero as zero bits.
IREE_UK_EXPORT void iree_uk_pack(
    const void* in_buffer, iree_uk_index_t in_offset,
    iree_uk_index_t in_stride0, void* out_buffer, iree_uk_index_t out_offset,
    iree_uk_index_t out_stride0, iree_uk_index_t in_size0,
    iree_uk_index_t in_size1, iree_uk_index_t out_size0,
    iree_uk_index_t out_size1, iree_uk_index_t out_size2,
    iree_uk_index_t out_size3, iree_uk_uint64_t padding_value,
    iree_uk_uint32_t flags, const iree_uk_uint64_t* cpu_data);

#endif  // IREE_BUILTINS_UKERNEL_PACK_H_
