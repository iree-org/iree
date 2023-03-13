// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_
#define IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_

#include "iree/builtins/ukernel/static_assert.h"

// This header is shared across:
//
// * C++ code under compiler/
// * C code   under runtime/.../ukernel/
// * asm code under runtime/.../ukernel/
//
// Being shared with compiler/ means that we should treat these flags as set
// in stone. Don't count on being able to remove or change the numerical value
// of an existing flag.
//
// Being shared with asm code means that the only thing that we can do here is
// #define literal integers. The C/C++ code only cares about the flags values
// but asm code also cares about the bit-position values (i.e. the log2's).
// Consistency between the two is guarded by static_assert's but only when
// the language is C/C++ (not assembly).

// ukernel flags are typically of type uint32.
// Some flags will be common to many ukernels, while others will be specific
// to one or a few ukernels. To make our 32 bits last as long as possible,
// we need to make some educated guess as to how to split them between common
// and ukernel-specific bits. Let's split them in half:
// * The low 16 bits (bits 0..15) are for common bits, expected to be shared
//   among many ukernels.
// * The high 16 bits (bits 16..31) are for ukernel-specific bits.

// Common bits (bits 0..15).
#define IREE_UK_FLAG_ACCUMULATE 0x1u
#define IREE_UK_FLAG_ACCUMULATE_BIT_POS 0

// `pack` ukernel-specific bits (bits 16..31).
#define IREE_UK_FLAG_PACK_TRANSPOSE_INNER 0x10000u
#define IREE_UK_FLAG_PACK_TRANSPOSE_INNER_BIT_POS 16
#define IREE_UK_FLAG_PACK_TRANSPOSE_OUTER 0x20000u
#define IREE_UK_FLAG_PACK_TRANSPOSE_OUTER_BIT_POS 17

// `unpack` ukernel-specific bits (bits 16..31).
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER 0x10000u
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER_BIT_POS 16
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER 0x20000u
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER_BIT_POS 17

// Static assertions ensuring consistency of the above flag values.
#define IREE_UK_ENSURE_CONSISTENT_FLAG(F) \
  IREE_UK_STATIC_ASSERT((F) == (1u << (F##_BIT_POS)))
IREE_UK_ENSURE_CONSISTENT_FLAG(IREE_UK_FLAG_ACCUMULATE);
IREE_UK_ENSURE_CONSISTENT_FLAG(IREE_UK_FLAG_PACK_TRANSPOSE_INNER);
IREE_UK_ENSURE_CONSISTENT_FLAG(IREE_UK_FLAG_PACK_TRANSPOSE_OUTER);

// `query_tile_sizes` ukernel-specific bits (bits 16..31).
// OPERAND_ROLE (bits 16..19) describes the role that a tensor plays in an
// operation, e.g. "left-hand-size operand" (e.g. in a matmul)
// Note: the _INTERNAL suffix conveys that the _MASK value should only be used
// by microkernels decoding flags, not by the compiler setting flags. Masks may
// have to change even if flag values don't.
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_MASK_INTERNAL 0xf0000u
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS 0x00000u
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS 0x10000u
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT 0x20000u
// OPERATION (bits 20..31, though may be shrunk as needed as this is currently
// only using bit 20 and will only grow as needed) describes the operation
// owning the tensor (that we are doing a query_tile_sizes for) as an operand.
// Note: the _INTERNAL suffix conveys that the _MASK value should only be used
// by microkernels decoding flags, not by the compiler setting flags. Masks may
// have to change even if flag values don't.
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MASK_INTERNAL 0xfff00000u
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32 0x000000u
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32 0x100000u

#endif  // IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_
