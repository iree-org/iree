// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_
#define IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_

// This header is shared across:
//
// * C++ code under compiler/
// * C code   under runtime/.../ukernel/
//
// Being shared with compiler/ means that we should treat these flags as set
// in stone. Don't count on being able to remove or change the numerical value
// of an existing flag.
//
// Ukernel flags are typically of type uint32. For now, we treat all flags as
// specific to one op, sometimes duplicating identical flags for multiple ops.
// In the future we might let multiple ops share common flags, but it is too
// early to tell which yet. We also might widen flags to uint64 as needed.

//===----------------------------------------------------------------------===//
// mmt4d
//===----------------------------------------------------------------------===//

// type enum
#define IREE_UK_FLAG_MMT4D_TYPE_MASK 0xFF
#define IREE_UK_FLAG_MMT4D_TYPE_NONE 0x00
#define IREE_UK_FLAG_MMT4D_TYPE_F32F32F32 0x01
#define IREE_UK_FLAG_MMT4D_TYPE_S8S8S32 0x02
#define IREE_UK_FLAG_MMT4D_TYPE_F16F16F32 0x03
#define IREE_UK_FLAG_MMT4D_TYPE_F16F16F16 0x04
#define IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32 0x05
#define IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16 0x06
#define IREE_UK_FLAG_MMT4D_TYPE_S16S16S32 0x07
#define IREE_UK_FLAG_MMT4D_TYPE_S16U4S32 0x08
#define IREE_UK_FLAG_MMT4D_TYPE_S16S8S32 0x09
#define IREE_UK_FLAG_MMT4D_TYPE_END 0x0A

// bit flags
#define IREE_UK_FLAG_MMT4D_ACCUMULATE 0x100
#define IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS 0x400

//===----------------------------------------------------------------------===//
// pack
//===----------------------------------------------------------------------===//

// type enum
#define IREE_UK_FLAG_PACK_TYPE_MASK 0xFF
#define IREE_UK_FLAG_PACK_TYPE_NONE 0x00
#define IREE_UK_FLAG_PACK_TYPE_F32F32 0x01
#define IREE_UK_FLAG_PACK_TYPE_I8I8 0x02
#define IREE_UK_FLAG_PACK_TYPE_I32I32 0x03
#define IREE_UK_FLAG_PACK_TYPE_F16F16 0x04
#define IREE_UK_FLAG_PACK_TYPE_BF16BF16 0x05

// bit flags
#define IREE_UK_FLAG_PACK_TRANSPOSE_INNER 0x100
#define IREE_UK_FLAG_PACK_TRANSPOSE_OUTER 0x200

//===----------------------------------------------------------------------===//
// unpack
//===----------------------------------------------------------------------===//

// type enum
#define IREE_UK_FLAG_UNPACK_TYPE_MASK 0xFF
#define IREE_UK_FLAG_UNPACK_TYPE_NONE 0x00
#define IREE_UK_FLAG_UNPACK_TYPE_F32F32 0x01
/* 0x02 reserved for I8I8 as in _PACK_? */
#define IREE_UK_FLAG_UNPACK_TYPE_I32I32 0x03
#define IREE_UK_FLAG_UNPACK_TYPE_F16F16 0x04
#define IREE_UK_FLAG_UNPACK_TYPE_BF16BF16 0x05

// bit flags
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER 0x100
#define IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER 0x200

//===----------------------------------------------------------------------===//
// query_tile_sizes
//===----------------------------------------------------------------------===//

// OPERAND_ROLE describes the role that a tensor plays in an
// operation, e.g. "left-hand-size operand" (e.g. in a matmul).
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_MASK 0xFF
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_NONE 0x00
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS 0x01
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS 0x02
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT 0x03

// OPERATION describes the operation the the operand (that we are doing a
// query_tile_sizes for) belongs to.
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MASK 0xFF00
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_NONE 0x0000
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32 0x0100
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32 0x0200
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F32 0x0300
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F16 0x0400
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16F32 0x0500
#define IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16BF16 0x0600

#endif  // IREE_BUILTINS_UKERNEL_EXPORTED_BITS_H_
