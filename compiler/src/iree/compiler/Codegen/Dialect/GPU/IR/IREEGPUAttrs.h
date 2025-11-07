// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::IREE::GPU {

//////////////////////////////////////////////////////////////////////////////
// MMASingleSubgroupLayout
//////////////////////////////////////////////////////////////////////////////
//
// Overview, terminology
// ---------------------
//
// MMASingleSubgroupLayout describes the layout of one operand of one subgroup
// level operation such as an MMA intrinsic.
//
// An MMA intrinsic, running on one thread, takes a 1D vector for each operand.
// The purpose of MMASingleSubgroupLayout is to describe the mapping
// from thread-id and 1D-vector-index into "semantical" dimensions (see next
// paragraph). For example, for the accumulator ("C") operand of
// @llvm.amdgcn.mfma.i32.32x32x8i8, which has type <16 x i32>, we want to know
// for each of those 16 elements where it belongs in the MxN tile, in terms of
// those "semantical" dimensions M and N, as a function of (1) the index of that
// element within the <16 x i32> vector and (2) the thread-id.
//
// Here, by "semantical dimensions" we mean the high-level dimensions like the
// M, N, K of matrix multiplication, or the M, N, K, Kb of scaled-matmul, or
// the B, M, N, K of batch matmul. Some of these dimensions occur in the operand
// that we are concerned with, some don't. For example, a matmul Lhs operand
// only has the M and K semantical dimensions. A scaled-matmul Lhs has the M, K,
// Kb semantical dimensions.
//
// Let us call "semantical rank" the number of semantical dimensions occuring
// in the operand that we are concerned with. That is often 2, but can be 3
// for some scaled-matmul operands and for all batch-matmul operands. This could
// also be 1 for vector operands of matrix-vector operations.
//
// General invariants
// ------------------
//
// Before we enter the more detailed description below, let us already state a
// few high-level invariants.
//
// 0. All the member arrays have length equal to the semantical rank. A common
//    enumeration order of semantical dimensions is used throughout.
//    Each array entry corresponds to one semantical dimension. For example, for
//    a MFMA Lhs operand, the semantical dims are enumerated as M, K. Thus the
//    array elements outer[0], thread[0], element[0] correspond to the M
//    dimension, and the [1] correspond to the K dimension.
// 1. For each semantical dimension, the product (outer[i] * thread[i] *
//    element[i]) equals the semantical dimension size, i.e., the tile size. For
//    example, in @llvm.amdgcn.mfma.i32.32x32x8i8, for the M and N dimensions,
//    these products equal 32.
// 2. The product of all the outer[i] times all the element[i] equals the
//    length of the vector operand to the intrinsic. It is the number of
//    elements that one intrinsic consumes on one thread.
// 3. The product of all the thread[i] is a divisor of subgroup size. It is
//    almost always equal to subgroup size. If not, then it is a strict divisor
//    of subgroup size and that means that multiple threads get the exact same
//    data, i.e., there is an implied broadcasting, as will be seen in the
//    modulo (t % thread [0]) below.
//
// Detailed semantics: case of semantic rank 1
// -------------------------------------------
//
// When the semantic rank is 1, meaning that there is only 1 semantic dimension
// (this would happen for a vector operand in a matrix-vector multiplication),
// say "M", the mapping is as follows:
//
// /* Here t == thread_id, i == vector_element_index */
// int map_vector_elem_index_to_semantic_dim_index(int t, int i) {
//   return (i % element[0]) + element[0] * (
//            (t % thread[0]) + thread[0] * (
//              i / element[0]
//            )
//          );
// }
//
// Notice that we didn't use outer[0]. It is a redundant parameter in this case
// since element[0] * outer[0] has to be the vector length as noted in the above
// "invariants".
//
// Also notice that in the (rare) case where thread [0] is smaller than subgroup
// size, multiple threads will get the same value of (t % thread [0]) and thus
// will get the same data.
//
// Detailed semantics: general case
// -------------------------------------------
//
// The general procedure is a generalization of the above rank-1 case:
// 1. Delinearize the vector index modulo (element[0] * ... * element[rank - 1])
//    into the grid of shape {element[0], ..., element[rank - 1]}.
//    Thus, the element[] array describes the tile that is stored in contiguous
//    elements of the intrinsics' vector operand. We will call it the
//    "element tile". The layout within the element tile is always "row-major"
//    in the sense that the last-enumerated semantic dimension is the
//    most-contiguous dimension.
// 2. Delinearize the thread-id modulo (thread[0] * ... * thread[rank - 1])
//    into the grid of shape {thread[0], ..., thread[rank - 1]}.
//    Thus, the different threads get different element tiles (from step 1)
//    except in the rare case that (thread[0] * ... * thread[rank - 1]) is less
//    than subgroup size, in which case the threads wrap around and share tiles.
//    * Unlike the element tiles from step 1, the distribution of these tiles to
//      threads is not necessarily following row-major order. The thread layout
//      is described by the tstrides. The meaning of tstrides[i] is: "as we move
//      by one element tile along semantic dimension i, we add tstrides[i]
//      to the thread_id". Note that in the rare case that multiple threads see
//      the same element tile, we can have tstrides[i] == 0.
// 3. Delinearize the "outer vector index", defined as the quotient
//    vector_index / (element[0] * ... * element[rank - 1]),
//    into the grid of shape {outer[0], ..., outer[rank - 1]}.
//    Just like the element-tiles, the layout of these "outer tiles" is
//    row-major in the sense that the last-enumerated semantic dimension is the
//    most-contiguous dimension. The outer dimensions describe the arrangement
//    of element tiles in the overall vector operand of the intrinsic. This is
//    used only by a minority of intrinsics to describe "non-contiguous" operand
//    tiles.
//
// Example: @llvm.amdgcn.mfma.i32.32x32x8i8 accumulator
// ----------------------------------------
//
// For the accumulator operand of @llvm.amdgcn.mfma.i32.32x32x8i8, we have:
//
//   outer    = {4,  1}
//   thread   = {2,  32}
//   tstrides = {32, 1}
//   element  = {4,  1}
//
// Let us first check the general invariants:
// 0. The arrays all have length 2, corresponding to the semantic rank 2 of
//    matmul accumulators, where the 2 semantic dimensions are M and N.
// 1. The semantic tile size is (M = 32, N = 32) and that does match the product
//    of the corresponding array elements outer[i] * thread[i] * element[i],
//    for instance 4 * 2 * 4 == 32.
// 2. The product of all the outer[i] times all the element[i] is:
//    4 * 1 * 4 * 1 == 16.
//    This corresponds to the @llvm.amdgcn.mfma.i32.32x32x8i8 intrinsic's vector
//    operand type, <16 x i32>.
// 3. The product of the thread[i] is 2 * 32 == 64. This corresponds to the
//    subgroup size 64 on CDNA3. It could also have been a divisor of that, but
//    here the exact match means that each thread receives a different tile.
//
// The tstrides here are such that the distribution of element-tiles to threads
// is "row-major": the fact that tstrides[1] == 1 means that as we move by
// one element tile down the N-dimension, we move to the next thread by tid.
//
// Let us now detail the exact layout:
// 1. Always start by looking at element[]. Here we see that the element tile
//    has shape 4x1.
// 2. The thread-grid has shape 2x32, and as noted above has row-major thread
//    distribution. As what is being distributed to threads here is element
//    tiles of shape 4x1, at this point we have distributed an overall 8x32
//    tile.
// 3. Finally, the outer[] completes the picture by saying that those 8x32
//    tiles are stacked vertically to form the overall 32x32 tile. The fact that
//    the vector length 16 is split into outer[0]==4 and element[0]==4 means
//    that these vectors contain groups of 4 matrix elements that are contiguous
//    along the M-dimension (the "element tiles"), but that there is a
//    discontinuity as the next 4 elements (the next "element tile") comes from
//    far away elsewhere in the C matrix, owing to the fact that thread[0] is
//    greater than 1. Example: thread 0 gets this accumulator ("C") operand:
//    { C[0,  0],  C[1,  0],  C[2,  0],  C[3,  0],
//      C[8,  0],  C[9,  0],  C[10, 0],  C[11, 0],
//      C[16, 0],  C[17, 0],  C[18, 0],  C[19, 0],
//      C[24, 0],  C[25, 0],  C[26, 0],  C[27, 0] }
//
struct MMASingleSubgroupLayout {
  // Internal dimensions (as in TileSwizzle::Dim::Kind::Internal) that are
  // outer-most in the layout. This happens when an MMA op, seen on a single
  // thread, has an operand that consists of multiple elements, and these elems
  // are NOT contiguous.
  // This is not used by every MMA op; ops which don't use that simply have 1's.
  SmallVector<int64_t, 2> outer;
  // Cross-thread dimensions (as in TileSwizzle::Dim::Kind::CrossThread).
  // This is the kind of dimension that is present in all GPU MMA ops, by
  // definition of "SIMT". It is still possible for one of the `thread` dims to
  // be 1, but not both.
  SmallVector<int64_t, 2> thread;
  // Strides corresponding to the cross-thread dimensions.
  SmallVector<int64_t, 2> tstrides;
  // Internal dimensions (as in TileSwizzle::Dim::Kind::Internal) that are
  // inner-most in the layout. This happens when an MMA op, seen on a single
  // thread, has an operand that consists of multiple elements, and these elems
  // are contiguous.
  // This is not used by every MMA op; ops which don't use that simply have 1's.
  SmallVector<int64_t, 2> element;
};

/// Helpers to return the M, N, K, and Kb sizes for the given scaled MMA
/// intrinsic. These sizes correspond to computation performed by a single
/// subgroup.
int64_t getMSize(ScaledMMAIntrinsic intrinsic);
int64_t getNSize(ScaledMMAIntrinsic intrinsic);
int64_t getKSize(ScaledMMAIntrinsic intrinsic);
int64_t getKbSize(ScaledMMAIntrinsic intrinsic);

/// Returns the subgroup size used by the given scaled MMA intrinsic.
int64_t getIntrinsicSubgroupSize(ScaledMMAIntrinsic intrinsic);

/// Helpers to return the M, N, and K sizes for the given MMA intrinsic.
/// These sizes correspond to computation performed by a single subgroup.
int64_t getMSize(MMAIntrinsic intrinsic);
int64_t getNSize(MMAIntrinsic intrinsic);
int64_t getKSize(MMAIntrinsic intrinsic);

/// Returns the subgroup size used by the given MMA intrinsic.
int64_t getIntrinsicSubgroupSize(MMAIntrinsic intrinsic);

constexpr int kMMAOperandLhs = 0;
constexpr int kMMAOperandRhs = 1;
constexpr int kMMAOperandAcc = 2;
constexpr int kScaledMMAOperandLhs = 0;
constexpr int kScaledMMAOperandRhs = 1;
constexpr int kScaledMMAOperandLhsScale = 2;
constexpr int kScaledMMAOperandRhsScale = 3;
constexpr int kScaledMMAOperandAcc = 4;

template <typename MMAIntrinsicType>
int isIntrinsicLhs(int operandIndex) {
  return operandIndex == (std::is_same_v<MMAIntrinsicType, ScaledMMAIntrinsic>
                              ? kScaledMMAOperandLhs
                              : kMMAOperandLhs);
}
template <typename MMAIntrinsicType>
int isIntrinsicRhs(int operandIndex) {
  return operandIndex == (std::is_same_v<MMAIntrinsicType, ScaledMMAIntrinsic>
                              ? kScaledMMAOperandRhs
                              : kMMAOperandRhs);
}
template <typename MMAIntrinsicType>
int isIntrinsicAcc(int operandIndex) {
  return operandIndex == (std::is_same_v<MMAIntrinsicType, ScaledMMAIntrinsic>
                              ? kScaledMMAOperandAcc
                              : kMMAOperandAcc);
}

template <typename MMAIntrinsicType>
int isIntrinsicLhsScale(int operandIndex) {
  return std::is_same_v<MMAIntrinsicType, ScaledMMAIntrinsic>
             ? (operandIndex == kScaledMMAOperandLhsScale)
             : false;
}
template <typename MMAIntrinsicType>
int isIntrinsicRhsScale(int operandIndex) {
  return std::is_same_v<MMAIntrinsicType, ScaledMMAIntrinsic>
             ? (operandIndex == kScaledMMAOperandRhsScale)
             : false;
}

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                int operandIndex);

MMASingleSubgroupLayout getSingleSubgroupLayout(MMAIntrinsic intrinsic,
                                                int operandIndex,
                                                bool colMajor);

MMASingleSubgroupLayout
getSingleSubgroupLayout(VirtualMMAIntrinsic virtualIntrinsic, int operandIndex,
                        bool colMajor);

MMASingleSubgroupLayout getSingleSubgroupLayout(VirtualMMAIntrinsic intrinsic,
                                                int operandIndex);

MMASingleSubgroupLayout
getSingleSubgroupLayout(IREE::Codegen::InnerTileDescAttrInterface mmaKind,
                        int operandIndex);

MMASingleSubgroupLayout getSingleSubgroupLayout(ScaledMMAIntrinsic intrinsic,
                                                int64_t operandIndex);

/// Returns the name of the tilling `level`, as used in the `lowering_config`
/// attribute.
StringRef getTilingLevelName(GPU::TilingLevel level);

/// Returns the name of the padding `level`, as used in the `padding_config`
/// attribute.
StringRef getPaddingLevelName(GPU::PaddingLevel level);

//===----------------------------------------------------------------------===//
// Implementations for operand promotion
//===----------------------------------------------------------------------===//

Value cacheSwizzlePromotionImpl(OpBuilder &builder, OpOperand &operand,
                                Attribute attr);

} // namespace mlir::iree_compiler::IREE::GPU

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h.inc"
#undef GET_ATTRDEF_CLASSES
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS_H_
