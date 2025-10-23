// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper to retrieve/set a target mma intrinsic.
IREE::Codegen::InnerTileDescAttrInterface getMmaKind(LoweringConfigAttr config);
void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                IREE::Codegen::InnerTileDescAttrInterface kind);

// The basis consists of two integer arrays:
//   - "counts": number of resource to use per dimension in the basis.
//   - "mapping": a projected permutation to map to basis to the operations
//     iteration space.
//
// Given a resource "x", the "basis" can be used to determine the distribution
// of an iteration space using:
//
// b = delinearize(x, counts)
// idx = apply(b, mapping)
struct Basis {
  SmallVector<int64_t> counts;
  SmallVector<int64_t> mapping;
};

// Dimension Expansion consists of a list of ReassociationIndices that specifies
// how each dimension in the original iteration space maps to the expanded
// iteration space. Each dimension can either remain unchanged or be split.
//
// For example, given a DimensionExpansion of [[0], [1], [2, 3]], the iteration
// space corresponding to the first and second dimensions remain unchanged. The
// iteration space corresponding to the third dimension index is split into
// two dimensions.
//
// The factor of expansion is determined by the `thread` level tiling sizes of
// the corresponding dimensions in the expanded space. The iteration
// space of the original dimension is split by the product of the thread tile
// sizes for all dimensions it maps to. For the above example, if we have thread
// = [0, 0, 1, 4], then:
// - Original d2 is split such that: d2_outer = d2 / 4, d2_inner = 4.
// - In the expanded space: thread[2] = 1, thread[3] = 4.
// - The product (1 * 4 = 4) determines the split factor of the iteration space
//   corresponding to partial_reduction[d2].
using DimensionExpansion = SmallVector<ReassociationIndices>;
FailureOr<DimensionExpansion>
getDimensionExpansion(IREE::GPU::LoweringConfigAttr config);

// Helper to retrieve/set distribution basis.
FailureOr<Basis> getBasis(IREE::GPU::LoweringConfigAttr config,
                          IREE::GPU::TilingLevel level);
void setBasis(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, const Basis &basis);

/// Helper to retrieve a list of operand indices to promote.
std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config);
/// Helper to retrieve a list of operand promotion types.
std::optional<ArrayRef<Attribute>>
getPromotionTypesList(LoweringConfigAttr config);
/// Append to `attrs` an `ArrayAttr` for `promotedOperands`.
/// The `promotionTypes` is an optional list of Attributes
/// describing how to promote each individual operand.
void appendPromotedOperandsList(MLIRContext *context,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands,
                                ArrayRef<Attribute> promotionTypes = {});
/// Create a new `LoweringConfigAttr` from `currAttr` with the promoted operands
/// list modified/set to `operands`. Optional `promotionTypes` specifies how to
/// promote each operand.
IREE::GPU::LoweringConfigAttr setPromotedOperandsList(
    MLIRContext *context, IREE::GPU::LoweringConfigAttr currAttr,
    ArrayRef<int64_t> operands,
    std::optional<ArrayRef<Attribute>> promotionTypes = std::nullopt);

/// Helper to retrieve list of operand to pad.
std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config,
                                                   bool paddingConv = false);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_GPULOWERINGCONFIGUTILS_H_
