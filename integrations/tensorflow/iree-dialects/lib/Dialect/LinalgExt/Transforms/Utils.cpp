// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

void mlir::iree_compiler::IREE::LinalgExt::completeOffsetsSizesAndStrides(
    OpBuilder &b, Location loc, Value tensor, ArrayRef<Value> leadingOffsets,
    ArrayRef<Value> leadingSizes, ArrayRef<Value> leadingStrides,
    SmallVectorImpl<Value> &offsets, SmallVectorImpl<Value> &sizes,
    SmallVectorImpl<Value> &strides) {
  assert(leadingOffsets.size() == leadingSizes.size() &&
         "expected matching lengths");
  assert(leadingSizes.size() == leadingStrides.size() &&
         "expected matching lengths");

  auto rankedTensorType = tensor.getType().cast<RankedTensorType>();
  int64_t tensorRank = rankedTensorType.getRank();
  int64_t leadingRank = leadingOffsets.size();
  offsets = SmallVector<Value>(leadingOffsets.begin(), leadingOffsets.end());
  sizes = SmallVector<Value>(leadingSizes.begin(), leadingSizes.end());
  strides = SmallVector<Value>(leadingStrides.begin(), leadingStrides.end());
  if (leadingRank >= tensorRank)
    return;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t i = leadingRank, e = tensorRank; i < e; ++i) {
    offsets.push_back(zero);
    sizes.push_back(b.createOrFold<tensor::DimOp>(loc, tensor, i));
    strides.push_back(one);
  }
}

/// Create a tensor::ExtractSliceOp by auto-completing the missing trailing
/// dimensions to always be offset = 0, size = dim, stride = 1.
Value mlir::iree_compiler::IREE::LinalgExt::
    createSubsetExtractOpFromLeadingOffsetsSizesAndStrides(
        OpBuilder &b, Location loc, Value tensor,
        ArrayRef<Value> leadingOffsets, ArrayRef<Value> leadingSizes,
        ArrayRef<Value> leadingStrides) {
  SmallVector<Value> offsets, sizes, strides;
  completeOffsetsSizesAndStrides(b, loc, tensor, leadingOffsets, leadingSizes,
                                 leadingStrides, offsets, sizes, strides);
  return b.createOrFold<tensor::ExtractSliceOp>(loc, tensor, offsets, sizes,
                                                strides);
}

/// Create a tensor::InsertSliceOp by auto-completing the missing trailing
/// dimensions to always be offset = 0, size = dim, stride = 1.
Value mlir::iree_compiler::IREE::LinalgExt::
    createSubsetInsertOpFromLeadingOffsetsSizesAndStrides(
        OpBuilder &b, Location loc, Value tensor, Value dest,
        ArrayRef<Value> leadingOffsets, ArrayRef<Value> leadingSizes,
        ArrayRef<Value> leadingStrides) {
  SmallVector<Value> offsets, sizes, strides;
  completeOffsetsSizesAndStrides(b, loc, tensor, leadingOffsets, leadingSizes,
                                 leadingStrides, offsets, sizes, strides);
  return b.createOrFold<tensor::InsertSliceOp>(loc, tensor, dest, offsets,
                                               sizes, strides);
}

/// Insert the `source` tensor into the `dest` tensor by creating the relevant
/// `subset_insert` op. The details of the `subset_insert` op are retrieved
/// from the `subset_extract` op so that they form a matching extract/insert
/// pair.
Value mlir::iree_compiler::IREE::LinalgExt::createMatchingSubsetInsertOp(
    OpBuilder &b, Location loc, tensor::ExtractSliceOp subsetExtractOp,
    Value source, Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, subsetExtractOp.getSource().getType(), source, dest,
      subsetExtractOp.offsets(), subsetExtractOp.sizes(),
      subsetExtractOp.strides(), subsetExtractOp.static_offsets(),
      subsetExtractOp.static_sizes(), subsetExtractOp.static_strides());
}

void mlir::iree_compiler::IREE::LinalgExt::createMatchingParallelSubsetInsertOp(
    OpBuilder &b, Location loc, tensor::ExtractSliceOp subsetExtractOp,
    Value source, Value dest) {
  b.create<tensor::ParallelInsertSliceOp>(
      loc, source, dest, subsetExtractOp.getMixedOffsets(),
      subsetExtractOp.getMixedSizes(), subsetExtractOp.getMixedStrides());
}
