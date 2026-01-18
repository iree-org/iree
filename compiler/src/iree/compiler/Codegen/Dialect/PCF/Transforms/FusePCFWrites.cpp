// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-pcf-fuse-pcf-writes"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_FUSEPCFWRITESPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

struct FusePCFWritesPass final
    : impl::FusePCFWritesPassBase<FusePCFWritesPass> {
  void runOnOperation() override;
};

/// Pattern to fuse pcf.write_slice with tensor.parallel_insert_slice from
/// scf.forall terminators.
struct FuseWriteSliceWithParallelInsert final
    : OpRewritePattern<PCF::WriteSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(PCF::WriteSliceOp writeSliceOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<PCF::WriteSliceOp> newWriteSlice =
        composeWriteSliceWithParallelInsert(rewriter, writeSliceOp);
    if (failed(newWriteSlice)) {
      return rewriter.notifyMatchFailure(
          writeSliceOp,
          "source is not an scf.forall with tensor.parallel_insert_slice");
    }
    return success();
  }
};

void FusePCFWritesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.add<FuseWriteSliceWithParallelInsert>(context);

  // Forall canonicalizations to drop unused results.
  scf::ForallOp::getCanonicalizationPatterns(patterns, &getContext());

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

FailureOr<PCF::WriteSliceOp>
composeWriteSliceWithParallelInsert(RewriterBase &rewriter,
                                    PCF::WriteSliceOp writeSliceOp) {
  // Check if the source is produced by an scf.forall.
  auto forallOp = writeSliceOp.getSource().getDefiningOp<scf::ForallOp>();
  if (!forallOp) {
    return failure();
  }

  // Get the result index being written.
  auto forallResult = dyn_cast<OpResult>(writeSliceOp.getSource());
  if (!forallResult) {
    return failure();
  }
  unsigned resultIdx = forallResult.getResultNumber();

  // Get the in_parallel terminator
  auto inParallelOp =
      cast<scf::InParallelOp>(forallOp.getRegion().front().getTerminator());

  // Find the tensor.parallel_insert_slice for this result.
  tensor::ParallelInsertSliceOp insertSliceOp = nullptr;
  for (Operation &op : inParallelOp.getYieldingOps()) {
    if (auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op)) {
      // Check if this insert targets the correct shared_out
      auto destArg = dyn_cast<BlockArgument>(insertOp.getDest());
      if (destArg && destArg.getOwner() == &forallOp.getRegion().front()) {
        // Map block argument to result index
        unsigned argIdx = destArg.getArgNumber() - forallOp.getRank();
        if (argIdx == resultIdx) {
          if (insertSliceOp) {
            return rewriter.notifyMatchFailure(
                forallOp, "unimplemented: multiple insert_slice producers");
          }
          insertSliceOp = insertOp;
        }
      }
    }
  }

  if (!insertSliceOp) {
    return failure();
  }

  // Collect all values used by the write_slice that are not the forall result.
  // These need to be available inside the forall body.
  SmallVector<Value> writeSliceOperands;
  writeSliceOperands.push_back(writeSliceOp.getDest());
  llvm::append_range(writeSliceOperands, writeSliceOp.getOffsets());
  llvm::append_range(writeSliceOperands, writeSliceOp.getStrides());

  // Move the definitions of these operands before the forall if they are
  // defined after it. This can happen if the producer of an operand dominates
  // the forall but is placed after it in the IR.
  if (failed(moveValueDefinitions(rewriter, writeSliceOperands, forallOp))) {
    return rewriter.notifyMatchFailure(
        writeSliceOp,
        "failed to move write_slice operand definitions before forall");
  }

  // Compose the offsets, sizes, and strides and insert the new write_slice
  // before the parallel_insert_slice in the forall body.
  // The new write_slice should use:
  // - source: insertSlice.getSource()
  // - dest: writeSlice.getDest()
  // - offsets: writeSlice.offsets + insertSlice.offsets * writeSlice.strides
  // - sizes: insertSlice.sizes
  // - strides: writeSlice.strides * insertSlice.strides

  SmallVector<OpFoldResult> composedOffsets;
  SmallVector<OpFoldResult> composedSizes = insertSliceOp.getMixedSizes();
  SmallVector<OpFoldResult> composedStrides;

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before the in_parallel terminator, not inside it.
  rewriter.setInsertionPoint(inParallelOp);

  SmallVector<OpFoldResult> writeOffsets = writeSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> insertOffsets = insertSliceOp.getMixedOffsets();
  SmallVector<OpFoldResult> writeStrides = writeSliceOp.getMixedStrides();
  SmallVector<OpFoldResult> insertStrides = insertSliceOp.getMixedStrides();

  // Compose offsets: writeOffset + insertOffset * writeStride.
  for (auto [writeOffset, insertOffset, writeStride] :
       llvm::zip_equal(writeOffsets, insertOffsets, writeStrides)) {
    Value writeOffsetVal = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), writeOffset);
    Value insertOffsetVal = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), insertOffset);
    Value writeStrideVal = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), writeStride);

    Value scaled = rewriter.createOrFold<arith::MulIOp>(
        insertSliceOp.getLoc(), insertOffsetVal, writeStrideVal);
    Value composed = rewriter.createOrFold<arith::AddIOp>(
        insertSliceOp.getLoc(), writeOffsetVal, scaled);
    composedOffsets.push_back(composed);
  }

  // Compose strides: writeStride * insertStride.
  for (auto [writeStride, insertStride] :
       llvm::zip_equal(writeStrides, insertStrides)) {
    Value writeStrideVal = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), writeStride);
    Value insertStrideVal = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), insertStride);
    Value composed = rewriter.createOrFold<arith::MulIOp>(
        insertSliceOp.getLoc(), writeStrideVal, insertStrideVal);
    composedStrides.push_back(composed);
  }

  // Handle rank-reduced parallel_insert_slice sources.
  // The source may have fewer dimensions than the destination sref (e.g.,
  // tensor<1024xf32> being inserted into tensor<512x10240xf32> with sizes
  // [1, 1024]). We need to expand the source to match the sref rank.
  Value source = insertSliceOp.getSource();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto srefType = cast<ShapedRefType>(writeSliceOp.getDest().getType());
  int64_t sourceRank = sourceType.getRank();
  int64_t destRank = srefType.getRank();

  if (sourceRank < destRank) {
    // Build reassociation map for expand_shape.
    // Unit dimensions (size 1) from composedSizes indicate dropped dimensions.
    SmallVector<int64_t> expandedShape;
    SmallVector<ReassociationIndices> reassociation;

    int64_t sourceIdx = 0;
    ReassociationIndices currentGroup;

    for (int64_t i = 0; i < destRank; ++i) {
      // Check if this dimension is a unit dimension (was dropped in rank
      // reduction).
      std::optional<int64_t> staticSize = getConstantIntValue(composedSizes[i]);
      bool isUnitDim = staticSize && *staticSize == 1;

      if (isUnitDim && sourceIdx < sourceRank) {
        // This is a unit dimension, check if it matches the source.
        int64_t sourceDimSize = sourceType.getDimSize(sourceIdx);
        if (sourceDimSize == 1) {
          // Source also has this as size 1, include it normally.
          expandedShape.push_back(1);
          currentGroup.push_back(i);
          sourceIdx++;
          if (sourceIdx <= sourceRank) {
            reassociation.push_back(currentGroup);
            currentGroup.clear();
          }
        } else {
          // This is a dropped dimension, add to current group.
          expandedShape.push_back(1);
          currentGroup.push_back(i);
        }
      } else if (sourceIdx < sourceRank) {
        // Non-unit dimension, maps to a source dimension.
        expandedShape.push_back(sourceType.getDimSize(sourceIdx));
        currentGroup.push_back(i);
        sourceIdx++;
        reassociation.push_back(currentGroup);
        currentGroup.clear();
      } else {
        // Extra trailing unit dimensions.
        expandedShape.push_back(1);
        if (!reassociation.empty()) {
          reassociation.back().push_back(i);
        } else {
          currentGroup.push_back(i);
        }
      }
    }

    // Handle any remaining group.
    if (!currentGroup.empty()) {
      reassociation.push_back(currentGroup);
    }

    auto expandedType =
        RankedTensorType::get(expandedShape, sourceType.getElementType());
    source = tensor::ExpandShapeOp::create(rewriter, insertSliceOp.getLoc(),
                                           expandedType, source, reassociation)
                 .getResult();
  }

  // Create the new write_slice before the terminator.
  auto newWriteSlice = PCF::WriteSliceOp::create(
      rewriter, insertSliceOp.getLoc(), source, writeSliceOp.getDest(),
      composedOffsets, composedSizes, composedStrides);

  // Erase the old write_slice.
  rewriter.eraseOp(writeSliceOp);

  return cast<PCF::WriteSliceOp>(newWriteSlice.getOperation());
}

} // namespace mlir::iree_compiler::IREE::PCF
