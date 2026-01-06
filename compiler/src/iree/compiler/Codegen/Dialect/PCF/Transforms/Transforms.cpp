// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-codegen-pcf-transforms"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Loop/Generic op cleanup
//===----------------------------------------------------------------------===//

namespace {

static bool isBlockArgDroppable(Value v, SetVector<Operation *> &opsToErase,
                                SetVector<Operation *> &opsToReplace) {
  if (!cast<PCF::ShapedRefType>(v.getType()).isReturnOnlySync()) {
    return false;
  }
  bool droppable = true;
  SetVector<Operation *> slice;
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    // TODO: Support more operations here.
    if (isa<PCF::WriteSliceOp, PCF::ReadSliceOp>(op)) {
      return true;
    } else {
      droppable = false;
      return false;
    }
  };
  getForwardSlice(v, &slice, options);

  // Separate operations into sets for erasure and replacement.
  // Since WriteSliceOp doesn't return anything we can just erase it, but
  // we need to replace the value of the ReadSliceOp with the tied init.
  for (Operation *op : slice) {
    if (isa<PCF::WriteSliceOp>(op)) {
      opsToErase.insert(op);
    } else if (isa<PCF::ReadSliceOp>(op)) {
      opsToReplace.insert(op);
    }
  }

  return droppable;
}

/// Replace ReadSliceOp uses when dropping unused results. If the sref argument
/// has a tied init, extract from it. Otherwise, create an empty tensor and
/// extract from that.
template <typename OpTy>
static LogicalResult replaceReadSliceOps(RewriterBase &rewriter, OpTy op,
                                         BlockArgument regionArg,
                                         SetVector<Operation *> &opsToReplace) {
  OpOperand *tiedInit =
      op.getTiedInit(regionArg.getArgNumber() -
                     op.getRegion().getNumArguments() + op->getNumResults());

  for (Operation *opToReplace : opsToReplace) {
    auto readOp = cast<PCF::ReadSliceOp>(opToReplace);
    if (readOp.getSource() != regionArg) {
      continue;
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(readOp);

    // Either read from a tied init if it exists or from an empty init based
    // on the result sizes.
    Value srcToReadFrom;
    if (tiedInit) {
      srcToReadFrom = tiedInit->get();
    } else {
      auto tensorType = cast<RankedTensorType>(readOp.getResultType());

      // Collect dynamic sizes from the read op's sizes
      SmallVector<Value> dynamicSizes;
      for (auto [idx, size] : llvm::enumerate(readOp.getMixedSizes())) {
        if (tensorType.isDynamicDim(idx)) {
          if (auto attr = dyn_cast<Attribute>(size)) {
            dynamicSizes.push_back(arith::ConstantIndexOp::create(
                rewriter, readOp.getLoc(), cast<IntegerAttr>(attr).getInt()));
          } else {
            dynamicSizes.push_back(cast<Value>(size));
          }
        }
      }

      srcToReadFrom = tensor::EmptyOp::create(
          rewriter, readOp.getLoc(), tensorType.getShape(),
          tensorType.getElementType(), dynamicSizes);
    }

    if (isa<VectorType>(readOp.getResultType())) {
      SmallVector<Value> indices;
      for (OpFoldResult offset : readOp.getMixedOffsets()) {
        if (auto attr = dyn_cast<Attribute>(offset)) {
          indices.push_back(arith::ConstantIndexOp::create(
              rewriter, readOp.getLoc(), cast<IntegerAttr>(attr).getInt()));
        } else {
          indices.push_back(cast<Value>(offset));
        }
      }
      auto vectorType = cast<VectorType>(readOp.getResultType());
      SmallVector<bool> inBounds(vectorType.getRank(), true);
      for (auto [inBound, vecSize, size] : llvm::zip_equal(
               inBounds, vectorType.getShape(), readOp.getMixedSizes())) {
        // If the size is dynamic or doesn't match the vector dimension, it's
        // out of bounds.
        if (auto attr = dyn_cast<Attribute>(size)) {
          inBound = vecSize == cast<IntegerAttr>(attr).getInt();
        } else {
          inBound = false;
        }
      }
      Value transferRead = vector::TransferReadOp::create(
          rewriter, readOp.getLoc(), vectorType, srcToReadFrom, indices,
          /*padding=*/std::nullopt, inBounds);
      rewriter.replaceOp(readOp, transferRead);
    } else {
      auto extractSlice = tensor::ExtractSliceOp::create(
          rewriter, readOp.getLoc(), srcToReadFrom, readOp.getMixedOffsets(),
          readOp.getMixedSizes(), readOp.getMixedStrides());
      rewriter.replaceOp(readOp, extractSlice.getResult());
    }
  }

  return success();
}

PCF::GenericOp cloneWithNewResultTypes(RewriterBase &rewriter,
                                       PCF::GenericOp genericOp,
                                       TypeRange newResultTypes,
                                       ArrayRef<Value> newTiedArgs,
                                       ArrayRef<Value> newDynamicSizes,
                                       ArrayRef<bool> newIsTied) {
  auto newGenericOp = PCF::GenericOp::create(
      rewriter, genericOp.getLoc(), newResultTypes, genericOp.getScope(),
      newTiedArgs, newDynamicSizes, newIsTied, genericOp.getNumIterators(),
      genericOp.getSyncOnReturn());
  newGenericOp.getRegion().takeBody(genericOp.getRegion());
  newGenericOp.getInitializer().takeBody(genericOp.getInitializer());
  newGenericOp.setNumLeadingArgs(genericOp.getNumLeadingArgs());
  return newGenericOp;
}

PCF::LoopOp cloneWithNewResultTypes(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                    TypeRange newResultTypes,
                                    ArrayRef<Value> newTiedArgs,
                                    ArrayRef<Value> newDynamicSizes,
                                    ArrayRef<bool> newIsTied) {
  auto newLoopOp =
      PCF::LoopOp::create(rewriter, loopOp.getLoc(), newResultTypes,
                          loopOp.getScope(), loopOp.getCount(), newTiedArgs,
                          newDynamicSizes, newIsTied, loopOp.getSyncOnReturn());
  newLoopOp.getRegion().takeBody(loopOp.getRegion());
  return newLoopOp;
}

template <typename OpTy>
static LogicalResult dropUnusedResults(RewriterBase &rewriter, OpTy op) {
  // Append the parameters for the new results to the existing lists.
  SmallVector<Type> newResultTypes;
  SmallVector<bool> newIsTied;
  SmallVector<Value> newDynamicSizes;
  SmallVector<Value> newTiedArgs;

  SmallVector<unsigned> blockArgsToDrop;
  SmallVector<Value> resultsToKeep;

  int64_t currSizesIndex = 0;
  int64_t currTiedIndex = 0;
  for (OpResult result : op->getResults()) {
    int64_t resultNum = result.getResultNumber();
    bool isTied = op.getIsTied()[resultNum];
    BlockArgument regionArg = op.getRegionRefArgs()[resultNum];
    // First verify that the result is droppable.
    SetVector<Operation *> opsToErase;
    SetVector<Operation *> opsToReplace;
    if (!result.use_empty() || !isa<RankedTensorType>(result.getType()) ||
        !isBlockArgDroppable(regionArg, opsToErase, opsToReplace)) {
      resultsToKeep.push_back(result);
      newResultTypes.push_back(result.getType());
      newIsTied.push_back(isTied);
      if (isTied) {
        newTiedArgs.push_back(op.getInits()[currTiedIndex]);
        ++currTiedIndex;
      } else {
        int64_t numDynamicDims =
            cast<ShapedType>(result.getType()).getNumDynamicDims();
        ValueRange currDynamicSizes =
            op.getDynamicSizes().slice(currSizesIndex, numDynamicDims);
        newDynamicSizes.append(currDynamicSizes.begin(),
                               currDynamicSizes.end());
        currSizesIndex += numDynamicDims;
      }
      continue;
    }

    // Replace all reads with direct reads from either the tied init or an empty
    // initial value.
    if (failed(replaceReadSliceOps(rewriter, op, regionArg, opsToReplace))) {
      return failure();
    }

    // Iterate the ops to erase in reverse to make sure ops without users are
    // erased first.
    for (auto op : llvm::reverse(opsToErase)) {
      rewriter.eraseOp(op);
    }

    blockArgsToDrop.push_back(regionArg.getArgNumber());

    // Increment the dynamic dim/tied operand counters to skip over them.
    if (isTied) {
      ++currTiedIndex;
    } else {
      currSizesIndex += cast<ShapedType>(result.getType()).getNumDynamicDims();
    }
  }

  if (newResultTypes.size() == op->getNumResults()) {
    return failure();
  }

  OpTy newOp = cloneWithNewResultTypes(rewriter, op, newResultTypes,
                                       newTiedArgs, newDynamicSizes, newIsTied);

  // Erase the block arguments associated with the dropped results. Iterate in
  // reverse so that we don't have to update the argument indices as we go.
  for (unsigned bbArgIndex : llvm::reverse(blockArgsToDrop)) {
    Block &entryBlock = newOp.getRegion().front();
    entryBlock.eraseArgument(bbArgIndex);
  }

  rewriter.replaceAllUsesWith(resultsToKeep, newOp.getResults());
  rewriter.eraseOp(op);
  return success();
}

struct DropUnusedGenericResult : public OpRewritePattern<IREE::PCF::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    return dropUnusedResults(rewriter, genericOp);
  }
};

struct DropUnusedLoopResult : public OpRewritePattern<IREE::PCF::LoopOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::PCF::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    return dropUnusedResults(rewriter, loopOp);
  }
};
} // namespace

void populatePCFDropUnusedResultPatterns(RewritePatternSet &patterns) {
  patterns.add<DropUnusedGenericResult, DropUnusedLoopResult>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::PCF
