// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Analysis/SliceAnalysis.h"

#define DEBUG_TYPE "iree-codegen-pcf-transforms"

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// Loop/Generic op cleanup
//===----------------------------------------------------------------------===//

namespace {

static bool isBlockArgDroppable(Value v, SetVector<Operation *> &slice) {
  if (!cast<PCF::ShapedRefType>(v.getType()).isParentScopeOnlySync()) {
    return false;
  }
  bool droppable = true;
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    // TODO: Support more operations here.
    if (!isa<PCF::WriteSliceOp>(op)) {
      droppable = false;
      return false;
    }
    return true;
  };
  getForwardSlice(v, &slice, options);
  return droppable;
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
    if (!result.use_empty() || !isa<RankedTensorType>(result.getType()) ||
        !isBlockArgDroppable(regionArg, opsToErase)) {
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
