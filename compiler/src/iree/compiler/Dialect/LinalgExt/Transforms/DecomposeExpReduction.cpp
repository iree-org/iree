// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEEXPREDUCTIONPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

struct DecomposeExpReductionPass final
    : impl::DecomposeExpReductionPassBase<DecomposeExpReductionPass> {
  using DecomposeExpReductionPassBase::DecomposeExpReductionPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

static LogicalResult captureUsedOperationsAndBlockArguments(
    linalg::GenericOp linalgOp, SetVector<int64_t> &usedInputs,
    SetVector<Operation *> &usedOperations, int64_t resultNumber) {
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&linalgOp](Operation *op) -> bool {
    return op->getBlock() == linalgOp.getBody();
  };
  auto yieldOp = cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());
  Value result = yieldOp.getOperand(resultNumber);

  if (failed(getBackwardSlice(result, &usedOperations, options)))
    return failure();

  // Get all block arguments used by the operations. If any of the arguments
  // used is a dpsInit argument other than resultNumber, return failure.
  for (Operation *op : usedOperations) {
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() != linalgOp.getBlock()) {
          continue;
        }
        int64_t argNumber = blockArg.getArgNumber();
        if (argNumber >= linalgOp.getNumDpsInputs() &&
            argNumber - linalgOp.getNumDpsInputs() != resultNumber) {
          return failure();
        }
        if (argNumber < linalgOp.getNumDpsInputs()) {
          usedInputs.insert(argNumber);
        }
      }
    }
  }

  return success();
}

struct DecomposeMultipleResults : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (linalgOp.getNumResults() <= 2) {
      return failure();
    }
    // Create num_results linalg.generics, each producing a single result (and
    // relying on canonicalizations to simplify).
    for (int64_t resultNumber : llvm::seq<int64_t>(linalgOp.getNumResults())) {
      rewriter.setInsertionPoint(linalgOp);
      auto yieldOp =
          cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());
      Value result = yieldOp.getOperand(resultNumber);
      // Get all operations required to produce this result.
      SetVector<Operation *> usedOperations;
      SetVector<int64_t> usedInputs;
      if (failed(captureUsedOperationsAndBlockArguments(
              linalgOp, usedInputs, usedOperations, resultNumber))) {
        return failure();
      }
      // Create a new linalg.generic operation for this result.
      SmallVector<Value> inputs =
          llvm::map_to_vector(usedInputs, [&](int64_t x) {
            return linalgOp.getDpsInputOperand(x)->get();
          });
      SmallVector<Value> inits = {
          linalgOp.getDpsInitOperand(resultNumber)->get()};

      SmallVector<AffineMap> indexingMaps =
          llvm::map_to_vector(usedInputs, [&](int64_t x) {
            return linalgOp.getIndexingMapsArray()[x];
          });
      indexingMaps.push_back(linalgOp.getIndexingMapMatchingResult(
          linalgOp->getOpResult(resultNumber)));
      llvm::SmallBitVector unusedDims = getUnusedDimsBitVector(indexingMaps);
      indexingMaps = compressUnusedDims(indexingMaps);
      SmallVector<utils::IteratorType> iteratorTypes;
      for (int64_t i : llvm::seq<int64_t>(linalgOp.getNumLoops())) {
        if (!unusedDims.test(i)) {
          iteratorTypes.push_back(linalgOp.getIteratorTypesArray()[i]);
        }
      }
      auto newOp = linalg::GenericOp::create(
          rewriter, linalgOp.getLoc(), TypeRange(inits), inputs, inits,
          indexingMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
            Block *oldBody = linalgOp.getBody();
            usedInputs.insert(resultNumber + linalgOp.getNumDpsInputs());
            IRMapping regionMapping;
            for (auto [oldBlockArgNum, newBlockArg] :
                 llvm::zip_equal(usedInputs, blockArgs)) {
              regionMapping.map(oldBody->getArgument(oldBlockArgNum),
                                newBlockArg);
            }
            for (Operation *usedOperation : usedOperations) {
              b.clone(*usedOperation, regionMapping);
            }
            linalg::YieldOp::create(b, loc, regionMapping.lookup(result));
          });
      if (unusedDims.none()) {
        newOp->setDiscardableAttrs(linalgOp->getDiscardableAttrDictionary());
      }
      rewriter.replaceAllUsesWith(linalgOp.getResult(resultNumber),
                                  newOp.getResult(0));
    }

    return success();
  }
};

struct DecomposeExpReduction : OpRewritePattern<ExpReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpReductionOp expReductionOp,
                                PatternRewriter &rewriter) const override {
    auto decomposeResults = expReductionOp.decomposeOperation(rewriter);
    if (failed(decomposeResults)) {
      return failure();
    }
    rewriter.replaceOp(expReductionOp,
                       decomposeResults->begin()->getDefiningOp());
    return success();
  }
};

} // namespace

void DecomposeExpReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<DecomposeExpReduction, DecomposeMultipleResults>(context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitOpError("Failed to apply patterns");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::LinalgExt