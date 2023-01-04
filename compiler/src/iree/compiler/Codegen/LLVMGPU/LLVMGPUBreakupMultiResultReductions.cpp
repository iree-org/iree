// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-breakup-multi-result-reductions"

namespace mlir {
namespace iree_compiler {

static FailureOr<std::array<Operation*, 2>> breakupReductionOp(
    RewriterBase& b, linalg::GenericOp op) {
  Operation* cloneOp = b.clone(*op.getOperation());
  for (int64_t i = 0, e = op.getNumDpsInits(); i < e; i++) {
    // Just replace the use of the reduction and let the dead code elimination
    // handle the clean up.
    SmallVector<Operation*, 4> combinerOps;
    if (matchReduction(op.getRegionOutputArgs(), i, combinerOps) &&
        combinerOps.size() == 1) {
      op.getResult(i).replaceAllUsesWith(cloneOp->getResult(i));
    }
  }
  return std::array<Operation*, 2>({op.getOperation(), cloneOp});
}

static bool containsDim(AffineMap map, unsigned dim) {
  for (AffineExpr expr : map.getResults()) {
    if (auto exprDim = expr.dyn_cast<AffineDimExpr>()) {
      if (exprDim.getPosition() == dim) {
        return true;
      }
    }
  }
  return false;
}

template <typename OpTy>
static SmallVector<NamedAttribute> pruneAttributeList(OpTy op) {
  auto opAttributes = op.getAttributeNames();
  llvm::StringSet<> elidedAttrs;
  elidedAttrs.insert(opAttributes.begin(), opAttributes.end());
  SmallVector<NamedAttribute> preservedAttrs;
  for (auto attr : op->getAttrs()) {
    if (elidedAttrs.count(attr.getName())) continue;
    preservedAttrs.push_back(attr);
  }
  return preservedAttrs;
}

namespace {

/// Convert reduction dimensions to parallel if there are no loop carried
/// dependencies.
struct ConvertReductionDims : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<unsigned> reductionDims;
    linalgOp.getReductionDims(reductionDims);
    SmallVector<unsigned> dimsToParallelize;
    for (unsigned dim : reductionDims) {
      if (llvm::all_of(linalgOp.getResults(), [&](OpResult result) {
            return containsDim(linalgOp.getIndexingMapMatchingResult(result),
                               dim);
          })) {
        dimsToParallelize.push_back(dim);
      }
    }
    if (dimsToParallelize.empty()) return failure();
    SmallVector<utils::IteratorType> newIteratorTypes =
        linalgOp.getIteratorTypesArray();
    for (unsigned dim : dimsToParallelize) {
      newIteratorTypes[dim] = utils::IteratorType::parallel;
    }
    SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
    auto genericOp = rewriter.create<linalg::GenericOp>(
        linalgOp.getLoc(), linalgOp.getResultTypes(), linalgOp.getInputs(),
        linalgOp.getOutputs(), linalgOp.getIndexingMapsArray(),
        newIteratorTypes);
    // Forward lowering config.
    if (auto loweringAttr = getLoweringConfig(linalgOp)) {
      setLoweringConfig(genericOp, loweringAttr);
    }
    BlockAndValueMapping mapping;
    linalgOp->getRegion(0).cloneInto(&genericOp.getRegion(),
                                     genericOp.getRegion().begin(), mapping);
    rewriter.replaceOp(linalgOp, genericOp.getResults());
    return success();
  }
};

/// Merge elementwise operations into their consumers.
struct MergeElementwiseOps : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand& opOperand : genericOp->getOpOperands()) {
      if (!linalg::areElementwiseOpsFusable(&opOperand)) continue;

      FailureOr<Operation*> fusedOp =
          linalg::fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusedOp)) {
        // Forward lowering config.
        if (auto loweringAttr = getLoweringConfig(genericOp)) {
          setLoweringConfig(fusedOp.value(), loweringAttr);
        }
        auto replacements =
            fusedOp.value()->getResults().take_back(genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

struct LLVMGPUBreakupMultiResultReductionsPass
    : public LLVMGPUBreakupMultiResultReductionsBase<
          LLVMGPUBreakupMultiResultReductionsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::GenericOp> reductions;
    funcOp.walk([&](linalg::GenericOp op) {
      if (op.getNumReductionLoops() > 0 && op.getNumResults() > 1)
        reductions.push_back(op);
    });
    for (linalg::GenericOp op : reductions) {
      IRRewriter rewriter(funcOp.getContext());
      rewriter.setInsertionPoint(op);
      if (failed(breakupReductionOp(rewriter, op))) {
        return signalPassFailure();
      }
    }

    // Clean up dead code within linalg ops.
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Convert reduction dims.
    RewritePatternSet patterns2(funcOp.getContext());
    patterns2.insert<ConvertReductionDims>(funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns2)))) {
      return signalPassFailure();
    }

    // merge linalg.generic ops.
    {
      RewritePatternSet fusionPatterns(funcOp.getContext());
      fusionPatterns.insert<MergeElementwiseOps>(funcOp.getContext());
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(fusionPatterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUBreakupMultiResultReductionsPass() {
  return std::make_unique<LLVMGPUBreakupMultiResultReductionsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
