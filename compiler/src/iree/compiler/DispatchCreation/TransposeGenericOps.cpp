// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- InterchangeTransposeGenericOps.cpp -------------------===//
//
// Interchange loops in generic ops to make the transpose happen on the outputs
// instead of inputs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_TRANSPOSEGENERICOPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// MakeReductionInnermostPattern
//===----------------------------------------------------------------------===//

/// For generic ops that are reduction, make the reduction the innermost
/// dimension.
struct MakeReductionInnermostPattern final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> interchange;
    bool needInterchange = false;
    unsigned numParallelLoop = genericOp.getNumParallelLoops();
    if (numParallelLoop == 0)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isParallelIterator(iter.value())) {
        interchange.push_back(iter.index());
        if (iter.index() >= numParallelLoop)
          needInterchange = true;
      }
    }
    // If all the parallel loops are outter loops skip the pattern.
    if (!needInterchange)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter.value())) {
        interchange.push_back(iter.index());
      }
    }
    return interchangeGenericOp(rewriter, genericOp, interchange);
  }
};

/// For elementwise ops that consumer values produced by named ops (or reduction
/// ops), the dispatch region fusion logic requires the indexing maps to be
/// identity (or projections that are not transposing as well). This pattern
/// fixes up elementwise operations for which that is not the case.
struct TransposeGenericOpPattern final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(genericOp, "no tensor semantics");
    }

    // Pattern needs to trigger only on  elementwise ops.
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
      return rewriter.notifyMatchFailure(genericOp, "not elementwise");
    }

    std::optional<AffineMap> mapForInterchange;

    for (auto operand : genericOp.getDpsInputOperands()) {
      // Check that the producer is a named op or a reduction op (i.e. not
      // elementwise op) with a single use.
      auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
      if (!producer || !llvm::hasSingleElement(producer->getUsers()) ||
          linalg::isElementwise(producer))
        continue;

      // check if the generic op has a non-identity map for the operand.
      auto indexingMap = genericOp.getMatchingIndexingMap(operand);
      // This is already identity. Nothing to do.
      if (indexingMap.isIdentity()) {
        return rewriter.notifyMatchFailure(genericOp, "already normalized");
      }
      // The map must be a permutation. If not, then look for other operand.
      if (!indexingMap.isPermutation())
        continue;

      if (!mapForInterchange)
        mapForInterchange = indexingMap;
    }

    if (!mapForInterchange) {
      return rewriter.notifyMatchFailure(genericOp, "no eligible operands");
    }
    // Make the input indexing maps identity by interchanging.
    auto interchange =
        llvm::map_to_vector(mapForInterchange->getResults(), [](AffineExpr e) {
          return cast<AffineDimExpr>(e).getPosition();
        });

    return interchangeGenericOp(rewriter, genericOp, interchange);
  }
};

struct TransposeGenericOpsPass final
    : public impl::TransposeGenericOpsPassBase<TransposeGenericOpsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MakeReductionInnermostPattern, TransposeGenericOpPattern>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
