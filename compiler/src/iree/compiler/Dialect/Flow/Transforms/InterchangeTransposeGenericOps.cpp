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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

struct TransposeGenericOpPattern : public OpRewritePattern<linalg::GenericOp> {
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
      auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
      if (!producer)
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

struct InterchangeTransposeGenericOpsPass
    : public InterchangeTransposeGenericOpsBase<
          InterchangeTransposeGenericOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeGenericOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createInterchangeTransposeGenericOpsPass() {
  return std::make_unique<InterchangeTransposeGenericOpsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
