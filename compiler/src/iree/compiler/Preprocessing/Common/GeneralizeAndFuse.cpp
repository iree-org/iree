// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

bool isMatmulOrBatchMatmul(linalg::LinalgOp linalgOp) {
  return linalg::isaContractionOpInterface(linalgOp) &&
         llvm::is_contained({2u, 3u}, linalgOp.getNumParallelLoops());
}

//===----------------------------------------------------------------------===//
// Generalize and fusion patterns.
//===----------------------------------------------------------------------===//

struct GeneralizeAndFusePass
    : public GeneralizeAndFuseBase<GeneralizeAndFusePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  template <typename LinalgOpType>
  class GeneralizeTargetNamedOpPattern final
      : public OpRewritePattern<LinalgOpType> {
   public:
    using OpRewritePattern<LinalgOpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(LinalgOpType linalgOp,
                                  PatternRewriter &rewriter) const override {
      // TODO: Check consumer is transposeOp.
      // TODO: Generalize transpos
      FailureOr<linalg::GenericOp> genericOp =
          linalg::generalizeNamedOp(rewriter, linalgOp);
      if (failed(genericOp)) return failure();
      return success();
    }
  };

  class FuseMatmulAndTranspose final
      : public OpRewritePattern<linalg::GenericOp> {
   public:
    using OpRewritePattern::OpRewritePattern;

    // Inspo:
    // https://github.com/llvm/llvm-project/blob/4f1c12425179608298dc39f5524ba2612609b5e4/mlir/lib/Dialect/Utils/StructuredOpsUtils.cpp
    LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                  PatternRewriter &rewriter) const override {
      const unsigned rhsId = 1;
      if (!isMatmulOrBatchMatmul(linalgOp)) return failure();
      Value rhs = linalgOp.getDpsInputOperand(rhsId)->get();
      auto transposeOp = dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp());
      if (!transposeOp) return failure();
      auto perm = transposeOp.getPermutation();
      auto indexingMaps = linalgOp.getIndexingMaps();
      auto rhsMap = indexingMaps[rhsId].cast<AffineMapAttr>().getValue();
      int64_t rank = perm.size();
      if (rhsMap.getNumResults() != rank) return failure();
      SmallVector<AffineExpr, 3> exprs;
      for (auto dim_id : perm) {
        exprs.push_back(rhsMap.getResult(dim_id));
      }
      AffineMap transposedRhsMap =
          AffineMap::get(rhsMap.getNumDims(), 0, exprs, getContext());

      // TODO: Fold transposeOp as transposed indexing for matmulOp.
      // Generate a map set.
      auto lhsMap = indexingMaps[0].cast<AffineMapAttr>().getValue();
      auto accMap = indexingMaps[2].cast<AffineMapAttr>().getValue();
      SmallVector<AffineMap> newIndexingMaps = {lhsMap, transposedRhsMap,
                                                accMap};

      // Generate new list of args.
      Value newRhs = transposeOp.getDpsInputOperand(0)->get();
      Value lhs = linalgOp.getDpsInputOperand(0)->get();
      Value acc = linalgOp.getDpsInitOperand(0)->get();
      SmallVector<Value> inputs = {lhs, newRhs};

      // Generate a new genericOp.
      linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
          linalgOp.getLoc(), linalgOp.getResultTypes(), /*inputs*/ inputs,
          /*outputs*/ acc, newIndexingMaps, linalgOp.getIteratorTypesArray());
      // Block consumerBlock = linalgOp->getRegion(0).front();
      // genericOp.getRegion().push_back(consumerBlock);
      // llvm::outs()<<"new op
      // regions:"<<genericOp.getOperation()->getNumRegions()<<"\n";
      // llvm::outs()<<"new op
      // regions:"<<genericOp.getOperation()->getNumRegions()<<"\n";
      // llvm::outs()<<"new op
      // blocks:"<<genericOp.getRegion().getBlocks().size()<<"\n";
      // llvm::outs()<<"old op
      // regions:"<<linalgOp.getOperation()->getNumRegions()<<"\n";
      // llvm::outs()<<"old op
      // blocks:"<<linalgOp.getRegion().getBlocks().size()<<"\n";
      rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                  genericOp.getRegion().begin());
      rewriter.replaceOp(linalgOp, genericOp->getResults());
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    // Generalize + Fuse pattern.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<GeneralizeTargetNamedOpPattern<linalg::MatmulOp>,
                      FuseMatmulAndTranspose>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createGeneralizeAndFusePass() {
  return std::make_unique<GeneralizeAndFusePass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
