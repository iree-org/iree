// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_LOWERTRANSFERGATHERSCATTERTOVECTORPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

static LogicalResult validateMaps(Operation *op, OperandRange indexVecs,
                                  AffineMap sourceMap,
                                  PatternRewriter &rewriter) {
  // TODO: vector.gather/scatter requires a single index vector.
  // We could flatten everything to 1-D to make it work.
  if (indexVecs.size() != 1) {
    return rewriter.notifyMatchFailure(op, "expected exactly one index vec");
  }

  if (!isa<VectorType>(indexVecs[0].getType())) {
    return rewriter.notifyMatchFailure(op, "index vec must be a vector type");
  }

  unsigned numResults = sourceMap.getNumResults();
  for (unsigned i = 0; i < numResults - 1; ++i) {
    if (!isa<AffineConstantExpr>(sourceMap.getResult(i))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-gathered dims must be constants");
    }
  }

  if (!isa<AffineSymbolExpr>(sourceMap.getResult(numResults - 1))) {
    return rewriter.notifyMatchFailure(op,
                                       "last dim must be a pure symbol expr");
  }
  return success();
}

struct LowerTransferGatherToVectorGather final
    : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    if (failed(
            validateMaps(op, op.getIndexVecs(), indexingMaps[0], rewriter))) {
      return failure();
    }

    Location loc = op.getLoc();
    VectorType resultType = op.getVectorType();
    Value indexVec = op.getIndexVecs()[0];

    Value mask = op.getMask();
    if (!mask) {
      auto maskType =
          VectorType::get(resultType.getShape(), rewriter.getI1Type());
      mask = arith::ConstantOp::create(rewriter, loc,
                                       DenseElementsAttr::get(maskType, true));
    }

    Value passthru =
        vector::BroadcastOp::create(rewriter, loc, resultType, op.getPadding());

    rewriter.replaceOpWithNewOp<vector::GatherOp>(op, resultType, op.getBase(),
                                                  op.getOffsets(), indexVec,
                                                  mask, passthru);
    return success();
  }
};

struct LowerTransferScatterToVectorScatter final
    : OpRewritePattern<TransferScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferScatterOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    if (failed(
            validateMaps(op, op.getIndexVecs(), indexingMaps[0], rewriter))) {
      return failure();
    }

    Location loc = op.getLoc();
    VectorType vectorType = op.getVectorType();
    Value indexVec = op.getIndexVecs()[0];

    Value mask = op.getMask();
    if (!mask) {
      auto maskType =
          VectorType::get(vectorType.getShape(), rewriter.getI1Type());
      mask = arith::ConstantOp::create(rewriter, loc,
                                       DenseElementsAttr::get(maskType, true));
    }

    Type resultType = op.hasTensorSemantics() ? op.getBase().getType() : Type{};
    auto scatterOp = vector::ScatterOp::create(rewriter, loc, resultType,
                                               op.getBase(), op.getOffsets(),
                                               indexVec, mask, op.getVector());
    if (op.hasTensorSemantics()) {
      rewriter.replaceOp(op, scatterOp.getResult());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct LowerTransferGatherScatterToVectorPass final
    : impl::LowerTransferGatherScatterToVectorPassBase<
          LowerTransferGatherScatterToVectorPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerTransferGatherToVectorGather,
                 LowerTransferScatterToVectorScatter>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateLowerTransferGatherScatterToVectorPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LowerTransferGatherToVectorGather,
               LowerTransferScatterToVectorScatter>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::VectorExt
