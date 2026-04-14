// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
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

#define GEN_PASS_DEF_LOWERTRANSFERGATHERTOGATHERPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

/// Lower a transfer_gather with a single index vec (gathered on the last
/// source dim) to vector.gather.
struct LowerTransferGatherToVectorGather final
    : OpRewritePattern<TransferGatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(TransferGatherOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle single index vec (single symbol).
    if (op.getIndexVecs().size() != 1) {
      return rewriter.notifyMatchFailure(op, "expected exactly one index vec");
    }

    // Scalar index types become broadcasts, not gathers, skip them.
    if (!isa<VectorType>(op.getIndexVecs()[0].getType())) {
      return rewriter.notifyMatchFailure(op, "index vec must be a vector type");
    }

    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    AffineMap sourceMap = indexingMaps[0];

    // Check that the symbol appears only in the last source dimension.
    unsigned numResults = sourceMap.getNumResults();
    for (unsigned i = 0; i < numResults - 1; ++i) {
      if (sourceMap.getResult(i).isFunctionOfSymbol(0)) {
        return rewriter.notifyMatchFailure(
            op, "symbol must only appear in last source dim");
      }
    }
    AffineExpr lastExpr = sourceMap.getResult(numResults - 1);
    if (!isa<AffineSymbolExpr>(lastExpr)) {
      return rewriter.notifyMatchFailure(
          op, "last source dim must be a pure symbol expr");
    }

    // Check that all non-symbol source dims are constants or dim exprs.
    for (unsigned i = 0; i < numResults - 1; ++i) {
      AffineExpr expr = sourceMap.getResult(i);
      if (!isa<AffineConstantExpr>(expr)) {
        return rewriter.notifyMatchFailure(
            op, "non-gathered source dims must be constants or dim exprs");
      }
    }

    if (op.getMask()) {
      return rewriter.notifyMatchFailure(
          op, "masked transfer_gather not yet supported");
    }

    Location loc = op.getLoc();
    VectorType resultType = op.getVectorType();
    Value indexVec = op.getIndexVecs()[0];

    auto maskType =
        VectorType::get(resultType.getShape(), rewriter.getI1Type());
    Value mask = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(maskType, true));

    Value passthru =
        vector::BroadcastOp::create(rewriter, loc, resultType, op.getPadding());

    rewriter.replaceOpWithNewOp<vector::GatherOp>(op, resultType, op.getBase(),
                                                  op.getOffsets(), indexVec,
                                                  mask, passthru);
    return success();
  }
};

struct LowerTransferGatherToGatherPass final
    : impl::LowerTransferGatherToGatherPassBase<
          LowerTransferGatherToGatherPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerTransferGatherToVectorGather>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::VectorExt
