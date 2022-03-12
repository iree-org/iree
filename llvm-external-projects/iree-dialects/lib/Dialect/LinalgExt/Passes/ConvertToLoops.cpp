// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

/// Recursive method that lowers one dimension of the `TiledOpInterface` to
/// scalar loops at a time.
static LogicalResult lowerToLoopsImpl(OpBuilder &builder,
                                      TiledOpInterface tilableOp,
                                      ArrayRef<Range> loopRanges,
                                      unsigned loopDepth,
                                      SmallVectorImpl<Value> &ivs) {
  Location loc = tilableOp.getLoc();
  if (loopDepth == loopRanges.size()) {
    return tilableOp.generateScalarImplementation(builder, loc, ivs);
  }
  LogicalResult status = success();
  builder.create<scf::ForOp>(
      loc, loopRanges[loopDepth].offset, loopRanges[loopDepth].size,
      loopRanges[loopDepth].stride, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        ivs.push_back(iv);
        status = lowerToLoopsImpl(b, tilableOp, loopRanges, loopDepth + 1, ivs);
        b.create<scf::YieldOp>(loc);
      });
  return status;
}

/// Main entry point for lowering `TiledOpInterface` op to loops.
static LogicalResult lowerToLoops(OpBuilder &builder,
                                  TiledOpInterface tilableOp) {
  SmallVector<Range> loopBounds = tilableOp.getIterationDomain(builder);
  SmallVector<Value> ivs;
  return lowerToLoopsImpl(builder, tilableOp, loopBounds, 0, ivs);
}

/// Pattern rewriter hook to lower a `TiledOpInterface` to loops.
namespace {
struct TiledOpInterfaceLowerToLoopsPattern : public RewritePattern {
  TiledOpInterfaceLowerToLoopsPattern(MLIRContext *context,
                                      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto tilableOp = dyn_cast<TiledOpInterface>(op);
    if (!tilableOp) {
      return failure();
    }
    if (llvm::any_of(tilableOp->getResults(),
                     [&](Value v) { return v.getType().isa<ShapedType>(); })) {
      return rewriter.notifyMatchFailure(
          tilableOp, "lower to loops needs to have tensor semantics");
    }
    if (failed(lowerToLoops(rewriter, tilableOp))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct LinalgExtToLoopsPass
    : public LinalgExtToLoopsBase<LinalgExtToLoopsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    mlir::arith::ArithmeticDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<TiledOpInterfaceLowerToLoopsPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
IREE::LinalgExt::createLinalgExtToLoopsPass() {
  return std::make_unique<LinalgExtToLoopsPass>();
}
