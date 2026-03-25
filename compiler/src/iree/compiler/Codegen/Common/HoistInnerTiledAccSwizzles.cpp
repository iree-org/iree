// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_HOISTINNERTILEDACCSWIZZLESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static bool isSwizzleOp(Operation *op) {
  return isa<vector::TransposeOp, vector::ShapeCastOp, vector::BroadcastOp>(op);
}

namespace {

struct WrapAccSwizzlesPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    auto loopLike = dyn_cast<LoopLikeOpInterface>(tiledOp->getParentOp());
    if (!loopLike || loopLike.getRegionIterArgs().empty()) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "not inside a loop with iter_args");
    }

    bool anyOutputMatched = false;
    for (int64_t outputIdx = 0, numOutputs = tiledOp.getOutputs().size();
         outputIdx < numOutputs; ++outputIdx) {
      Value accOperand = tiledOp.getOutputs()[outputIdx];

      SmallVector<Operation *> prefixOps;
      Value accRoot = accOperand;
      while (auto *defOp = accRoot.getDefiningOp()) {
        if (!defOp->hasOneUse() || !isSwizzleOp(defOp)) {
          break;
        }
        prefixOps.push_back(defOp);
        accRoot = defOp->getOperand(0);
      }

      SmallVector<Operation *> suffixOps;
      Value suffixEnd = tiledOp.getResult(outputIdx);
      while (suffixEnd.hasOneUse()) {
        Operation *user = *suffixEnd.getUsers().begin();
        if (!isSwizzleOp(user)) {
          break;
        }
        suffixOps.push_back(user);
        suffixEnd = user->getResult(0);
      }

      if (prefixOps.empty() || suffixOps.empty()) {
        continue;
      }
      if (!isa<BlockArgument>(accRoot)) {
        continue;
      }

      rewriter.setInsertionPoint(prefixOps.back());
      auto prefixHoist = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), "acc_swizzle_to", "acc_swizzle_from",
          accRoot, [&](OpBuilder &b, Location loc, ValueRange args) {
            Value v = args[0];
            for (auto *op : llvm::reverse(prefixOps)) {
              IRMapping mapping;
              mapping.map(op->getOperand(0), v);
              v = b.clone(*op, mapping)->getResult(0);
            }
            return SmallVector<Value>{v};
          });
      rewriter.replaceAllUsesWith(accOperand, prefixHoist.getResult(0));
      for (auto *op : prefixOps) {
        if (op->use_empty()) {
          rewriter.eraseOp(op);
        }
      }

      Value suffixInput = tiledOp.getResult(outputIdx);
      rewriter.setInsertionPointAfter(suffixOps.back());
      auto suffixHoist = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), "acc_swizzle_from", "acc_swizzle_to",
          TypeRange{suffixEnd.getType()}, suffixInput,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value v = args[0];
            for (auto *op : suffixOps) {
              IRMapping mapping;
              mapping.map(op->getOperand(0), v);
              v = b.clone(*op, mapping)->getResult(0);
            }
            return SmallVector<Value>{v};
          });
      rewriter.replaceAllUsesWith(suffixEnd, suffixHoist.getResult(0));
      for (auto *op : llvm::reverse(suffixOps)) {
        if (op->use_empty()) {
          rewriter.eraseOp(op);
        }
      }

      anyOutputMatched = true;
    }

    return success(anyOutputMatched);
  }
};

struct HoistInnerTiledAccSwizzlesPass final
    : impl::HoistInnerTiledAccSwizzlesPassBase<HoistInnerTiledAccSwizzlesPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<WrapAccSwizzlesPattern>(context);
    bool changed = false;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     GreedyRewriteConfig(), &changed))) {
      return signalPassFailure();
    }

    if (changed) {
      if (failed(IREE::Util::eliminateHoistableConversions(getOperation()))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
