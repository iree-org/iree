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
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_HOISTINNERTILEDACCRESHAPESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

// Look for operations that reshape vectors to or from the form needed by
// intrinsics, which are hard to hoist from loops up in vector distribute as
// currently architected.
static bool isReshapeOp(Operation *op) {
  return isa<vector::TransposeOp, vector::ShapeCastOp, vector::BroadcastOp>(op);
}

static constexpr llvm::StringLiteral kAccReshapeTo = "acc_reshape_to_intrinsic";
static constexpr llvm::StringLiteral kAccReshapeFrom =
    "acc_reshape_from_intrinsic";

namespace {

struct WrapAccReshapesPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using Base = OpRewritePattern<IREE::Codegen::InnerTiledOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    auto loopLike = dyn_cast<LoopLikeOpInterface>(tiledOp->getParentOp());
    if (!loopLike || loopLike.getRegionIterArgs().empty()) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "not inside a loop with iter_args");
    }

    bool anyOutputMatched = false;
    for (size_t outputIdx = 0, numOutputs = tiledOp.getOutputs().size();
         outputIdx < numOutputs; ++outputIdx) {
      Value accOperand = tiledOp.getOutputs()[outputIdx];

      SmallVector<Operation *> prefixOps;
      Value accRoot = accOperand;
      while (auto *defOp = accRoot.getDefiningOp()) {
        if (!defOp->hasOneUse() || !isReshapeOp(defOp)) {
          break;
        }
        prefixOps.push_back(defOp);
        accRoot = defOp->getOperand(0);
      }

      SmallVector<Operation *> suffixOps;
      Value suffixEnd = tiledOp.getResult(outputIdx);
      while (suffixEnd.hasOneUse()) {
        Operation *user = *suffixEnd.getUsers().begin();
        if (!isReshapeOp(user)) {
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
      // Wrap the prefix reshapes (iter_arg -> inner_tiled accumulator shape)
      // in a hoistable_conversion so the pair can be hoisted out of the loop.
      auto prefixHoist = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), /*tag=*/kAccReshapeTo,
          /*inverseTag=*/kAccReshapeFrom, accRoot,
          [&](OpBuilder &b, Location loc, ValueRange args) {
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
      // Wrap the suffix reshapes (inner_tiled result -> iter_arg shape)
      // as the inverse conversion.
      auto suffixHoist = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), /*tag=*/kAccReshapeFrom,
          /*inverseTag=*/kAccReshapeTo, TypeRange{suffixEnd.getType()},
          suffixInput, [&](OpBuilder &b, Location loc, ValueRange args) {
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

struct HoistInnerTiledAccReshapesPass final
    : impl::HoistInnerTiledAccReshapesPassBase<HoistInnerTiledAccReshapesPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<WrapAccReshapesPattern>(context);
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
