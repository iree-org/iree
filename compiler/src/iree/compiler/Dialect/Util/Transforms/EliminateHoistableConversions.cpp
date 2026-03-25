// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Util {

static constexpr StringLiteral kWasHoistedAttr = "iree.util.was_hoisted";

namespace {

// Checks that F and G are marked as mutual inverses so that we can cancel or
// hoist them.
static bool areInversePairs(HoistableConversionOp f, HoistableConversionOp g) {
  return f.getTag() == g.getInverseTag() && f.getInverseTag() == g.getTag();
}

/// Returns true if the conversion could plausibly have been hoisted: it's
/// inside a loop with iter_args and at least one of its inputs traces back
/// to a block argument through a chain of single-operand ops.
static bool isPlausiblyHoistable(HoistableConversionOp hc) {
  auto loopParent = dyn_cast_if_present<LoopLikeOpInterface>(hc->getParentOp());
  if (!loopParent || loopParent.getRegionIterArgs().empty()) {
    return false;
  }
  for (Value input : hc.getInputs()) {
    Value v = input;
    while (auto *defOp = v.getDefiningOp()) {
      if (defOp->getNumOperands() != 1) {
        break;
      }
      v = defOp->getOperand(0);
    }
    if (isa<BlockArgument>(v)) {
      return true;
    }
  }
  return false;
}

/// Clones a HoistableConversionOp with new inputs, marking the clone as
/// hoisted. Returns the cloned op's results.
static ValueRange cloneConversionOp(OpBuilder &builder,
                                    HoistableConversionOp op,
                                    ValueRange newInputs) {
  IRMapping mapping;
  for (auto [oldInput, newInput] : llvm::zip_equal(op.getInputs(), newInputs)) {
    mapping.map(oldInput, newInput);
  }
  auto *cloned = builder.clone(*op.getOperation(), mapping);
  cloned->setDiscardableAttr(kWasHoistedAttr, builder.getUnitAttr());
  return cloned->getResults();
}

/// Inlines the body of a HoistableConversionOp into its parent block,
/// replacing the op with its body's return values.
static void inlineConversionBody(RewriterBase &rewriter,
                                 HoistableConversionOp op) {
  Block &body = op.getBody().front();
  auto returnOp = cast<ReturnOp>(body.getTerminator());

  for (auto [arg, input] :
       llvm::zip_equal(body.getArguments(), op.getInputs())) {
    rewriter.replaceAllUsesWith(arg, input);
  }

  SmallVector<Value> returnValues(returnOp.getOperands());
  rewriter.eraseOp(returnOp);

  Block *parentBlock = op->getBlock();
  parentBlock->getOperations().splice(op->getIterator(), body.getOperations());

  rewriter.replaceOp(op, returnValues);
}

//===----------------------------------------------------------------------===//
// Directly cancelling hoistable conversions
//===----------------------------------------------------------------------===//

struct CancelInversePairPattern : OpRewritePattern<HoistableConversionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistableConversionOp g,
                                PatternRewriter &rewriter) const override {
    if (g.getInputs().empty()) {
      return rewriter.notifyMatchFailure(g, "no inputs to cancel with");
    }

    auto f = g.getInputs()[0].getDefiningOp<HoistableConversionOp>();
    if (!f) {
      return rewriter.notifyMatchFailure(g, "couldn't find potential inverse");
    }

    if (!areInversePairs(f, g)) {
      return rewriter.notifyMatchFailure(
          g, "potential inverse doesn't have correct tag");
    }

    if (f.getResults().size() != g.getInputs().size()) {
      return rewriter.notifyMatchFailure(g,
                                         "doesn't have sane number of inputs "
                                         "as potential inverse has outputs");
    }
    for (auto [fResult, gInput] :
         llvm::zip_equal(f.getResults(), g.getInputs())) {
      if (fResult != gInput) {
        return rewriter.notifyMatchFailure(
            g, "some input doesn't line up with potential inverses");
      }
    }

    if (g.getResults().size() != f.getInputs().size()) {
      return rewriter.notifyMatchFailure(
          g, "G's results don't match F's inputs in size");
    }
    for (auto [gResult, fInput] :
         llvm::zip_equal(g.getResults(), f.getInputs())) {
      if (gResult.getType() != fInput.getType()) {
        return rewriter.notifyMatchFailure(
            g, "type mismatch for replacement results");
      }
    }

    rewriter.replaceOp(g, f.getInputs());
    if (f->use_empty()) {
      rewriter.eraseOp(f);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Hoisting conversions at loop edges
//===----------------------------------------------------------------------===//

/// Matches a HoistableConversionOp G whose results are yielded back to loop
/// iter_args, finds the matching F that consumes those iter_args, and hoists
/// both conversions out of the loop. F may consume a subset of the loop's
/// iter_args (partial hoisting).
struct HoistConversionFromLoopPattern
    : OpRewritePattern<HoistableConversionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HoistableConversionOp g,
                                PatternRewriter &rewriter) const override {
    auto loopLike = dyn_cast_if_present<LoopLikeOpInterface>(g->getParentOp());
    if (!loopLike) {
      return rewriter.notifyMatchFailure(g, "not directly in a loop");
    }

    auto yieldedMutable = loopLike.getYieldedValuesMutable();
    if (!yieldedMutable || yieldedMutable->empty()) {
      return rewriter.notifyMatchFailure(g,
                                         "enclosing loop doesn't yield values");
    }

    auto allIterArgs = loopLike.getRegionIterArgs();
    if (allIterArgs.empty()) {
      return rewriter.notifyMatchFailure(g, "loop has no iter args");
    }

    MutableArrayRef<OpOperand> yieldOperands = *yieldedMutable;

    // Find which yield operands correspond to G's results.
    // Note: this is O(N^2), but we expect the numbers are small enough in
    // practice that it should be faster than constructing a map.
    SmallVector<unsigned> yieldIndices;
    for (unsigned i = 0, e = yieldOperands.size(); i < e; ++i) {
      for (unsigned ri = 0, re = g.getResults().size(); ri < re; ++ri) {
        if (yieldOperands[i].get() == g.getResult(ri)) {
          yieldIndices.push_back(i);
          break;
        }
      }
    }
    if (yieldIndices.size() != g.getResults().size()) {
      return rewriter.notifyMatchFailure(
          g, "not all results are yielded to loop iter_args");
    }

    for (unsigned ri = 0, re = g.getResults().size(); ri < re; ++ri) {
      if (yieldOperands[yieldIndices[ri]].get() != g.getResult(ri)) {
        return rewriter.notifyMatchFailure(
            g, "yield operand ordering doesn't match result ordering");
      }
    }

    SmallVector<BlockArgument> iterArgs;
    for (unsigned yi : yieldIndices) {
      iterArgs.push_back(allIterArgs[yi]);
    }

    HoistableConversionOp f = nullptr;
    for (BlockArgument iterArg : iterArgs) {
      for (Operation *user : iterArg.getUsers()) {
        auto candidate = dyn_cast<HoistableConversionOp>(user);
        if (!candidate) {
          continue;
        }
        if (!f) {
          f = candidate;
        } else if (f != candidate) {
          return rewriter.notifyMatchFailure(
              g, "multiple hoistable_conversion ops consume the iter_args");
        }
      }
      if (!f) {
        return rewriter.notifyMatchFailure(
            g, "no hoistable_conversion consumes the iter_args");
      }
    }

    if (f.getInputs().size() != iterArgs.size()) {
      return rewriter.notifyMatchFailure(
          g, "F doesn't consume exactly the matched iter_args");
    }
    for (auto [fInput, iterArg] : llvm::zip_equal(f.getInputs(), iterArgs)) {
      if (fInput != iterArg) {
        return rewriter.notifyMatchFailure(
            g, "F's inputs don't match the iter_args in order");
      }
    }

    if (!areInversePairs(f, g)) {
      return rewriter.notifyMatchFailure(g, "F and G are not inverse pairs");
    }

    // Pre-conditions are good: actually hoist.
    unsigned numOldIterArgs = allIterArgs.size();

    auto initsMutable = loopLike.getInitsMutable();
    SmallVector<Value> oldInits = llvm::map_to_vector(
        yieldIndices, [&](unsigned yi) { return initsMutable[yi].get(); });

    // Clone F before the loop to transform initial values.
    rewriter.setInsertionPoint(loopLike);
    ValueRange newInitValues = cloneConversionOp(rewriter, f, oldInits);

    bool fHasOtherUses = llvm::any_of(iterArgs, [&](BlockArgument iterArg) {
      return llvm::any_of(iterArg.getUses(), [&](OpOperand &use) {
        return use.getOwner() != f.getOperation();
      });
    });

    NewYieldValuesFn yieldFn =
        [&](OpBuilder &, Location,
            ArrayRef<BlockArgument>) -> SmallVector<Value> {
      return llvm::to_vector(g.getInputs());
    };

    auto maybeNewLoop = loopLike.replaceWithAdditionalYields(
        rewriter, newInitValues,
        /*replaceInitOperandUsesInLoop=*/false, yieldFn);
    if (failed(maybeNewLoop)) {
      return failure();
    }

    // replaceWithAdditionalYields erases the old loop; f and g are now in
    // the new loop body.
    LoopLikeOpInterface newLoop = *maybeNewLoop;
    auto allNewLoopIterArgs = newLoop.getRegionIterArgs();

    auto newIterArgs =
        allNewLoopIterArgs.slice(numOldIterArgs, newInitValues.size());

    if (fHasOtherUses) {
      rewriter.setInsertionPoint(f);
      SmallVector<Value> convertedBack =
          cloneConversionOp(rewriter, g, newIterArgs);
      SmallVector<BlockArgument> oldIterArgsInNewLoop;
      for (unsigned yi : yieldIndices) {
        oldIterArgsInNewLoop.push_back(allNewLoopIterArgs[yi]);
      }
      for (auto [oldArg, converted] :
           llvm::zip_equal(oldIterArgsInNewLoop, convertedBack)) {
        rewriter.replaceAllUsesExcept(oldArg, converted, f);
      }
    }
    rewriter.replaceOp(f, newIterArgs);

    auto loopResults = newLoop->getResults();
    auto newLoopResults =
        loopResults.slice(numOldIterArgs, newInitValues.size());

    // Clone G after the loop to convert results back.
    rewriter.setInsertionPointAfter(newLoop);
    SmallVector<Value> postGResults =
        cloneConversionOp(rewriter, g, newLoopResults);

    for (auto [idx, yi] : llvm::enumerate(yieldIndices)) {
      rewriter.replaceAllUsesWith(loopResults[yi], postGResults[idx]);
    }

    // We've hoisted the conversion, snap the old iter_arg to make it trivially
    // dead.
    MutableArrayRef<OpOperand> newYieldedMutable =
        *newLoop.getYieldedValuesMutable();
    for (unsigned yi : yieldIndices) {
      newYieldedMutable[yi].assign(allNewLoopIterArgs[yi]);
    }

    if (g->use_empty()) {
      rewriter.eraseOp(g);
    }

    return success();
  }
};

} // namespace

LogicalResult eliminateHoistableConversions(Operation *op) {
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  patterns.add<HoistConversionFromLoopPattern>(context, /*benefit=*/2);
  patterns.add<CancelInversePairPattern>(context, /*benefit=*/1);
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    return failure();
  }

  SmallVector<HoistableConversionOp> toInline;
  op->walk([&](HoistableConversionOp hc) { toInline.push_back(hc); });
  IRRewriter rewriter(context);
  for (auto hc : toInline) {
    if (!hc->hasAttr(kWasHoistedAttr) && isPlausiblyHoistable(hc)) {
      hc->emitRemark(
          "hoistable_conversion was not hoisted or cancelled; inlining in "
          "place")
          << " " << hc.getTag() << " inverting " << hc.getInverseTag();
    }
    rewriter.setInsertionPoint(hc);
    inlineConversionBody(rewriter, hc);
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::Util
