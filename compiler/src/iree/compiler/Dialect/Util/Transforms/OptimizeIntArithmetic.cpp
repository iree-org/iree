// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-util-optimize-int-arithmetic"
using llvm::dbgs;

using namespace mlir::dataflow;

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_OPTIMIZEINTARITHMETICPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

// An index_cast from i64 to index is a no-op on targets where index is
// 64 bits. But on targets where index is 32bits, it is a truncate. On these
// platforms, demoting to an index is only conservatively correct if all
// operands and all results are within the unsigned 32bit bounds.
// While there is a good chance that such arithmetic that exceeds these
// bounds is simply wrong/overflow-ridden, we opt to do no harm and preseve
// the exact results. This optimization is targeted at "small" sequences
// anyway and this catches everything known to exist. If needed, this rule
// could be dropped if it is ever appropriate to unconditionally assume
// 64bit semantics.
static constexpr uint64_t SAFE_INDEX_UNSIGNED_MAX_VALUE =
    std::numeric_limits<uint32_t>::max();

//===----------------------------------------------------------------------===//
// Signed -> Unsigned patterns
// Note that there is an upstream UnsignedWhenEquivalent pass but it uses
// DialectConversion and legality vs simple patterns, so we cannot use it.
// Some support code has been adapted from that pass, though.
//===----------------------------------------------------------------------===//

/// Succeeds when a value is statically non-negative in that it has a lower
/// bound on its value (if it is treated as signed) and that bound is
/// non-negative.
static bool staticallyLegalToConvertToUnsigned(DataFlowSolver &solver,
                                               Value v) {
  auto *result = solver.lookupState<IntegerValueRangeLattice>(v);
  if (!result || result->getValue().isUninitialized()) {
    return false;
  }
  const ConstantIntRanges &range = result->getValue().getValue();
  bool isNonNegative = range.smin().isNonNegative();
  Type type = v.getType();
  if (isa<IndexType>(type)) {
    bool canSafelyTruncate =
        range.umin().getZExtValue() <= SAFE_INDEX_UNSIGNED_MAX_VALUE &&
        range.umax().getZExtValue() <= SAFE_INDEX_UNSIGNED_MAX_VALUE;
    return isNonNegative && canSafelyTruncate;
  } else {
    return isNonNegative;
  }
}

/// Succeeds if an op can be converted to its unsigned equivalent without
/// changing its semantics. This is the case when none of its openands or
/// results can be below 0 when analyzed from a signed perspective.
static LogicalResult
staticallyLegalToConvertToUnsignedOp(DataFlowSolver &solver, Operation *op) {
  auto nonNegativePred = [&solver](Value v) -> bool {
    bool isNonNegative = staticallyLegalToConvertToUnsigned(solver, v);
    return isNonNegative;
  };
  return success(llvm::all_of(op->getOperands(), nonNegativePred) &&
                 llvm::all_of(op->getResults(), nonNegativePred));
}

template <typename Signed, typename Unsigned>
struct ConvertOpToUnsigned : public OpRewritePattern<Signed> {
  ConvertOpToUnsigned(MLIRContext *context, DataFlowSolver &solver)
      : OpRewritePattern<Signed>(context), solver(solver) {}

  LogicalResult matchAndRewrite(Signed op,
                                PatternRewriter &rewriter) const override {
    if (failed(staticallyLegalToConvertToUnsignedOp(solver, op)))
      return failure();
    rewriter.replaceOpWithNewOp<Unsigned>(op, op->getResultTypes(),
                                          op->getOperands(), op->getAttrs());
    return success();
  }

  DataFlowSolver &solver;
};

//===----------------------------------------------------------------------===//
// Int64 -> unsigned index demotion
// Torch does a lot of indexy manipulation using scalar i64 ops. We undo these
// here and treat them as index when safe to do so. Since the casts can block
// optimizations, it can be useful to eliminate them when possible.
//===----------------------------------------------------------------------===//

// Matches IR like:
//   %5 = arith.addi %0, %1 : int64
//   %6 = arith.index_castui %5 : int64 to index
//
// And moves the index_castui to the producer's operands:
//   %3 = arith.index_castui %0 : int64 to index
//   %4 = arith.index_castui %1 : int64 to index
//   %5 = arith.addi %3, %4 : index
//
struct ConvertUnsignedI64IndexCastProducerToIndex
    : public OpRewritePattern<arith::IndexCastUIOp> {
  ConvertUnsignedI64IndexCastProducerToIndex(MLIRContext *context,
                                             DataFlowSolver &solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(arith::IndexCastUIOp origIndexOp,
                                PatternRewriter &rewriter) const override {
    Type inType = origIndexOp.getIn().getType();
    Type outType = origIndexOp.getOut().getType();
    if (!inType.isSignlessInteger(64) || !isa<IndexType>(outType))
      return failure();

    Operation *producer = origIndexOp.getIn().getDefiningOp();
    if (!producer)
      return failure();
    auto producerResult = producer->getResult(0);
    if (!producerResult.hasOneUse())
      return failure();

    auto pred = [&](Value v) -> bool {
      auto *result = solver.lookupState<IntegerValueRangeLattice>(v);
      if (!result || result->getValue().isUninitialized()) {
        return false;
      }
      const ConstantIntRanges &range = result->getValue().getValue();
      bool isInBounds =
          range.umin().getZExtValue() <= SAFE_INDEX_UNSIGNED_MAX_VALUE &&
          range.umax().getZExtValue() <= SAFE_INDEX_UNSIGNED_MAX_VALUE;
      return isInBounds;
    };
    auto isOpStaticallyLegal = [&](Operation *op) -> bool {
      return llvm::all_of(op->getOperands(), pred) &&
             llvm::all_of(op->getResults(), pred);
    };

    if (!isa_and_present<arith::AddIOp, arith::CeilDivUIOp, arith::DivUIOp,
                         arith::MaxUIOp, arith::MinUIOp, arith::MulIOp,
                         arith::RemUIOp, arith::SubIOp>(producer))
      return failure();
    if (!isOpStaticallyLegal(producer))
      return failure();

    // Make modifications.
    rewriter.modifyOpInPlace(producer, [&]() {
      rewriter.setInsertionPoint(producer);
      for (auto &operand : producer->getOpOperands()) {
        if (operand.get().getType() != inType)
          continue;
        Value newOperand = rewriter.create<arith::IndexCastUIOp>(
            producer->getLoc(), outType, operand.get());
        operand.set(newOperand);
      }
      producer->getResult(0).setType(outType);
    });
    origIndexOp.getOut().replaceAllUsesWith(producer->getResult(0));
    rewriter.eraseOp(origIndexOp);

    return success();
  }

  DataFlowSolver &solver;
};

//===----------------------------------------------------------------------===//
// Index -> int32 assumption narrowing
// If we're narrowing `index` values to `i32`, a `util.assume.int` on `index`
// introduces unnecessary zero-extensions and truncations to/from `index`
// when introducing assumptions.
//===----------------------------------------------------------------------===//
struct RemoveIndexCastForAssumeOfI32
    : public OpRewritePattern<Util::AssumeIntOp> {
  RemoveIndexCastForAssumeOfI32(MLIRContext *context, DataFlowSolver &solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(Util::AssumeIntOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallBitVector needNarrowing(op.getNumOperands(), false);
    for (auto [idx, arg] : llvm::enumerate(op.getOperands())) {
      if (!arg.getType().isIndex())
        continue;
      auto castOp = arg.getDefiningOp<arith::IndexCastUIOp>();
      if (!castOp)
        continue;
      Value castIn = castOp.getIn();
      Type intType = castIn.getType();
      if (intType.getIntOrFloatBitWidth() > 32)
        continue;

      needNarrowing[idx] = true;
    }
    if (needNarrowing.none())
      return failure();

    SmallVector<Value> newArgs;
    newArgs.reserve(op.getNumOperands());
    for (auto [idx, arg] : llvm::enumerate(op.getOperands())) {
      if (!needNarrowing[idx]) {
        newArgs.push_back(arg);
        continue;
      }
      newArgs.push_back(arg.getDefiningOp<arith::IndexCastUIOp>().getIn());
    }
    ArrayAttr assumptions = op.getAssumptionsAttr();
    auto newOp = rewriter.create<Util::AssumeIntOp>(
        op.getLoc(), ValueTypeRange<ArrayRef<Value>>{newArgs}, newArgs,
        assumptions);
    SmallVector<Value> replacements(newOp.getResults());
    for (auto [newRes, oldRes] :
         llvm::zip_equal(replacements, op.getResults())) {
      if (newRes.getType() != oldRes.getType()) {
        newRes = rewriter.create<arith::IndexCastUIOp>(
            op.getLoc(), oldRes.getType(), newRes);
      }
      // Preserve assumption state.
      auto *oldState = solver.lookupState<IntegerValueRangeLattice>(oldRes);
      if (oldState) {
        (void)solver.getOrCreateState<IntegerValueRangeLattice>(newRes)->join(
            *oldState);
      }
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }

  DataFlowSolver &solver;
};

//===----------------------------------------------------------------------===//
// scf.for induction variable range narrowing
// If the induction variable of an scf.for can be represented as an I32,
// make that change to save on registers etc.
//===----------------------------------------------------------------------===//
struct NarrowSCFForIvToI32 : public OpRewritePattern<scf::ForOp> {
  NarrowSCFForIvToI32(MLIRContext *context, DataFlowSolver &solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();
    Value iv = forOp.getInductionVar();
    Type srcType = iv.getType();
    if (!srcType.isIndex() && !srcType.isInteger(64))
      return rewriter.notifyMatchFailure(forOp, "IV isn't an index or i64");
    if (!staticallyLegalToConvertToUnsigned(solver, iv))
      return rewriter.notifyMatchFailure(forOp, "IV isn't non-negative");
    if (!staticallyLegalToConvertToUnsigned(solver, forOp.getStep()))
      return rewriter.notifyMatchFailure(forOp, "Step isn't non-negative");
    auto *ivState = solver.lookupState<IntegerValueRangeLattice>(iv);
    if (ivState->getValue().getValue().smax().getActiveBits() > 31)
      return rewriter.notifyMatchFailure(forOp, "IV won't fit in signed int32");

    Type i32 = rewriter.getI32Type();
    auto doCastDown = [&](Value v) -> Value {
      if (srcType.isIndex())
        return rewriter.create<arith::IndexCastUIOp>(loc, i32, v);
      else
        return rewriter.create<arith::TruncIOp>(loc, i32, v);
    };
    Value newLb = doCastDown(forOp.getLowerBound());
    Value newUb = doCastDown(forOp.getUpperBound());
    Value newStep = doCastDown(forOp.getStep());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&forOp.getRegion().front());
      Value castBackOp;
      if (srcType.isIndex()) {
        castBackOp =
            rewriter.create<arith::IndexCastUIOp>(iv.getLoc(), srcType, iv);
      } else {
        castBackOp = rewriter.create<arith::ExtUIOp>(iv.getLoc(), srcType, iv);
      }
      (void)solver.getOrCreateState<IntegerValueRangeLattice>(castBackOp)
          ->join(*ivState);
      rewriter.replaceAllUsesExcept(iv, castBackOp, castBackOp.getDefiningOp());
    }
    solver.eraseState(iv);
    rewriter.modifyOpInPlace(forOp, [&]() {
      iv.setType(i32);
      forOp.getLowerBoundMutable().assign(newLb);
      forOp.getUpperBoundMutable().assign(newUb);
      forOp.getStepMutable().assign(newStep);
    });
    return success();
  }

  DataFlowSolver &solver;
};

//===----------------------------------------------------------------------===//
// Divisibility
//===----------------------------------------------------------------------===//

static LogicalResult getDivisibility(DataFlowSolver &solver, Operation *op,
                                     Value value, PatternRewriter &rewriter,
                                     ConstantIntDivisibility &out) {
  auto *div = solver.lookupState<IntegerDivisibilityLattice>(value);
  if (!div || div->getValue().isUninitialized())
    return rewriter.notifyMatchFailure(op,
                                       "divisibility could not be determined");

  out = div->getValue().getValue();
  LLVM_DEBUG(dbgs() << "  * Resolved divisibility: " << out << "\n");
  return success();
}

struct RemUIDivisibilityByConstant : public OpRewritePattern<arith::RemUIOp> {
  RemUIDivisibilityByConstant(MLIRContext *context, DataFlowSolver &solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(arith::RemUIOp op,
                                PatternRewriter &rewriter) const override {
    APInt rhsConstant;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhsConstant)))
      return rewriter.notifyMatchFailure(op, "rhs is not constant");

    ConstantIntDivisibility lhsDiv;
    if (failed(getDivisibility(solver, op, op.getLhs(), rewriter, lhsDiv)))
      return failure();

    uint64_t rhsValue = rhsConstant.getZExtValue();
    if (rhsValue > 0 && lhsDiv.udiv() > 0) {
      if (lhsDiv.udiv() % rhsValue != 0)
        return rewriter.notifyMatchFailure(op, "rhs does not divide lhs");

      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, rewriter.getZeroAttr(op.getResult().getType()));
      return success();
    }

    return failure();
  }

  DataFlowSolver &solver;
};

//===----------------------------------------------------------------------===//
// Affine expansion
// affine.apply expansion can fail after producing a lot of IR. Since this is
// a bad thing to be doing as part of our overall iteration, we do it as a
// preprocessing walk. This also lets it be well behaved with respect to
// error messaging, etc. We will likely replace this with a more integrated
// version at some point which can use the bounds analysis to avoid corners
// of the original.
//===----------------------------------------------------------------------===//

void expandAffineOps(Operation *rootOp) {
  IRRewriter rewriter(rootOp->getContext());
  rootOp->walk([&](affine::AffineApplyOp op) {
    LLVM_DEBUG(dbgs() << "** Expand affine.apply: " << op << "\n");
    rewriter.setInsertionPoint(op);
    auto maybeExpanded =
        mlir::affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                      llvm::to_vector<4>(op.getOperands()));
    if (!maybeExpanded) {
      LLVM_DEBUG(dbgs() << "** ERROR: Failed to expand affine.apply\n");
      return;
    }
    rewriter.replaceOp(op, *maybeExpanded);
  });
}

//===----------------------------------------------------------------------===//
// General optimization patterns
//===----------------------------------------------------------------------===//

struct ElideTruncOfIndexCast : public OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp truncOp,
                                PatternRewriter &rewriter) const override {
    Operation *producer = truncOp.getOperand().getDefiningOp();
    if (!producer)
      return failure();
    if (!isa<arith::IndexCastOp, arith::IndexCastUIOp>(producer))
      return failure();
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(
        truncOp, truncOp.getResult().getType(), producer->getOperand(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass setup
//===----------------------------------------------------------------------===//

class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  void notifyOperationModified(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults()) {
      s.eraseState(res);
    }
  }

  DataFlowSolver &s;
};

class OptimizeIntArithmeticPass
    : public impl::OptimizeIntArithmeticPassBase<OptimizeIntArithmeticPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    expandAffineOps(op);

    DataFlowSolver solver;
    // Needed to make the dead code analyis not be too conservative.
    solver.load<SparseConstantPropagation>();
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    solver.load<IntegerDivisibilityAnalysis>();
    DataFlowListener listener(solver);
    RewritePatternSet patterns(ctx);

    // Populate upstream arith patterns.
    arith::populateIntRangeOptimizationsPatterns(patterns, solver);

    if (narrowToI32) {
      arith::populateIntRangeNarrowingPatterns(patterns, solver, {32});
      patterns.add<NarrowSCFForIvToI32, RemoveIndexCastForAssumeOfI32>(ctx,
                                                                       solver);
    }

    // Populate canonicalization patterns.
    auto arithDialect = ctx->getOrLoadDialect<arith::ArithDialect>();
    for (const RegisteredOperationName &name : ctx->getRegisteredOperations()) {
      if (&name.getDialect() == arithDialect)
        name.getCanonicalizationPatterns(patterns, ctx);
    }

    // General optimization patterns.
    patterns.add<ElideTruncOfIndexCast>(ctx);

    // Populate unsigned conversion patterns.
    patterns.add<ConvertUnsignedI64IndexCastProducerToIndex,
                 ConvertOpToUnsigned<arith::CeilDivSIOp, arith::CeilDivUIOp>,
                 ConvertOpToUnsigned<arith::DivSIOp, arith::DivUIOp>,
                 ConvertOpToUnsigned<arith::FloorDivSIOp, arith::DivUIOp>,
                 ConvertOpToUnsigned<arith::IndexCastOp, arith::IndexCastUIOp>,
                 ConvertOpToUnsigned<arith::RemSIOp, arith::RemUIOp>,
                 ConvertOpToUnsigned<arith::MinSIOp, arith::MinUIOp>,
                 ConvertOpToUnsigned<arith::MaxSIOp, arith::MaxUIOp>,
                 ConvertOpToUnsigned<arith::ExtSIOp, arith::ExtUIOp>>(ctx,
                                                                      solver);

    // Populate divisibility patterns.
    patterns.add<RemUIDivisibilityByConstant>(ctx, solver);

    GreedyRewriteConfig config;
    // Results in fewer recursive data flow flushes/cycles on modification.
    config.useTopDownTraversal = false;
    config.listener = &listener;

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    for (int i = 0;; ++i) {
      LLVM_DEBUG(dbgs() << "  * Starting iteration: " << i << "\n");
      if (failed(solver.initializeAndRun(op))) {
        emitError(op->getLoc()) << "failed to perform int range analysis";
        return signalPassFailure();
      }

      LLVM_DEBUG(
          dbgs() << "  * Finished Running Solver -- Applying Patterns\n");
      bool changed = false;
      if (failed(applyPatternsGreedily(op, frozenPatterns, config, &changed))) {
        emitError(op->getLoc())
            << "int arithmetic optimization failed to converge on iteration "
            << i;
        return signalPassFailure();
      }

      if (!changed)
        break;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
