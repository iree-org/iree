// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include <limits>

#define DEBUG_TYPE "iree-codegen-optimize-comparison-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_OPTIMIZECOMPARISONOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using P = arith::CmpIPredicate;

/// Normalize a comparison so the broadcast operand is on the RHS.
/// Returns the (possibly flipped) predicate, the vector operand, and the
/// broadcast source scalar. Returns std::nullopt if neither operand is a
/// broadcast of a scalar.
static std::optional<std::tuple<arith::CmpIPredicate, Value, Value>>
normalizeBroadcastOperands(arith::CmpIOp cmpOp) {
  Value lhs = cmpOp.getLhs();
  Value rhs = cmpOp.getRhs();
  arith::CmpIPredicate pred = cmpOp.getPredicate();

  // Try RHS broadcast first.
  if (auto bc = rhs.getDefiningOp<vector::BroadcastOp>()) {
    if (bc.getSource().getType().isIntOrIndex()) {
      return std::make_tuple(pred, lhs, bc.getSource());
    }
  }
  // Try LHS broadcast -- swap operands so broadcast is on the RHS.
  // Swapping operands flips the comparison direction:
  //   lt(a,b) = gt(b,a), le(a,b) = ge(b,a), etc.
  if (auto bc = lhs.getDefiningOp<vector::BroadcastOp>()) {
    if (bc.getSource().getType().isIntOrIndex()) {
      P swapped;
      switch (pred) {
      case P::eq:
        swapped = P::eq;
        break;
      case P::ne:
        swapped = P::ne;
        break;
      case P::slt:
        swapped = P::sgt;
        break;
      case P::sle:
        swapped = P::sge;
        break;
      case P::sgt:
        swapped = P::slt;
        break;
      case P::sge:
        swapped = P::sle;
        break;
      case P::ult:
        swapped = P::ugt;
        break;
      case P::ule:
        swapped = P::uge;
        break;
      case P::ugt:
        swapped = P::ult;
        break;
      case P::uge:
        swapped = P::ule;
        break;
      default:
        assert(false && "unexpected CmpIPredicate");
      }
      return std::make_tuple(swapped, rhs, bc.getSource());
    }
  }
  return std::nullopt;
}

/// Return true if the predicate is strict (< or >, not <= or >=).
static bool isStrictPredicate(arith::CmpIPredicate pred) {
  return pred == P::slt || pred == P::sgt || pred == P::ult || pred == P::ugt;
}

/// Return true if the predicate is a less-than variant (lt or le).
static bool isLessThanPredicate(arith::CmpIPredicate pred) {
  return pred == P::slt || pred == P::sle || pred == P::ult || pred == P::ule;
}

/// Try to simplify a vector cmpi to a scalar comparison broadcast using
/// divisibility analysis. When the broadcast scalar's divisibility guarantees
/// that the comparison result is the same for all vector lanes, rewrite to a
/// scalar comparison broadcast.
// TODO(Max191): Consider moving this into the upstream OptimizeIntArithmetic /
// int-range-optimizations pass if it ever runs after vector lowering (or after
// vector flattening/unrolling). Currently this must run post-vector-lowering
// to see the 1-D mask computations.
static void simplifyDivisibleCmpI(IRRewriter &rewriter, arith::CmpIOp cmpOp,
                                  DataFlowSolver &solver) {
  auto resultType = dyn_cast<VectorType>(cmpOp.getResult().getType());
  if (!resultType) {
    return;
  }

  auto normalized = normalizeBroadcastOperands(cmpOp);
  if (!normalized) {
    return;
  }
  auto [pred, vectorVal, broadcastSource] = *normalized;

  // Get range of the vector operand (signed — divisibility analysis is
  // signed, and eq/ne are signedness-agnostic).
  auto *vecRangeLattice =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(vectorVal);
  if (!vecRangeLattice || vecRangeLattice->getValue().isUninitialized()) {
    return;
  }
  const ConstantIntRanges &vecRange = vecRangeLattice->getValue().getValue();

  auto *divLattice = solver.lookupState<IREE::Util::IntegerDivisibilityLattice>(
      broadcastSource);
  if (!divLattice || divLattice->getValue().isUninitialized()) {
    return;
  }
  int64_t sdiv = divLattice->getValue().getValue().sdiv();
  if (sdiv <= 0) {
    return;
  }

  int64_t vecMinS = vecRange.smin().getSExtValue();
  int64_t vecMaxS = vecRange.smax().getSExtValue();

  LDBG() << "Divisibility cmpi analysis (pred="
         << arith::stringifyCmpIPredicate(pred) << "):";
  LDBG() << "  scalar sdiv=" << sdiv << " vecRange=[" << vecMinS << ", "
         << vecMaxS << "]";

  auto floorDiv = [](int64_t a, int64_t d) -> int64_t {
    int64_t q = a / d;
    // Adjust for negative values: if remainder is nonzero and a is negative,
    // C++ truncation went toward zero but we need toward -inf.
    if (a % d != 0 && a < 0) {
      q -= 1;
    }
    return q;
  };

  // eq/ne: if no multiple of sdiv falls within [vecMin, vecMax], then no
  // vector element can ever equal the scalar, so eq is always false and ne
  // is always true.
  if (pred == P::eq || pred == P::ne) {
    // Guard: vecMinS - 1 must not underflow.
    if (vecMinS == std::numeric_limits<int64_t>::min()) {
      return;
    }
    // "No multiple in [vecMin, vecMax]" iff floor((vecMin-1)/sdiv) ==
    // floor(vecMax/sdiv).
    if (floorDiv(vecMinS - 1, sdiv) != floorDiv(vecMaxS, sdiv)) {
      return;
    }
    LDBG() << "  -> No multiple of sdiv in vector range. Folding to constant.";
    Location loc = cmpOp.getLoc();
    rewriter.setInsertionPoint(cmpOp);
    bool value = (pred == P::ne);
    auto constAttr = DenseElementsAttr::get(resultType, value);
    auto constOp = arith::ConstantOp::create(rewriter, loc, constAttr);
    rewriter.replaceOp(cmpOp, constOp);
    return;
  }

  // The divisibility analysis (sdiv) is only valid for signed integers —
  // unsigned overflow/underflow breaks the modular arithmetic assumptions.
  bool isUnsigned =
      (pred == P::ult || pred == P::ule || pred == P::ugt || pred == P::uge);
  if (isUnsigned) {
    return;
  }

  bool isLessThan = isLessThanPredicate(pred);
  bool isStrict = isStrictPredicate(pred);

  // Check that the vector range lies within a single bucket of size sdiv.
  // The condition is that the vector's full range does not span any multiple
  // of sdiv that would cause different vector elements to fall on different
  // sides of the comparison. The exact interval to check depends on whether
  // the predicate is strict:
  //   slt/sge (strict == lessThan): no multiple of sdiv in [vecMin+1, vecMax]
  //     → floor(vecMin / sdiv) == floor(vecMax / sdiv)
  //   sle/sgt (strict != lessThan): no multiple of sdiv in [vecMin, vecMax-1]
  //     → floor((vecMin-1) / sdiv) == floor((vecMax-1) / sdiv)
  bool needsAdjust = (isStrict != isLessThan);
  if (needsAdjust) {
    // Guard: vecMinS - 1 must not underflow.
    if (vecMinS == std::numeric_limits<int64_t>::min()) {
      return;
    }
    if (floorDiv(vecMinS - 1, sdiv) != floorDiv(vecMaxS - 1, sdiv)) {
      LDBG() << "  -> Vector range spans multiple buckets. Skipping.";
      return;
    }
  } else {
    if (floorDiv(vecMinS, sdiv) != floorDiv(vecMaxS, sdiv)) {
      LDBG() << "  -> Vector range spans multiple buckets. Skipping.";
      return;
    }
  }

  // The threshold distinguishes all-true from all-false:
  //   slt/sge: all-true boundary at vecMax+1 (strict comparison on vec)
  //   sle/sgt: all-true boundary at vecMax (non-strict includes vecMax)
  int64_t threshold;
  if (needsAdjust) {
    threshold = vecMaxS;
  } else {
    if (vecMaxS == std::numeric_limits<int64_t>::max()) {
      return;
    }
    threshold = vecMaxS + 1;
  }

  LDBG() << "  threshold=" << threshold;
  LDBG() << "  -> Uniform. Rewriting to scalar comparison.";

  Location loc = cmpOp.getLoc();
  rewriter.setInsertionPoint(cmpOp);

  // Emitted predicate:
  //   slt/sle (lessThan): broadcast(scalar sge threshold)
  //   sge/sgt (not lessThan): broadcast(scalar slt threshold)
  arith::CmpIPredicate scalarPred =
      isLessThan ? arith::CmpIPredicate::sge : arith::CmpIPredicate::slt;

  Value thresholdVal;
  if (broadcastSource.getType().isIndex()) {
    thresholdVal = arith::ConstantIndexOp::create(rewriter, loc, threshold);
  } else {
    thresholdVal = arith::ConstantIntOp::create(
        rewriter, loc, threshold,
        broadcastSource.getType().getIntOrFloatBitWidth());
  }

  Value scalarCmp = arith::CmpIOp::create(rewriter, loc, scalarPred,
                                          broadcastSource, thresholdVal);
  auto broadcast =
      vector::BroadcastOp::create(rewriter, loc, resultType, scalarCmp);
  rewriter.replaceOp(cmpOp, broadcast);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct OptimizeComparisonOpsPass final
    : impl::OptimizeComparisonOpsPassBase<OptimizeComparisonOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    // Setup dataflow analyses.
    DataFlowSolver solver;
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    solver.load<IREE::Util::IntegerDivisibilityAnalysis>();
    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    IRRewriter rewriter(context);

    // Collect cmpi ops first since simplification may replace them.
    SmallVector<arith::CmpIOp> cmpOps;
    funcOp.walk([&](arith::CmpIOp op) { cmpOps.push_back(op); });
    for (arith::CmpIOp cmpOp : cmpOps) {
      simplifyDivisibleCmpI(rewriter, cmpOp, solver);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
