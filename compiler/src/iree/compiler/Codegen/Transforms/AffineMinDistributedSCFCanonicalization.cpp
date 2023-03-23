// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AffineMinDistributedSCFCanonicalization.cpp ------------------------===//
//
// Implements Canonicalization of affine.min in presence of distributed loops
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

static bool isDivisible(Value v, int64_t dividend);

/// Return true if we can prove that affineMinOp result is positive and
/// divisible by the given |dividend|. This is true if all the the results of
/// the associated affine map are positive and divisible by |dividend|.
/// This speciically look for the following pattern:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
static bool affineMinOpDivisible(AffineMinOp minOp, int64_t dividend) {
  if (!minOp.getSymbolOperands().empty() ||
      minOp.getAffineMap().getNumResults() != 2) {
    return {};
  }
  Value iv;
  Value ub;
  Value lb;
  Value step;
  // Check if any of the dimensions is a ForOp or ParallelOp induction variable.
  for (auto dim : minOp.getDimOperands()) {
    auto ivArg = dim.dyn_cast<BlockArgument>();
    if (!ivArg) continue;
    Operation *containingOp = ivArg.getOwner()->getParentOp();
    auto forOp = dyn_cast_or_null<scf::ForOp>(containingOp);
    if (forOp && forOp.getInductionVar() == dim) {
      iv = dim;
      ub = forOp.getUpperBound();
      lb = forOp.getLowerBound();
      step = forOp.getStep();
      break;
    }
    auto parallelOp = dyn_cast_or_null<scf::ParallelOp>(containingOp);
    if (!parallelOp) continue;
    for (auto [index, inductionVar] :
         llvm::enumerate(parallelOp.getInductionVars())) {
      if (inductionVar == dim) {
        iv = dim;
        ub = parallelOp.getUpperBound()[index];
        lb = parallelOp.getLowerBound()[index];
        step = parallelOp.getStep()[index];
        break;
      }
    }
    if (iv) break;
  }
  if (!iv) return false;
  // Calculate the affine map representing `%ub - %iv`.
  AffineExpr ivDim;
  AffineExpr ubDim;
  for (auto [index, dim] : llvm::enumerate(minOp.getDimOperands())) {
    if (dim == iv) {
      ivDim = getAffineDimExpr(index, minOp.getContext());
    } else if (dim == ub) {
      ubDim = getAffineDimExpr(index, minOp.getContext());
    } else {
      return false;
    }
  }

  if (!ubDim) {
    if (auto cstUb = ub.getDefiningOp<arith::ConstantIndexOp>()) {
      ubDim = getAffineConstantExpr(cstUb.value(), minOp.getContext());
    } else {
      return false;
    }
  }
  AffineExpr diffExp = ubDim - ivDim;
  // Check that all the affine map results are either constant divisible by
  // `dividend` or equal to `%ub - %iv`.
  for (AffineExpr result : minOp.getAffineMap().getResults()) {
    if (auto cst = result.dyn_cast<AffineConstantExpr>()) {
      if (cst.getValue() <= 0 || cst.getValue() % dividend != 0) return false;
    } else {
      if (diffExp != result) return false;
    }
  }
  // Now check that for every value of the induction variable `%ub - %iv` is
  // divisible by `dividend`. It is true if the lower bounder, the upper bound
  // and the step are all divisible by `dividend`.
  std::array<Value, 3> loopOperands = {lb, step, ub};
  return llvm::all_of(loopOperands,
                      [dividend](Value v) { return isDivisible(v, dividend); });
}

/// Return true if we can prove that the value |v| is always divisible by the
/// constant |dividend|. Return false otherwise.
static bool isDivisible(Value v, int64_t dividend) {
  MLIRContext *ctx = v.getContext();
  // Create an expression (d0) -> (d0 % n) and try to simplify it.
  AffineExpr mod = getAffineDimExpr(0, ctx) % dividend;
  AffineMap modMap = AffineMap::get(1, 0, {mod}, ctx);
  SmallVector<Value> ops(1, v);
  fullyComposeAffineMapAndOperands(&modMap, &ops);
  canonicalizeMapAndOperands(&modMap, &ops);
  modMap = simplifyAffineMap(modMap);
  auto cst = modMap.getResult(0).dyn_cast<AffineConstantExpr>();
  if (cst) return (cst.getValue() == 0);
  // If the map doesn't fold to 0 but simplifies to (d0 %n) with d0 an
  // affine.min, check if all the results of the affine.min's map are divisible
  // by `dividend`.
  if (modMap.getResult(0) != mod) return false;
  assert(ops.size() == 1);
  auto minOp = ops[0].getDefiningOp<AffineMinOp>();
  return (minOp && affineMinOpDivisible(minOp, dividend));
}

/// Try to fold a affine.min op by matching the following form:
/// ```
/// scf.for %iv = %lb to %ub step %step
///   %affine.min affine_map<(d0, d1) -> (N, d0 - d1)>(%ub, %iv)
/// ```
/// With N a compile time constant. This operations can be replace by
/// `%cN = arith.constant N : index` if we can prove that %lb, %step and %ub are
/// divisible by N.
static std::optional<int64_t> foldAffineMin(AffineMinOp minOp) {
  AffineMap map = minOp.getAffineMap();
  int64_t constantResult = 0;
  for (AffineExpr result : map.getResults()) {
    if (auto cst = result.dyn_cast<AffineConstantExpr>()) {
      constantResult = cst.getValue();
    }
  }
  if (constantResult == 0) return {};
  // If afine.min map's results are all positive and divisible by
  // `constantResult` then it can be replaced by `constantResult`.
  if (affineMinOpDivisible(minOp, constantResult)) return constantResult;
  return {};
}

namespace {
struct AffineMinDistributedSCFCanonicalizationPattern
    : public mlir::OpRewritePattern<mlir::AffineMinOp> {
  using OpRewritePattern<mlir::AffineMinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::AffineMinOp minOp, mlir::PatternRewriter &rewriter) const override {
    std::optional<int64_t> cst = foldAffineMin(minOp);
    if (!cst) return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(minOp,
                                                   rewriter.getIndexAttr(*cst));
    return success();
  }
};

/// Pass to be able to test AffineMinDistributedSCFCanonicalizationPattern
/// individually.
struct AffineMinDistributedSCFCanonicalizationPass
    : public PassWrapper<AffineMinDistributedSCFCanonicalizationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AffineMinDistributedSCFCanonicalizationPass)

  StringRef getArgument() const override {
    return "iree-codegen-affinemin-scf-canonicalization";
  }

  StringRef getDescription() const override {
    return "Pass to run pass cleaning up affineMinOp after tiling and "
           "distribute.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet foldPattern(&getContext());
    populateAffineMinSCFCanonicalizationPattern(foldPattern);
    FrozenRewritePatternSet frozenPatterns(std::move(foldPattern));

    // Explicitly walk and apply the pattern locally to avoid more general
    // folding on the rest of the IR.
    SmallVector<Operation *> minOps;
    funcOp.walk([&minOps](AffineMinOp minOp) {
      minOps.push_back(minOp.getOperation());
    });
    (void)applyOpPatternsAndFold(minOps, frozenPatterns);
  }
};
}  // namespace

void populateAffineMinSCFCanonicalizationPattern(RewritePatternSet &patterns) {
  patterns.add<AffineMinDistributedSCFCanonicalizationPattern>(
      patterns.getContext());
}

static PassRegistration<AffineMinDistributedSCFCanonicalizationPass> pass([] {
  return std::make_unique<AffineMinDistributedSCFCanonicalizationPass>();
});

}  // namespace iree_compiler
}  // namespace mlir
