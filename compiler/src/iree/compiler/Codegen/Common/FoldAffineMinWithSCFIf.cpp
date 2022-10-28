// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldAffineMinWithSCFIf.cpp -----------------------------------------===//
//
// This file contains patterns to canonicalize affine.min uses inside scf::IfOp
// by applying the condition to the affine min.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-affinemin-with-scfif"

namespace mlir {
namespace iree_compiler {

namespace {

static FailureOr<unsigned> findIndexOfConstResult(const AffineMap &map,
                                                  int64_t value) {
  for (unsigned i = 0, e = map.getNumResults(); i != e; ++i) {
    if (auto cstExpr = map.getResult(i).dyn_cast<AffineConstantExpr>()) {
      if (cstExpr.getValue() == value) return i;
    }
  }

  return failure();
}

static SmallVector<OpOperand *> getUsesInRegion(Value value, Region *region) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (region->isAncestor(use.getOwner()->getParentRegion())) {
      uses.push_back(&use);
    }
  }
  return uses;
}

LogicalResult matchTileAndDistributionLoop(Value iv, OpFoldResult &lb,
                                           OpFoldResult &ub,
                                           OpFoldResult &step) {
  scf::ForOp forOp = scf::getForInductionVarOwner(iv);
  if (!forOp) return failure();

  auto loopInfo = isTiledAndDistributedLoop(forOp);
  if (!loopInfo) return failure();

  Optional<int64_t> untiledStep = getConstantIntValue(loopInfo->untiledStep);
  // For IREE right now the original untiled loop should have step 1..
  if (!untiledStep || *untiledStep != 1) return failure();
  // ..and we tile according to some static tile sizes for processors.
  if (!loopInfo->tileSize) return failure();

  lb = loopInfo->untiledLowerBound;
  ub = loopInfo->untiledUpperBound;
  // The "step" expected by the upstream utility is really the tiling size.
  step = OpBuilder(iv.getContext()).getIndexAttr(loopInfo->tileSize.value());
  return success();
}

/// Match affine.min affine_map<(d0) -> (-d0 + UB, TILESIZE)>(%iv) from a
/// tile-and-distribution loop.
static LogicalResult matchBoundedTileSizeOp(AffineMinOp minOp, int64_t &lb,
                                            int64_t &ub, int64_t &tileSize) {
  if (minOp.getNumOperands() != 1) return failure();

  AffineMap map = minOp.getAffineMap();
  if (map.getNumResults() != 2) return failure();

  OpFoldResult lbOfr, ubOfr, tileSizeOfr;
  if (failed(matchTileAndDistributionLoop(minOp.getOperand(0), lbOfr, ubOfr,
                                          tileSizeOfr)))
    return failure();

  Optional<int64_t> lbOpt = getConstantIntValue(lbOfr);
  if (!lbOpt) return failure();
  Optional<int64_t> ubOpt = getConstantIntValue(ubOfr);
  if (!ubOpt) return failure();
  Optional<int64_t> tileSizeOpt = getConstantIntValue(tileSizeOfr);
  if (!tileSizeOpt) return failure();

  lb = *lbOpt;
  ub = *ubOpt;
  tileSize = *tileSizeOpt;
  return success();
}

/// Folds `affine.min` if its constant value argument is used by scf.if,
/// where the op is folded into one of the two values.
///
/// ```mlir
/// %min = affine.min(%a, %cst)
/// %cmp = icmp.eq %min, %cst
/// scf.if %cmp {
///   // expect the uses of %min are already replaced by %cst
/// } else {
///   // use of %min can be replaced by %a
/// }
/// ```
struct FoldAffineMinWithSCFIf final : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto cmpOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmpOp) return failure();
    if (cmpOp.getPredicate() != arith::CmpIPredicate::eq) return failure();

    AffineMinOp minOp;
    Value b;
    if (isa<AffineMinOp>(cmpOp.getLhs().getDefiningOp())) {
      minOp = cast<AffineMinOp>(cmpOp.getLhs().getDefiningOp());
      b = cmpOp.getRhs();
    } else if (isa<AffineMinOp>(cmpOp.getRhs().getDefiningOp())) {
      minOp = cast<AffineMinOp>(cmpOp.getRhs().getDefiningOp());
      b = cmpOp.getLhs();
    } else {
      return failure();
    }
    auto cstOp = dyn_cast<arith::ConstantIndexOp>(b.getDefiningOp());
    if (!cstOp) return failure();
    int64_t cstInt = cstOp.value();

    AffineMap minMap = minOp.getAffineMap();
    if (minMap.getNumResults() != 2) return failure();

    FailureOr<unsigned> affineCstIndexOrFailure =
        findIndexOfConstResult(minMap, cstInt);
    if (failed(affineCstIndexOrFailure)) return failure();
    unsigned affineCstIndex = *affineCstIndexOrFailure;

    // find the uses of min in the else block.
    SmallVector<OpOperand *> usesInElse =
        getUsesInRegion(minOp, &ifOp.getElseRegion());
    if (usesInElse.empty()) return failure();

    OpBuilder builder(ifOp);
    OpBuilder::InsertionGuard guard(builder);
    int64_t lb, ub, tileSize;
    Value valueToReplace;
    if (succeeded(matchBoundedTileSizeOp(minOp, lb, ub, tileSize))) {
      // The other part is the partial tile size, (ub - lb) % tileSize.
      valueToReplace = builder.create<arith::ConstantIndexOp>(
          minOp.getLoc(), (ub - lb) % tileSize);
    } else {
      // A general case. Construct a new affine map without the constant result.
      AffineMap newMap = minMap.dropResult(affineCstIndex);
      valueToReplace = builder.create<AffineApplyOp>(minOp.getLoc(), newMap,
                                                     minOp.getMapOperands());
    }
    for (auto use : usesInElse) {
      use->set(valueToReplace);
    }
    return success();
  }
};

struct FoldAffineMinWithSCFIfPass final
    : public FoldAffineMinWithSCFIfBase<FoldAffineMinWithSCFIfPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFoldAffineMinWithSCFIfPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

void populateFoldAffineMinWithSCFIfPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldAffineMinWithSCFIf>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
createFoldAffineMinWithSCFIfPass() {
  return std::make_unique<FoldAffineMinWithSCFIfPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
