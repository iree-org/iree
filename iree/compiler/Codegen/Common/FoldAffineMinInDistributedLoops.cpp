// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldAffineMinInDistributedLoops.cpp --------------------------------===//
//
// This file contains patterns to canonicalize affine.min ops inside tiled and
// distributed loops. These affine.min ops typically take the loop induction
// variable as operands and are used as the size for subtensors/subviews to
// guard against partial tiles. In the case of perfect tiling, they can go away
// to allow exposing static information for vectorization.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-affinemin-in-distributed-loops"

namespace mlir {
namespace iree_compiler {

/// Gets the given `attrOrValue` as a Value by creating constant ops for
/// attributes.
static Value getAsValue(OpFoldResult attrOrValue, OpBuilder &builder,
                        Location loc) {
  if (Value val = attrOrValue.dyn_cast<Value>()) return val;
  auto attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

#ifndef NDEBUG
inline raw_ostream &operator<<(raw_ostream &os,
                               const LoopTilingAndDistributionInfo &info) {
  os << "Loop tiling and distribution info:\n"
     << "\t[untiled lower bound] " << info.untiledLowerBound << "\n"
     << "\t[untiled upper bound] " << info.untiledUpperBound << "\n"
     << "\t[untiled step] " << info.untiledStep << "\n"
     << "\t[tile size] " << info.tileSize << "\n"
     << "\t[processor dimension] " << info.processorDistributionDim << "\n";
  return os;
}
#endif

namespace {

/// Folds `affine.min` ops over induction variables of tiled loops that are
/// distributed to processors, where we have the structure:
///
/// ```mlir
/// %id = hal.interface.workgroup.id ...
/// %count = hal.interface.workgroup.count ...
/// %offset = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%id, ...]
/// %size = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%count, ...]
/// scf.for %iv = %offset to ... step %size { ... }
/// ```
///
/// For such loops, we need to ignore the distribution aspect and get the
/// lower bound, upper bound, and step for only the tiling aspect, so that we
/// can reuse upstream utilities to prove that the `affine.min` ops are tightly
/// bound so that we can replace them with the tight bound.
struct FoldAffineMinOverDistributedLoopInductionVariable final
    : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override {
    Location loc = minOp.getLoc();

    auto loopMatcher = [&](Value iv, Value &lb, Value &ub, Value &step) {
      scf::ForOp forOp = scf::getForInductionVarOwner(iv);
      if (!forOp) return failure();

      auto loopInfo = isTiledAndDistributedLoop(forOp);
      if (!loopInfo) return failure();
      LLVM_DEBUG(llvm::dbgs() << *loopInfo);

      Optional<int64_t> untiledStep =
          getConstantIntValue(loopInfo->untiledStep);
      // For IREE right now the original untiled loop should have step 1..
      if (!untiledStep || *untiledStep != 1) return failure();
      // ..and we tile according to some static tile sizes for processors.
      if (!loopInfo->tileSize) return failure();

      lb = getAsValue(loopInfo->untiledLowerBound, rewriter, loc);
      ub = getAsValue(loopInfo->untiledUpperBound, rewriter, loc);
      // The "step" expected by the upstream utility is really the tiling size.
      step = rewriter.create<arith::ConstantIndexOp>(
          loc, loopInfo->tileSize.getValue());
      return success();
    };

    return scf::canonicalizeMinMaxOpInLoop(
        rewriter, minOp, minOp.getAffineMap(), minOp.operands(), /*isMin=*/true,
        loopMatcher);
  }
};

/// Folds `affine.apply` ops over induction variables that actually can only
/// take one single value.
//
// TODO: This should be removed and use the above pattern. But for some reason
// the above pattern does not handle a specific corner case:
//
// ```mlir
// scf.for %iv = %c0 to %c4 step %c4 {
//   %0 = affine.min affine_map<(d0, d1)[] -> (4, d1 - d0)> (%iv, %c4)
// ```
//
// The above will be folded into
//
// ```mlir
// scf.for %iv = %c0 to %c4 step %c4 {
//   %0 = affine.apply affine_map<(d0) -> (-d0 + 4)>(%iv)
// ```
//
// But we would expect `%0` to be `%c4` entirely. It looks like a bug. So use
// the following specific pattern to stop gap for now.
struct FoldAffineApplyOverSingleValueInductionVariable final
    : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    for (auto indexedOperand : llvm::enumerate(applyOp.getMapOperands())) {
      Value iv = indexedOperand.value();
      scf::ForOp forOp = scf::getForInductionVarOwner(iv);
      if (!forOp) continue;

      auto loopInfo = isTiledAndDistributedLoop(forOp);
      if (!loopInfo) continue;
      LLVM_DEBUG(llvm::dbgs() << *loopInfo);

      auto lbAttr = loopInfo->untiledLowerBound.dyn_cast<Attribute>();
      auto ubAttr = loopInfo->untiledUpperBound.dyn_cast<Attribute>();
      auto stepAttr = loopInfo->untiledStep.dyn_cast<Attribute>();
      auto tileSize = loopInfo->tileSize;
      if (!lbAttr || !ubAttr || !stepAttr || !tileSize) continue;

      int lb = lbAttr.cast<IntegerAttr>().getInt();
      int ub = ubAttr.cast<IntegerAttr>().getInt();
      int step = stepAttr.cast<IntegerAttr>().getInt();

      // For IREE right now the original untiled loop should have lower bound of
      // 1 and step 1. Then we tile to processors with each processor handling
      // `tileSize`.
      if (lb == 0 && step == 1 && lb < ub && *tileSize == ub) {
        // This is just tiling the whole workload into one tile.
        auto operands = llvm::to_vector<4>(applyOp.getMapOperands());
        operands[indexedOperand.index()] =
            getAsValue(loopInfo->untiledLowerBound, rewriter, iv.getLoc());
        rewriter.replaceOpWithNewOp<AffineMinOp>(
            applyOp, applyOp.getAffineMap(), operands);
        return success();
      }
    }

    return failure();
  }
};

struct FoldAffineMinInDistributedLoopsPass final
    : public FoldAffineMinInDistributedLoopsBase<
          FoldAffineMinInDistributedLoopsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFoldAffineMinInDistributedLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldAffineMinOverDistributedLoopInductionVariable,
               FoldAffineApplyOverSingleValueInductionVariable>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<FuncOp>>
createFoldAffineMinInDistributedLoopsPass() {
  return std::make_unique<FoldAffineMinInDistributedLoopsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
