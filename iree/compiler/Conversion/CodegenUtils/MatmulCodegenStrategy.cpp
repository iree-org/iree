// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// -----------------------------------------------------------------------------
// This is a copy of the matmul strategy infrastructure existing in mlir_edge.
// This version will be removed once this gets upstreamed to common mlir.
// Please try to limit changes in this code only minor changes or make sure the
// changes are applied in mlir_edge as well.

#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;          // NOLINT
using namespace mlir::linalg;  // NOLINT

#define DEBUG_TYPE "matmul-codegen-strategy"

//===----------------------------------------------------------------------===//
// TODO: Cleanup and upstream these to go into core. Please ignore for now !
//===----------------------------------------------------------------------===//
static void hoistRedundantCopies(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](linalg::FillOp op) {
      auto loop = op.getParentOfType<scf::ForOp>();
      if (!loop) return;

      for (auto operand : op.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand)) return;

      // Hoist fill before.
      op.getOperation()->moveBefore(loop);
      changed = true;
    });

    func.walk([&](linalg::CopyOp op) {
      auto loop = op.getParentOfType<scf::ForOp>();
      if (!loop) return;

      for (auto operand : op.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand)) return;

      Value sourceView = op.getInput(0);
      while (auto subViewOp = sourceView.getDefiningOp<SubViewOp>())
        sourceView = subViewOp.getViewSource();

      // Source traces back to a block argument.
      if (sourceView.isa<BlockArgument>()) {
        op.getOperation()->moveBefore(loop);
      } else {
        assert(sourceView.getDefiningOp<ViewOp>() ||
               sourceView.getDefiningOp<AllocOp>() ||
               sourceView.getDefiningOp<AllocaOp>());
        op.getOperation()->moveAfter(loop);
      }
      changed = true;
    });
  }
}

/// Substitute scf.for = %lb to %ub step %step by an AffineExpr expressing:
///   `%lb + %step * new_dim` where
/// 1. the AffineExpr for %lb is either an AffineConstantExpr or an
/// AffineDimExpr depending on whether the value is constant or not.
/// 2. the AffineExpr for %step is either an AffineConstantExpr or an
/// AffineSymbolExpr depending on whether the value is constant or not.
///
static void substitute(scf::ForOp forOp, SmallVectorImpl<AffineExpr> &exprs,
                       SmallVectorImpl<Value> &dims,
                       SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = forOp.getContext();
  auto lbConstant = forOp.lowerBound().getDefiningOp<ConstantIndexOp>();
  AffineExpr lb = lbConstant ? getAffineConstantExpr(lbConstant.getValue(), ctx)
                             : getAffineDimExpr(dims.size(), ctx);

  auto stepConstant = forOp.step().getDefiningOp<ConstantIndexOp>();
  AffineExpr step = stepConstant
                        ? getAffineConstantExpr(stepConstant.getValue(), ctx)
                        : getAffineSymbolExpr(symbols.size(), ctx);

  if (!lbConstant) dims.push_back(forOp.lowerBound());
  if (!stepConstant) symbols.push_back(forOp.step());
  exprs.push_back(lb + step * getAffineDimExpr(dims.size(), ctx));

  auto ubConstant = forOp.upperBound().getDefiningOp<ConstantIndexOp>();
  AffineExpr ub = ubConstant ? getAffineConstantExpr(ubConstant.getValue(), ctx)
                             : getAffineDimExpr(dims.size(), ctx);
  if (!ubConstant) dims.push_back(forOp.upperBound());
  exprs.push_back(ub);

  dims.push_back(forOp.getInductionVar());
}

/// Substitue dimensions coming from forOp or AffineMin. Return false if it has
/// unknown dimension operands.
static bool substitute(AffineMinOp minOp, SmallVectorImpl<AffineExpr> &exprs,
                       SmallVectorImpl<Value> &dims,
                       SmallVectorImpl<Value> &symbols) {
  if (minOp.getDimOperands().empty()) return false;
  for (Value v : minOp.getDimOperands()) {
    if (auto forOp = scf::getForInductionVarOwner(v)) {
      substitute(forOp, exprs, dims, symbols);
      continue;
    }
    if (auto parentMinOp = v.getDefiningOp<AffineMinOp>()) {
      substitute(parentMinOp, exprs, dims, symbols);
      continue;
    }
    // If couldn't substitue the dimension give up and use the original map.
    return false;
  }
  return true;
}

/// Perform folding of chains of AffineMinOp.
struct AffineMinCanonicalizationPattern : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult AffineMinCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "\nCanonicalize AffineMin: "
                          << *minOp.getOperation() << "\n");

  int64_t min = std::numeric_limits<int64_t>::max();
  for (auto e : minOp.map().getResults())
    if (auto cstExpr = e.dyn_cast<AffineConstantExpr>())
      min = std::min(min, cstExpr.getValue());
  if (min == std::numeric_limits<int64_t>::max()) return failure();

  MLIRContext *ctx = minOp.getContext();
  AffineMap map;
  SmallVector<Value, 4> operands;
  SmallVector<AffineExpr, 4> exprs;
  SmallVector<Value, 4> dims, symbols;
  if (substitute(minOp, exprs, dims, symbols)) {
    operands = dims;
    operands.append(symbols.begin(), symbols.end());

    map = AffineMap::get(dims.size(), symbols.size(), exprs, ctx);
    LLVM_DEBUG(llvm::dbgs() << "Substitution map: " << map << "\n");
  } else {
    map = minOp.getAffineMap();
    operands = minOp.getDimOperands();
    operands.append(minOp.getSymbolOperands().begin(),
                    minOp.getSymbolOperands().end());
  }
  SmallVector<AffineExpr, 4> modExprs;
  for (unsigned idx = 0, e = map.getNumResults(); idx < e; ++idx)
    modExprs.push_back(getAffineDimExpr(idx, ctx) % min);
  map = AffineMap::get(map.getNumResults(), 0, modExprs, ctx).compose(map);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);

  LLVM_DEBUG(llvm::dbgs() << "Post mod: " << map << "\n";
             llvm::interleaveComma(operands, llvm::dbgs()));

  if (!llvm::all_of(map.getResults(), [](AffineExpr e) {
        if (auto cst = e.dyn_cast<AffineConstantExpr>())
          return cst.getValue() == 0;
        return false;
      }))
    return failure();

  rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp, min);
  return success();
}
//===----------------------------------------------------------------------===//
// END TODO
//===----------------------------------------------------------------------===//

void MatmulCodegenStrategy::transform(FuncOp func) const {
  MLIRContext *context = func.getContext();
  // Emplace patterns one at a time while also maintaining a simple chained
  // state transition.
  unsigned stepCount = 0;
  SmallVector<OwningRewritePatternList, 4> stage1Patterns;
  auto zeroState = Identifier::get(std::to_string(stepCount), context);
  auto currentState = zeroState;
  for (auto &t : transformationSequence) {
    auto nextState = Identifier::get(std::to_string(++stepCount), context);
    auto marker = (currentState == zeroState)
                      ? linalg::LinalgMarker({}, nextState)
                      : linalg::LinalgMarker(currentState, nextState);
    stage1Patterns.emplace_back(t->buildRewritePatterns(context, marker));
    currentState = nextState;
  }

  OwningRewritePatternList stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  // Add extra patterns to canonicalize AffineMin in combination with scf loops
  // operations after tiling.
  stage2Patterns.insert<AffineMinCanonicalizationPattern,
                        AffineMinSCFCanonicalizationPattern>(context);

  auto stage3Transforms = [](Operation *op) {
    promoteSingleIterationLoops(cast<FuncOp>(op));
    return success();
  };
  linalg::applyStagedPatterns(func, stage1Patterns, stage2Patterns,
                              stage3Transforms);

  auto postStageTransforms = [this](Operation *op) {
    // Run LICM and hoisting patterns after all the stages as we want to
    // unrolling before moving transfer ops out of the loop.
    if (hoistInvariantCode) {
      PassManager pm(op->getContext());
      pm.addPass(createLoopInvariantCodeMotionPass());
      if (failed(pm.run(op->getParentOfType<ModuleOp>())))
        llvm_unreachable("Unexpected failure in cleanup pass pipeline.");
      hoistViewAllocOps(cast<FuncOp>(op));
      hoistRedundantVectorTransfers(cast<FuncOp>(op));
      hoistRedundantCopies(cast<FuncOp>(op));
    }
  };
  postStageTransforms(func);
  if (lowering != nullptr) lowering(func);
}

// Parametric lowering of vector contract for CPU target.
static void cpuLowering(
    FuncOp func, const vector::VectorTransformsOptions &vectorTransformsOptions,
    const VectorTransferToSCFOptions &vectorToSCFOptions) {
  // Programmatic controlled lowering of vector.contract only.
  MLIRContext *context = func.getContext();
  OwningRewritePatternList vectorContractLoweringPatterns;
  vectorContractLoweringPatterns
      .insert<ContractionOpToOuterProductOpLowering,
              ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          vectorTransformsOptions, context);

  applyPatternsAndFoldGreedily(func, vectorContractLoweringPatterns);

  // Programmatic controlled lowering of vector.transfer only.
  OwningRewritePatternList vectorToLoopsPatterns;
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                        vectorToSCFOptions);
  applyPatternsAndFoldGreedily(func, vectorToLoopsPatterns);
}

MatmulCodegenStrategy &MatmulCodegenStrategy::setDefaultCPULowering() {
  auto lowering = [this](FuncOp func) {
    cpuLowering(func, vectorTransformsOptions, vectorToSCFOptions);
  };
  return setLoweringFunction(lowering);
}
