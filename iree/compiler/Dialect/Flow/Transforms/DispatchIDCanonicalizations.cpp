// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-flow-canonicalize-dispatch-ids"

/// This file contains patterns allowing canonicalization taking advantage  of
/// flow dispatch ops. When the ID range is know we may be able to calculate the
/// min/max of the derived values and may allow folding affine.min or ForOp.
/// This file duplicate some of the logic from
/// `linalg::AffineMinSCFCanonicalizationPattern`. In the future we could unify
/// this code back by upstreaminng an interface to MLIR to represent generic ID
/// ops.
namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Traverse `e` and return an AffineExpr where all occurrences of `dim` have
/// been replaced by either:
///  - `min` if `positivePath` is true when we reach an occurrence of `dim`
///  - `max` if `positivePath` is true when we reach an occurrence of `dim`
/// `positivePath` is negated each time we hit a multiplicative or divisive
/// binary op with a constant negative coefficient.
static AffineExpr substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                               AffineExpr max, bool positivePath = true) {
  if (e == dim) return positivePath ? min : max;
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    if (bin.getKind() == mlir::AffineExprKind::Add)
      return substWithMin(lhs, dim, min, max, positivePath) +
             substWithMin(rhs, dim, min, max, positivePath);

    auto c1 = bin.getLHS().dyn_cast<AffineConstantExpr>();
    auto c2 = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (c1 && c1.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), c1, substWithMin(rhs, dim, min, max, !positivePath));
    if (c2 && c2.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), substWithMin(lhs, dim, min, max, !positivePath), c2);
    return getAffineBinaryOpExpr(
        bin.getKind(), substWithMin(lhs, dim, min, max, positivePath),
        substWithMin(rhs, dim, min, max, positivePath));
  }
  return e;
}

/// Traverse the `dims` and substitute known min or max expressions in place of
/// values whose range is known. We know the range of some operations based on
/// their semantic.
static AffineMap substitute(AffineMap map, SmallVectorImpl<Value> &dims,
                            SmallVectorImpl<Value> &symbols) {
  auto exprs = llvm::to_vector<4>(map.getResults());
  bool simplified;
  do {
    simplified = false;
    for (AffineExpr &expr : exprs) {
      bool substituted = true;
      while (substituted) {
        substituted = false;
        for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
          Value dim = dims[dimIdx];
          AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
          LLVM_DEBUG(llvm::dbgs()
                     << "Subst: " << dim << " @ " << dimExpr << "\n");
          AffineExpr substitutedExpr;
          // TODO: Generalize IDs/Count by adding an interface to support more
          // kind of ID Ops.
          if (auto idOp =
                  dim.getDefiningOp<IREE::Flow::DispatchWorkgroupIDOp>()) {
            OpBuilder b(map.getContext());
            auto zero = b.getAffineConstantExpr(0);
            auto parent =
                idOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
            Value ub =
                parent.workgroup_count()[idOp.dimension().getZExtValue()];
            AffineExpr ubExpr = getAffineDimExpr(dims.size(), map.getContext());
            dims.push_back(ub);
            substitutedExpr = substWithMin(expr, dimExpr, zero, ubExpr - 1);
          } else if (auto countOp = dim.getDefiningOp<
                                    IREE::Flow::DispatchWorkgroupCountOp>()) {
            auto parent =
                countOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
            Value count =
                parent.workgroup_count()[countOp.dimension().getZExtValue()];
            AffineExpr countExpr =
                getAffineDimExpr(dims.size(), map.getContext());
            dims.push_back(count);
            substitutedExpr = expr.replace(dimExpr, countExpr);
          }
          if (!substitutedExpr) continue;

          substituted = (substitutedExpr != expr);
          expr = substitutedExpr;
        }
      }

      // Cleanup and simplify the results.
      // This needs to happen outside of the loop iterating on dims.size() since
      // it modifies dims.
      SmallVector<Value, 4> operands(dims.begin(), dims.end());
      operands.append(symbols.begin(), symbols.end());
      auto map = AffineMap::get(dims.size(), symbols.size(), exprs,
                                exprs.front().getContext());
      AffineMap orignalMap = map;

      LLVM_DEBUG(llvm::dbgs() << "Map to simplify: " << map << "\n");

      // Pull in affine.apply operations and compose them fully into the
      // result.
      fullyComposeAffineMapAndOperands(&map, &operands);
      canonicalizeMapAndOperands(&map, &operands);
      map = simplifyAffineMap(map);
      // Assign the results.
      exprs.assign(map.getResults().begin(), map.getResults().end());
      dims.assign(operands.begin(), operands.begin() + map.getNumDims());
      symbols.assign(operands.begin() + map.getNumDims(), operands.end());
      simplified |= map != orignalMap;

      LLVM_DEBUG(llvm::dbgs() << "Map simplified: " << map << "\n");
    }
  } while (simplified);

  assert(!exprs.empty() && "Unexpected empty exprs");
  return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
}

struct AffineMinSCFCanonicalizationPattern
    : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult AffineMinSCFCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Canonicalize AffineMinSCF: "
                          << *minOp.getOperation() << "\n");

  SmallVector<Value, 4> dims(minOp.getDimOperands()),
      symbols(minOp.getSymbolOperands());
  AffineMap map = substitute(minOp.getAffineMap(), dims, symbols);

  LLVM_DEBUG(llvm::dbgs() << "Resulting map: " << map << "\n");

  // Check whether any of the expressions, when subtracted from all other
  // expressions, produces only >= 0 constants. If so, it is the min.
  for (auto e : minOp.getAffineMap().getResults()) {
    LLVM_DEBUG(llvm::dbgs() << "Candidate min: " << e << "\n");
    if (!e.isSymbolicOrConstant()) continue;

    auto isNonPositive = [](AffineExpr e) {
      if (auto cst = e.dyn_cast<AffineConstantExpr>())
        return cst.getValue() < 0;
      return true;
    };

    // Build the subMap and check everything is statically known to be
    // positive.
    SmallVector<AffineExpr, 4> subExprs;
    subExprs.reserve(map.getNumResults());
    for (auto ee : map.getResults()) subExprs.push_back(ee - e);
    MLIRContext *ctx = minOp.getContext();
    AffineMap subMap = simplifyAffineMap(
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), subExprs, ctx));
    LLVM_DEBUG(llvm::dbgs() << "simplified subMap: " << subMap << "\n");
    if (llvm::any_of(subMap.getResults(), isNonPositive)) continue;

    // Static min found.
    if (auto cst = e.dyn_cast<AffineConstantExpr>()) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp, cst.getValue());
    } else {
      auto resultMap = AffineMap::get(0, map.getNumSymbols(), {e}, ctx);
      SmallVector<Value, 4> resultOperands = dims;
      resultOperands.append(symbols.begin(), symbols.end());
      canonicalizeMapAndOperands(&resultMap, &resultOperands);
      resultMap = simplifyAffineMap(resultMap);
      rewriter.replaceOpWithNewOp<AffineApplyOp>(minOp, resultMap,
                                                 resultOperands);
    }
    return success();
  }

  return failure();
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(scf::ForOp op) {
  // Calculate the minimum value of ub - lb. If it is strictly positive it
  // means the loop will always run at least once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr iterZero = ub - lb;
  auto map = AffineMap::get(dims.size(), 0, iterZero);
  AffineMap simplifiedMap = substitute(map, dims, symbols);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() > 0) return true;
  }
  return false;
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(scf::ForOp op) {
  // Calculate the minimum of lb + step - ub. If it is positive it means the
  // loop never run more than once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.lowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.upperBound());
  AffineExpr step = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.step());
  AffineExpr iterOne = lb + step - ub;
  auto map = AffineMap::get(dims.size(), 0, iterOne);

  AffineMap simplifiedMap = substitute(map, dims, symbols);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() >= 0) return true;
  }
  return false;
}

/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle the case where we know that the loop doesn't run more than
    // once but the loop may not run at least once by replace the `loop` with an
    // `if`.
    if (!(alwaysRunsFirstIteration(op) && neverRunsSecondIteration(op)))
      return failure();

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.lowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());
    replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
    return success();
  }
};

class DispatchIDCanonicalizationPass
    : public PassWrapper<DispatchIDCanonicalizationPass,
                         OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    OwningRewritePatternList patterns;
    patterns.insert<SimplifyTrivialLoops, AffineMinSCFCanonicalizationPattern>(
        context);
    applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDispatchIDCanonicalizationPass() {
  return std::make_unique<DispatchIDCanonicalizationPass>();
}

static PassRegistration<DispatchIDCanonicalizationPass> pass(
    "iree-flow-dispatch-id-canonicalizations",
    "Canonicalization patterns related to flow disptach ops.");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
