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
// This code will be removed once this gets upstreamed to common mlir.
// Please try to limit changes in this code only minor changes.

#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"

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
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;          // NOLINT
using namespace mlir::linalg;  // NOLINT

#define DEBUG_TYPE "linalg-transform-utils"

//===----------------------------------------------------------------------===//
// TODO: Cleanup and upstream these to go into core. Please ignore for now !
//===----------------------------------------------------------------------===//
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

/// Return a fused vector::ContractionOp which represents a patterns such as:
///
/// ```mlir
///    %c0 = vector.constant 0: ...
///    %c = vector.contract %a, %b, %c0: ...
///    %e = add %c, %d: ...
/// ```
///
/// by:
///
/// ```mlir
///    %e = vector.contract %a, %b, %d: ...
/// ```
///
/// Return null if the canonicalization does not apply.
// TODO: This should be a folding of Add into Contract in core but while they
// live in different dialects, it is not possible without unnatural
// dependencies.
vector::ContractionOp mlir::canonicalizeContractionAdd(Operation *op) {
  if (!isa<AddIOp, AddFOp>(op)) return nullptr;

  OpBuilder builder(op);
  auto canonicalize = [](OpBuilder &b, Value maybeContraction,
                         Value otherOperand) -> vector::ContractionOp {
    vector::ContractionOp contractionOp =
        dyn_cast_or_null<vector::ContractionOp>(
            maybeContraction.getDefiningOp());
    if (!contractionOp) return nullptr;
    if (auto maybeZero =
            dyn_cast_or_null<ConstantOp>(contractionOp.acc().getDefiningOp())) {
      if (maybeZero.value() == b.getZeroAttr(contractionOp.acc().getType())) {
        BlockAndValueMapping bvm;
        bvm.map(contractionOp.acc(), otherOperand);
        return cast<vector::ContractionOp>(b.clone(*contractionOp, bvm));
      }
    }
    return nullptr;
  };

  Value a = op->getOperand(0), b = op->getOperand(1);
  vector::ContractionOp contract = canonicalize(builder, a, b);
  contract = contract ? contract : canonicalize(builder, b, a);
  return contract;
}
//===----------------------------------------------------------------------===//
// END TODO
//===----------------------------------------------------------------------===//
