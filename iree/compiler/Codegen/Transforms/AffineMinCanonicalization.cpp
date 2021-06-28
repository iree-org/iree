// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AffineMinCanonicalization.cpp --------------------------------------===//
//
// Fold chains of AffineMinOp
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-codegen-affine-min-canonicalize"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

//===----------------------------------------------------------------------===//
// TODO: Cleanup and upstream these to go into core. Please ignore for now !
//===----------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {
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

namespace {
/// Perform folding of chains of AffineMinOp.
struct AffineMinCanonicalizationPattern
    : public mlir::OpRewritePattern<mlir::AffineMinOp> {
  using OpRewritePattern<mlir::AffineMinOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::AffineMinOp minOp, mlir::PatternRewriter &rewriter) const override;
};
}  // namespace

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

void populateAffineMinCanonicalizationPattern(RewritePatternSet &patterns) {
  patterns.add<AffineMinCanonicalizationPattern>(patterns.getContext());
}

}  // namespace iree_compiler
}  // namespace mlir
