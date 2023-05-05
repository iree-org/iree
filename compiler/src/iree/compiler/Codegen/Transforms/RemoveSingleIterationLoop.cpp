// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- RemoveSingleIterationLoop.cpp - Remove single iteration loops ------===//
//
// Removes loops that are known to be single-trip count even when the loop
// itself might be distributed.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-codegen-remove-single-iteration"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir {
namespace iree_compiler {

/// Compose map with apply affine ops and try to simplify it.
static void combineAndSimplifyMap(AffineMap &map, SmallVectorImpl<Value> &dims,
                                  SmallVectorImpl<Value> &symbols) {
  SmallVector<Value, 4> operands(dims.begin(), dims.end());
  operands.append(symbols.begin(), symbols.end());
  // Pull in affine.apply operations and compose them fully into the
  // result.
  fullyComposeAffineMapAndOperands(&map, &operands);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  // Assign the results.
  dims.assign(operands.begin(), operands.begin() + map.getNumDims());
  symbols.assign(operands.begin() + map.getNumDims(), operands.end());
}

/// Replace dimensions and symbols with known range in the map expression.
// TODO: Use core function once the interface using a lambda lands.
static AffineMap substituteMin(AffineMap map, SmallVectorImpl<Value> &dims,
                               SmallVectorImpl<Value> &symbols,
                               GetMinMaxExprFn getMinMaxExpr) {
  combineAndSimplifyMap(map, dims, symbols);
  auto exprs = llvm::to_vector<4>(map.getResults());
  for (AffineExpr &expr : exprs) {
    bool substituted = true;
    while (substituted) {
      substituted = false;
      for (unsigned dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
        Value dim = dims[dimIdx];
        auto minMax = getMinMaxExpr(dim, dims, symbols);
        if (!minMax) continue;
        AffineExpr dimExpr = getAffineDimExpr(dimIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << dim << " @ " << dimExpr << "\n");
        LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
        // Substitute occurrences of `dimExpr` by either the min expression or
        // the max expression depending on whether the value is used with a
        // positive or negative  coefficient.
        AffineExpr substitutedExpr =
            substWithMin(expr, dimExpr, minMax->first, minMax->second);
        LLVM_DEBUG(DBGS() << "After: " << substitutedExpr << "\n");
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
      // Substitute symbols
      for (unsigned symIdx = 0; symIdx < symbols.size(); ++symIdx) {
        Value sym = symbols[symIdx];
        auto minMax = getMinMaxExpr(sym, dims, symbols);
        if (!minMax) continue;
        AffineExpr symExpr = getAffineSymbolExpr(symIdx, expr.getContext());
        LLVM_DEBUG(DBGS() << "Subst: " << sym << " @ " << symExpr << "\n");
        LLVM_DEBUG(DBGS() << "Before: " << expr << "\n");
        AffineExpr substitutedExpr =
            substWithMin(expr, symExpr, minMax->first, minMax->second);
        LLVM_DEBUG(DBGS() << "After: " << substitutedExpr << "\n");
        substituted = (substitutedExpr != expr);
        expr = substitutedExpr;
      }
    }

    map = AffineMap::get(dims.size(), symbols.size(), exprs,
                         exprs.front().getContext());
    // Cleanup and simplify the results.
    // This needs to happen outside of the loop iterating on dims.size() since
    // it modifies dims.
    combineAndSimplifyMap(map, dims, symbols);
    // Assign the results.
    exprs.assign(map.getResults().begin(), map.getResults().end());
    LLVM_DEBUG(DBGS() << "Map simplified: " << map << "\n");
  }

  assert(!exprs.empty() && "Unexpected empty exprs");
  return AffineMap::get(dims.size(), symbols.size(), exprs, map.getContext());
}

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Return true if we can prove that the we always run at least the first
/// iteration of the ForOp.
static bool alwaysRunsFirstIteration(scf::ForOp op, GetMinMaxExprFn getMinMax) {
  // Calculate the minimum value of ub - lb. If it is strictly positive it
  // means the loop will always run at least once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.getLowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.getUpperBound());
  AffineExpr iterZero = ub - lb;
  auto map = AffineMap::get(dims.size(), 0, iterZero);
  AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() > 0) return true;
  }
  return false;
}

/// Return true if we can prove that the we never run more than one iteration of
/// the ForOp.
static bool neverRunsSecondIteration(scf::ForOp op, GetMinMaxExprFn getMinMax) {
  // Calculate the minimum of lb + step - ub. If it is positive it means the
  // loop never run more than once.
  MLIRContext *ctx = op->getContext();
  SmallVector<Value, 4> dims;
  SmallVector<Value, 4> symbols;
  AffineExpr lb = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.getLowerBound());
  AffineExpr ub = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.getUpperBound());
  AffineExpr step = getAffineDimExpr(dims.size(), ctx);
  dims.push_back(op.getStep());
  AffineExpr iterOne = lb + step - ub;
  auto map = AffineMap::get(dims.size(), 0, iterOne);

  AffineMap simplifiedMap = substituteMin(map, dims, symbols, getMinMax);
  assert(simplifiedMap.getNumResults() == 1);
  if (auto cst = simplifiedMap.getResult(0).dyn_cast<AffineConstantExpr>()) {
    if (cst.getValue() >= 0) return true;
  }
  return false;
}

namespace {
/// Rewriting pattern that replaces single-iteration loops with their bodies.
struct SimplifyTrivialLoops : public OpRewritePattern<scf::ForOp> {
  SimplifyTrivialLoops(MLIRContext *context, GetMinMaxExprFn getMinMax)
      : OpRewritePattern<scf::ForOp>(context, 1), getMinMax(getMinMax) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Handle the case where we know that the loop doesn't run more than
    // once but the loop may not run at least once by replace the `loop` with an
    // `if`.
    if (!(alwaysRunsFirstIteration(op, getMinMax) &&
          neverRunsSecondIteration(op, getMinMax))) {
      return failure();
    }

    // The first iteration is always run and the second iteration is never run
    // so the loop always have 1 iteration. Inline its body and remove the loop.
    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(op.getNumIterOperands() + 1);
    blockArgs.push_back(op.getLowerBound());
    llvm::append_range(blockArgs, op.getIterOperands());
    replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
    return success();
  }

 private:
  GetMinMaxExprFn getMinMax;
};

}  // namespace

void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              GetMinMaxExprFn getMinMaxFn) {
  patterns.add<SimplifyTrivialLoops>(patterns.getContext(), getMinMaxFn);
}

}  // namespace iree_compiler
}  // namespace mlir
