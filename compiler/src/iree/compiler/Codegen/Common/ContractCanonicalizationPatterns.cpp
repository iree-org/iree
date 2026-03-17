// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern for canonicalizing vector.contract to the BMNK layout:
//   iteration space: [batch | lhs-free | rhs-free | reduction]
//   LHS operand:     [batch, lhs-free, reduction]
//   RHS operand:     [batch, rhs-free, reduction]
//   ACC operand:     [batch, lhs-free, rhs-free]
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Transforms.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTCONTRACTCANONICALIZELAYOUTPATTERNSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Computes the operand permutation needed to move `targetIters[i]` to
/// `targetPositions[i]` within `map`. Iters not present in `map` are silently
/// skipped. Returns std::nullopt if no present iter needs to move.
static std::optional<std::pair<SmallVector<int64_t>, AffineMap>>
getOperandPermToMoveItersToPositions(AffineMap map,
                                     ArrayRef<int64_t> targetIters,
                                     ArrayRef<int64_t> targetPositions,
                                     MLIRContext *ctx) {
  SmallVector<int64_t> sources, destinations;
  for (auto [iter, pos] : llvm::zip_equal(targetIters, targetPositions)) {
    std::optional<unsigned> curPos =
        map.getResultPosition(getAffineDimExpr(iter, ctx));
    if (!curPos.has_value()) {
      continue;
    }
    sources.push_back(static_cast<int64_t>(*curPos));
    destinations.push_back(pos);
  }
  if (sources.empty() || sources == destinations) {
    return std::nullopt;
  }

  SmallVector<int64_t> perm =
      computePermutationVector(map.getNumResults(), sources, destinations);
  SmallVector<AffineExpr> newResults;
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    newResults.push_back(map.getResult(perm[i]));
  }
  AffineMap newMap = AffineMap::get(map.getNumDims(), 0, newResults, ctx);
  return std::make_pair(perm, newMap);
}

/// Canonicalizes vector.contract so that the iteration space is ordered
/// [batch | lhs-free | rhs-free | reduction] and each operand's dimensions
/// follow the canonical layout:
///   LHS: [batch, lhs-free, reduction]
///   RHS: [batch, rhs-free, reduction]
///   ACC: [batch, lhs-free, rhs-free]
///
/// All dimension groups are reordered in a single rewrite step, producing
/// at most one transpose per operand, one mask transpose, and one result
/// inverse-transpose.
struct CanonicalizeContractLayout final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = maps[0], rhsMap = maps[1], accMap = maps[2];

    if (!llvm::all_of(
            maps, [](AffineMap map) { return map.isProjectedPermutation(); })) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "indexing maps are not all "
                                         "projected permutations");
    }

    MLIRContext *ctx = rewriter.getContext();

    FailureOr<linalg::ContractionDimensions> maybeDims =
        linalg::inferContractionDims(maps);
    if (failed(maybeDims)) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "not a contraction-like op");
    }

    SmallVector<int64_t> batchIters(maybeDims->batch.begin(),
                                    maybeDims->batch.end());
    SmallVector<int64_t> lhsFreeIters(maybeDims->m.begin(), maybeDims->m.end());
    SmallVector<int64_t> rhsFreeIters(maybeDims->n.begin(), maybeDims->n.end());

    int64_t numBatch = batchIters.size();
    int64_t numLhsFree = lhsFreeIters.size();
    int64_t numRhsFree = rhsFreeIters.size();

    if (numBatch + numLhsFree + numRhsFree == 0) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "no parallel dims to canonicalize");
    }

    // Iteration-space target: [batch | lhsFree | rhsFree | reduction].
    SmallVector<int64_t> currentIterOrder, canonicalIterOrder;
    for (int64_t i = 0; i < numBatch; ++i) {
      currentIterOrder.push_back(batchIters[i]);
      canonicalIterOrder.push_back(i);
    }
    for (int64_t i = 0; i < numLhsFree; ++i) {
      currentIterOrder.push_back(lhsFreeIters[i]);
      canonicalIterOrder.push_back(numBatch + i);
    }
    for (int64_t i = 0; i < numRhsFree; ++i) {
      currentIterOrder.push_back(rhsFreeIters[i]);
      canonicalIterOrder.push_back(numBatch + numLhsFree + i);
    }

    // Per-operand targets use operand-local positions:
    //   LHS: [batch @ 0..B-1, lhsFree @ B..B+M-1]
    //   RHS: [batch @ 0..B-1, rhsFree @ B..B+N-1]
    //   ACC: [batch @ 0..B-1, lhsFree @ B..B+M-1, rhsFree @ B+M..B+M+N-1]
    SmallVector<int64_t> lhsTargetIters, lhsTargetPos;
    SmallVector<int64_t> rhsTargetIters, rhsTargetPos;
    SmallVector<int64_t> accTargetIters, accTargetPos;
    for (int64_t i = 0; i < numBatch; ++i) {
      lhsTargetIters.push_back(batchIters[i]);
      lhsTargetPos.push_back(i);
      rhsTargetIters.push_back(batchIters[i]);
      rhsTargetPos.push_back(i);
      accTargetIters.push_back(batchIters[i]);
      accTargetPos.push_back(i);
    }
    for (int64_t i = 0; i < numLhsFree; ++i) {
      lhsTargetIters.push_back(lhsFreeIters[i]);
      lhsTargetPos.push_back(numBatch + i);
      accTargetIters.push_back(lhsFreeIters[i]);
      accTargetPos.push_back(numBatch + i);
    }
    for (int64_t i = 0; i < numRhsFree; ++i) {
      rhsTargetIters.push_back(rhsFreeIters[i]);
      rhsTargetPos.push_back(numBatch + i);
      accTargetIters.push_back(rhsFreeIters[i]);
      accTargetPos.push_back(numBatch + numLhsFree + i);
    }

    bool iterOk = (currentIterOrder == canonicalIterOrder);
    bool lhsOk = !getOperandPermToMoveItersToPositions(lhsMap, lhsTargetIters,
                                                       lhsTargetPos, ctx)
                      .has_value();
    bool rhsOk = !getOperandPermToMoveItersToPositions(rhsMap, rhsTargetIters,
                                                       rhsTargetPos, ctx)
                      .has_value();
    bool accOk = !getOperandPermToMoveItersToPositions(accMap, accTargetIters,
                                                       accTargetPos, ctx)
                      .has_value();

    if (iterOk && lhsOk && rhsOk && accOk) {
      return rewriter.notifyMatchFailure(contractOp, "already canonical");
    }

    Location loc = contractOp.getLoc();
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    Value acc = contractOp.getAcc();

    if (auto p = getOperandPermToMoveItersToPositions(lhsMap, lhsTargetIters,
                                                      lhsTargetPos, ctx)) {
      lhs = vector::TransposeOp::create(rewriter, loc, lhs, p->first);
      lhsMap = p->second;
    }
    if (auto p = getOperandPermToMoveItersToPositions(rhsMap, rhsTargetIters,
                                                      rhsTargetPos, ctx)) {
      rhs = vector::TransposeOp::create(rewriter, loc, rhs, p->first);
      rhsMap = p->second;
    }

    SmallVector<int64_t> accResultPerm;
    if (auto p = getOperandPermToMoveItersToPositions(accMap, accTargetIters,
                                                      accTargetPos, ctx)) {
      acc = vector::TransposeOp::create(rewriter, loc, acc, p->first);
      accResultPerm = p->first;
      accMap = p->second;
    }

    Value mask = maskOp ? maskOp.getMask() : Value();
    SmallVector<Attribute> newIterTypesVec(iteratorTypes.begin(),
                                           iteratorTypes.end());

    if (!iterOk) {
      SmallVector<int64_t> iterPerm = computePermutationVector(
          iteratorTypes.size(), currentIterOrder, canonicalIterOrder);
      SmallVector<int64_t> invIterPerm = invertPermutationVector(iterPerm);

      AffineMap permMap = AffineMap::getPermutationMap(invIterPerm, ctx);
      lhsMap = lhsMap.compose(permMap);
      rhsMap = rhsMap.compose(permMap);
      accMap = accMap.compose(permMap);

      SmallVector<Attribute> permuted(iteratorTypes.size());
      for (int64_t i = 0, e = iteratorTypes.size(); i < e; ++i) {
        permuted[i] = iteratorTypes[iterPerm[i]];
      }
      newIterTypesVec = permuted;

      if (mask) {
        mask = vector::TransposeOp::create(rewriter, loc, mask, iterPerm);
      }
    }

    SmallVector<AffineMap> newMaps = {lhsMap, rhsMap, accMap};
    auto newContract = vector::ContractionOp::create(
        rewriter, loc, lhs, rhs, acc, rewriter.getAffineMapArrayAttr(newMaps),
        rewriter.getArrayAttr(newIterTypesVec), contractOp.getKind());

    Value result;
    if (mask) {
      Operation *masked = vector::maskOperation(rewriter, newContract, mask);
      result = masked->getResult(0);
    } else {
      result = newContract.getResult();
    }

    if (!accResultPerm.empty()) {
      result = vector::TransposeOp::create(
          rewriter, loc, result, invertPermutationVector(accResultPerm));
    }

    return result;
  }
};

//===----------------------------------------------------------------------===//
// Test pass
//===----------------------------------------------------------------------===//

struct TestContractCanonicalizeLayoutPatternsPass final
    : impl::TestContractCanonicalizeLayoutPatternsPassBase<
          TestContractCanonicalizeLayoutPatternsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateContractLayoutCanonicalizationPatterns(patterns, /*benefit=*/1);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateContractLayoutCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit) {
  patterns.add<CanonicalizeContractLayout>(patterns.getContext(), benefit);
}

} // namespace mlir::iree_compiler
