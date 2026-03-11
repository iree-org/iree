// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for unrolling vector.contract along batch/free dimensions and
// lowering pure-reduction contracts to vector.multi_reduction.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/ContractUnrollingPatterns.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTCONTRACTUNROLLINGPATTERNSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Creates a new affine map with the specified iterator removed.
/// All iterator indices greater than the removed one are decremented.
static AffineMap dropIteratorFromMap(AffineMap map, int64_t iterIdx,
                                     MLIRContext *ctx) {
  SmallVector<AffineExpr> newResults;
  for (int64_t i = 0, e = map.getNumResults(); i < e; ++i) {
    int64_t dimPos = map.getDimPosition(i);
    if (dimPos == iterIdx) {
      continue;
    }
    int64_t newDimPos = dimPos > iterIdx ? dimPos - 1 : dimPos;
    newResults.push_back(getAffineDimExpr(newDimPos, ctx));
  }
  return AffineMap::get(map.getNumDims() - 1, 0, newResults, ctx);
}

/// Creates new iterator types with the specified iterator removed.
static SmallVector<Attribute> dropIteratorType(ArrayAttr iteratorTypes,
                                               int64_t iterIdx) {
  SmallVector<Attribute> result(iteratorTypes.begin(), iteratorTypes.end());
  result.erase(result.begin() + iterIdx);
  return result;
}

/// Shared implementation for unrolling a vector.contract along its outermost
/// accumulator dimension. The caller must have validated that the outermost
/// ACC dimension corresponds to the intended iterator class (batch, lhsFree,
/// or rhsFree). `sliceLhs`/`sliceRhs` control which operands are extracted
/// along dim 0; ACC and mask are always sliced.
static FailureOr<Value> unrollContractAlongOutermostDim(
    vector::ContractionOp contractOp, vector::MaskingOpInterface maskOp,
    PatternRewriter &rewriter, bool sliceLhs, bool sliceRhs) {
  SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
  ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = contractOp.getLoc();

  int64_t iterIdx = maps[2].getDimPosition(0);
  int64_t dimSize =
      cast<VectorType>(contractOp.getAcc().getType()).getDimSize(0);

  Value lhs = contractOp.getLhs();
  Value rhs = contractOp.getRhs();
  Value acc = contractOp.getAcc();
  Value mask = maskOp ? maskOp.getMask() : Value();

  SmallVector<AffineMap> newMaps;
  for (AffineMap map : maps) {
    newMaps.push_back(dropIteratorFromMap(map, iterIdx, ctx));
  }
  SmallVector<Attribute> newIterTypes =
      dropIteratorType(iteratorTypes, iterIdx);
  ArrayAttr newMapsAttr = rewriter.getAffineMapArrayAttr(newMaps);
  ArrayAttr newIterTypesAttr = rewriter.getArrayAttr(newIterTypes);

  SmallVector<Value> lhsSlices, rhsSlices, accSlices, maskSlices;
  for (int64_t i = 0; i < dimSize; ++i) {
    if (sliceLhs) {
      lhsSlices.push_back(vector::ExtractOp::create(rewriter, loc, lhs, i));
    }
    if (sliceRhs) {
      rhsSlices.push_back(vector::ExtractOp::create(rewriter, loc, rhs, i));
    }
    accSlices.push_back(vector::ExtractOp::create(rewriter, loc, acc, i));
    if (mask) {
      maskSlices.push_back(vector::ExtractOp::create(rewriter, loc, mask, i));
    }
  }

  SmallVector<Value> results;
  for (int64_t i = 0; i < dimSize; ++i) {
    Value lhsOperand = sliceLhs ? lhsSlices[i] : lhs;
    Value rhsOperand = sliceRhs ? rhsSlices[i] : rhs;

    auto newContract = vector::ContractionOp::create(
        rewriter, loc, lhsOperand, rhsOperand, accSlices[i], newMapsAttr,
        newIterTypesAttr, contractOp.getKind());

    if (mask) {
      Operation *maskedOp =
          mlir::vector::maskOperation(rewriter, newContract, maskSlices[i]);
      results.push_back(maskedOp->getResult(0));
    } else {
      results.push_back(newContract.getResult());
    }
  }

  auto resultType = cast<VectorType>(contractOp.getResultType());
  Value result = ub::PoisonOp::create(rewriter, loc, resultType);
  for (int64_t i = 0; i < dimSize; ++i) {
    result = vector::InsertOp::create(rewriter, loc, results[i], result, i);
  }
  return result;
}

/// Checks shared preconditions for unrolling patterns: at least 2 iterators,
/// all projected permutation maps, and no scalar-result maps.
static LogicalResult checkCommonPreconditions(vector::ContractionOp contractOp,
                                              PatternRewriter &rewriter) {
  ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
  if (iteratorTypes.size() < 2) {
    return rewriter.notifyMatchFailure(contractOp, "need at least 2 iterators");
  }

  SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();

  if (!llvm::all_of(maps,
                    [](AffineMap m) { return m.isProjectedPermutation(); })) {
    return rewriter.notifyMatchFailure(contractOp,
                                       "maps are not projected permutations");
  }

  if (maps[0].getNumResults() == 0 || maps[1].getNumResults() == 0 ||
      maps[2].getNumResults() == 0) {
    return rewriter.notifyMatchFailure(
        contractOp, "one or more maps have no results (pure reduction)");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnrollContractAlongBatchDim
//===----------------------------------------------------------------------===//

/// Unrolls vector.contract along a batch dimension.
///
/// A batch dimension is a parallel iterator that appears in all three
/// operands (lhs, rhs, acc) at their outermost position. This pattern
/// extracts slices along the batch dimension, creates smaller contracts,
/// and assembles the results.
///
/// Example:
/// ```mlir
/// // Before (batch dim b at position 0):
/// %result = vector.contract {
///     indexing_maps = [
///         affine_map<(b, m, n, k) -> (b, m, k)>,
///         affine_map<(b, m, n, k) -> (b, k, n)>,
///         affine_map<(b, m, n, k) -> (b, m, n)>
///     ],
///     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
/// } %A, %B, %C : vector<2x4x3xf32>, vector<2x3x5xf32> into vector<2x4x5xf32>
///
/// // After:
/// %a0 = vector.extract %A[0] : vector<4x3xf32> from vector<2x4x3xf32>
/// %b0 = vector.extract %B[0] : vector<3x5xf32> from vector<2x3x5xf32>
/// %c0 = vector.extract %C[0] : vector<4x5xf32> from vector<2x4x5xf32>
/// %r0 = vector.contract {
///     indexing_maps = [
///         affine_map<(m, n, k) -> (m, k)>,
///         affine_map<(m, n, k) -> (k, n)>,
///         affine_map<(m, n, k) -> (m, n)>
///     ],
///     iterator_types = ["parallel", "parallel", "reduction"]
/// } %a0, %b0, %c0 : ... into vector<4x5xf32>
/// // ... repeat for index 1 ...
/// %init = ub.poison : vector<2x4x5xf32>
/// %res_0 = vector.insert %r0, %init[0]
/// %res_1 = vector.insert %r1, %res_0[1]
/// ```
struct UnrollContractAlongBatchDim final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    if (failed(checkCommonPreconditions(contractOp, rewriter))) {
      return failure();
    }

    ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();

    int64_t lhsIter = maps[0].getDimPosition(0);
    int64_t rhsIter = maps[1].getDimPosition(0);
    int64_t accIter = maps[2].getDimPosition(0);

    if (lhsIter != rhsIter) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "LHS and RHS have different iterators at outermost position");
    }
    if (rhsIter != accIter) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "RHS and ACC have different iterators at outermost position");
    }

    int64_t batchIterIdx = lhsIter;
    if (!vector::isParallelIterator(iteratorTypes[batchIterIdx])) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "outermost iterator is not parallel (not a batch dimension)");
    }

    return unrollContractAlongOutermostDim(
        contractOp, maskOp, rewriter, /*sliceLhs=*/true, /*sliceRhs=*/true);
  }
};

//===----------------------------------------------------------------------===//
// UnrollContractAlongLhsFreeDim
//===----------------------------------------------------------------------===//

/// Unrolls vector.contract along a free LHS dimension.
///
/// A free LHS dimension is a parallel iterator that appears in the LHS
/// and accumulator/result, but NOT in the RHS. This pattern extracts
/// slices along the free dimension from LHS and ACC, while reusing RHS
/// (which is broadcast semantically across this dimension).
///
/// Example:
/// ```mlir
/// // Before (free LHS dim 'm' at iterator position 0):
/// %result = vector.contract {
///     indexing_maps = [
///         affine_map<(m, n, k) -> (m, k)>,  // LHS: 4x8
///         affine_map<(m, n, k) -> (k, n)>,  // RHS: 8x6 (no m)
///         affine_map<(m, n, k) -> (m, n)>   // ACC: 4x6
///     ],
///     iterator_types = ["parallel", "parallel", "reduction"]
/// } %A, %B, %C : vector<4x8xf32>, vector<8x6xf32> into vector<4x6xf32>
///
/// // After:
/// %a0 = vector.extract %A[0] : vector<8xf32> from vector<4x8xf32>
/// %c0 = vector.extract %C[0] : vector<6xf32> from vector<4x6xf32>
/// // Note: B is NOT extracted - reused for all iterations
/// %r0 = vector.contract {
///     indexing_maps = [
///         affine_map<(n, k) -> (k)>,
///         affine_map<(n, k) -> (k, n)>,
///         affine_map<(n, k) -> (n)>
///     ],
///     iterator_types = ["parallel", "reduction"]
/// } %a0, %B, %c0 : vector<8xf32>, vector<8x6xf32> into vector<6xf32>
/// // ... repeat for m=1,2,3, then assemble results ...
/// ```
struct UnrollContractAlongLhsFreeDim final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    if (failed(checkCommonPreconditions(contractOp, rewriter))) {
      return failure();
    }

    ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = maps[0];
    AffineMap rhsMap = maps[1];
    AffineMap accMap = maps[2];

    int64_t lhsIter = lhsMap.getDimPosition(0);
    int64_t accIter = accMap.getDimPosition(0);

    if (lhsIter != accIter) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "LHS and ACC have different iterators at outermost position");
    }

    int64_t freeLhsIterIdx = lhsIter;
    MLIRContext *ctx = rewriter.getContext();
    if (rhsMap.getResultPosition(getAffineDimExpr(freeLhsIterIdx, ctx))
            .has_value()) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "outermost LHS iterator also appears in RHS (not a free LHS dim)");
    }

    if (!vector::isParallelIterator(iteratorTypes[freeLhsIterIdx])) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "outermost LHS iterator is not parallel (not a free dimension)");
    }

    return unrollContractAlongOutermostDim(
        contractOp, maskOp, rewriter, /*sliceLhs=*/true, /*sliceRhs=*/false);
  }
};

//===----------------------------------------------------------------------===//
// UnrollContractAlongRhsFreeDim
//===----------------------------------------------------------------------===//

/// Unrolls vector.contract along a free RHS dimension.
///
/// A free RHS dimension is a parallel iterator that appears in the RHS
/// and accumulator/result, but NOT in the LHS. This pattern extracts
/// slices along the free dimension from RHS and ACC, while reusing LHS
/// (which is broadcast semantically across this dimension).
///
/// Example:
/// ```mlir
/// // Before (free RHS dim 'n' at iterator position 0):
/// %result = vector.contract {
///     indexing_maps = [
///         affine_map<(n, m, k) -> (m, k)>,  // LHS: 4x8 (no n)
///         affine_map<(n, m, k) -> (n, k)>,  // RHS: 6x8
///         affine_map<(n, m, k) -> (n, m)>   // ACC: 6x4
///     ],
///     iterator_types = ["parallel", "parallel", "reduction"]
/// } %A, %B, %C : vector<4x8xf32>, vector<6x8xf32> into vector<6x4xf32>
///
/// // After:
/// %b0 = vector.extract %B[0] : vector<8xf32> from vector<6x8xf32>
/// %c0 = vector.extract %C[0] : vector<4xf32> from vector<6x4xf32>
/// // Note: A is NOT extracted - reused for all iterations
/// %r0 = vector.contract {
///     indexing_maps = [
///         affine_map<(m, k) -> (m, k)>,
///         affine_map<(m, k) -> (k)>,
///         affine_map<(m, k) -> (m)>
///     ],
///     iterator_types = ["parallel", "reduction"]
/// } %A, %b0, %c0 : vector<4x8xf32>, vector<8xf32> into vector<4xf32>
/// // ... repeat for n=1..5, then assemble results ...
/// ```
struct UnrollContractAlongRhsFreeDim final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    if (failed(checkCommonPreconditions(contractOp, rewriter))) {
      return failure();
    }

    ArrayAttr iteratorTypes = contractOp.getIteratorTypes();
    SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
    AffineMap lhsMap = maps[0];
    AffineMap rhsMap = maps[1];
    AffineMap accMap = maps[2];

    int64_t rhsIter = rhsMap.getDimPosition(0);
    int64_t accIter = accMap.getDimPosition(0);

    if (rhsIter != accIter) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "RHS and ACC have different iterators at outermost position");
    }

    int64_t freeRhsIterIdx = rhsIter;
    MLIRContext *ctx = rewriter.getContext();
    if (lhsMap.getResultPosition(getAffineDimExpr(freeRhsIterIdx, ctx))
            .has_value()) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "outermost RHS iterator also appears in LHS (not a free RHS dim)");
    }

    if (!vector::isParallelIterator(iteratorTypes[freeRhsIterIdx])) {
      return rewriter.notifyMatchFailure(
          contractOp,
          "outermost RHS iterator is not parallel (not a free dimension)");
    }

    return unrollContractAlongOutermostDim(
        contractOp, maskOp, rewriter, /*sliceLhs=*/false, /*sliceRhs=*/true);
  }
};

//===----------------------------------------------------------------------===//
// Test pass
//===----------------------------------------------------------------------===//

struct TestContractUnrollingPatternsPass final
    : impl::TestContractUnrollingPatternsPassBase<
          TestContractUnrollingPatternsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateContractUnrollingPatterns(patterns, /*benefit=*/1);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateContractUnrollingPatterns(RewritePatternSet &patterns,
                                       PatternBenefit benefit) {
  patterns.add<UnrollContractAlongBatchDim>(patterns.getContext(), benefit);
  patterns.add<UnrollContractAlongLhsFreeDim>(patterns.getContext(), benefit);
  patterns.add<UnrollContractAlongRhsFreeDim>(patterns.getContext(), benefit);
}

} // namespace mlir::iree_compiler
