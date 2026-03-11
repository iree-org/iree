// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Eliminates redundant SMFMAC accumulator expand/collapse pairs emitted by
// VDMFMA lowering. After LowerIREEGPUOps, each VDMFMA lowers to:
//
//   expand(vector<2xf32> → vector<4xf32>) → SMFMAC chain → collapse(→
//   vector<2xf32>)
//
// When K-unrolling > 1, the collapse from one SMFMAC chain feeds the expand of
// the next chain within the same loop iteration. The collapse→expand pair is
// mathematically redundant because SMFMAC accumulation is additive:
//   collapse([d0, d1, d2, d3]) = [d0+d1, d2+d3]
//   expand([d0+d1, d2+d3]) = [d0+d1, 0, d2+d3, 0]
// Since SMFMAC adds to the accumulator, we can pass the 4-element form
// directly between chains. The final collapse produces equivalent results
// (modulo floating-point associativity, which is acceptable for this use case).
//
// Additionally, for the loop-carried accumulator dependency, this pass hoists
// the expand before the loop and the collapse after the loop, keeping the
// accumulator in 4-element expanded form throughout the loop.
//
// This is a targeted hack for VDMFMA skinny GEMM (M=8) performance. The
// pattern match depends on the exact IR structure emitted by buildVDMFMAOps.

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-rocdl-hoist-smfmac-acc-expand"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLHOISTSMFMACACCEXPANDPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Returns true if `val` is a splat zero constant (float or integer).
static bool isZeroSplatConstant(Value val) {
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!attr || !attr.isSplat()) {
    return false;
  }
  Type elemTy = attr.getElementType();
  if (isa<FloatType>(elemTy)) {
    return attr.getSplatValue<APFloat>().isZero();
  }
  if (isa<IntegerType>(elemTy)) {
    return attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

/// Matches the expand pattern: vector.shuffle %src, %zero [0, 2, 1, 3]
/// where %zero is a zero constant. Returns the source value if matched.
static Value matchExpandSource(vector::ShuffleOp shuffle) {
  ArrayRef<int64_t> mask = shuffle.getMask();
  if (mask.size() != 4 || mask[0] != 0 || mask[1] != 2 || mask[2] != 1 ||
      mask[3] != 3) {
    return nullptr;
  }
  // V1 is the source, V2 must be zero.
  if (!isZeroSplatConstant(shuffle.getV2())) {
    return nullptr;
  }
  // V1 must be vector<2xT>.
  auto srcTy = dyn_cast<VectorType>(shuffle.getV1().getType());
  if (!srcTy || srcTy.getRank() != 1 || srcTy.getDimSize(0) != 2) {
    return nullptr;
  }
  return shuffle.getV1();
}

/// Matches the collapse pattern:
///   %evens = vector.shuffle %src, %src [0, 2]
///   %odds = vector.shuffle %src, %src [1, 3]
///   %result = arith.addf %evens, %odds   (or arith.addi)
/// Returns the 4-element source value if matched, nullptr otherwise.
static Value matchCollapseSource(Value collapseResult) {
  Operation *addOp = collapseResult.getDefiningOp();
  if (!addOp || (!isa<arith::AddFOp>(addOp) && !isa<arith::AddIOp>(addOp))) {
    return nullptr;
  }

  auto evensShuffle = dyn_cast_if_present<vector::ShuffleOp>(
      addOp->getOperand(0).getDefiningOp());
  auto oddsShuffle = dyn_cast_if_present<vector::ShuffleOp>(
      addOp->getOperand(1).getDefiningOp());
  if (!evensShuffle || !oddsShuffle) {
    return nullptr;
  }

  // Check masks: evens = [0, 2], odds = [1, 3].
  ArrayRef<int64_t> evensMask = evensShuffle.getMask();
  ArrayRef<int64_t> oddsMask = oddsShuffle.getMask();
  if (evensMask.size() != 2 || evensMask[0] != 0 || evensMask[1] != 2) {
    return nullptr;
  }
  if (oddsMask.size() != 2 || oddsMask[0] != 1 || oddsMask[1] != 3) {
    return nullptr;
  }

  // Both shuffles must use the same source as both V1 and V2.
  Value source = evensShuffle.getV1();
  if (source != evensShuffle.getV2() || source != oddsShuffle.getV1() ||
      source != oddsShuffle.getV2()) {
    return nullptr;
  }

  // The source must be vector<4xT>.
  auto sourceTy = dyn_cast<VectorType>(source.getType());
  if (!sourceTy || sourceTy.getRank() != 1 || sourceTy.getDimSize(0) != 4) {
    return nullptr;
  }

  return source;
}

/// Peephole: Eliminate expand(collapse(X)) → X.
///
/// When a collapse result feeds directly into an expand, the pair is redundant
/// in the context of SMFMAC accumulation. This pass replaces the expand result
/// with the original 4-element source of the collapse.
///
/// Pattern:
///   %evens = vector.shuffle %X, %X [0, 2]
///   %odds = vector.shuffle %X, %X [1, 3]
///   %collapsed = arith.addf %evens, %odds : vector<2xf32>
///   %expanded = vector.shuffle %collapsed, %zero [0, 2, 1, 3] : vector<4xf32>
///
/// Replaced with: all uses of %expanded → %X
static int64_t eliminateExpandCollapsePairs(IRRewriter &rewriter,
                                            FunctionOpInterface funcOp) {
  int64_t count = 0;
  SmallVector<vector::ShuffleOp> expandOps;

  funcOp.walk([&](vector::ShuffleOp shuffle) {
    Value expandSrc = matchExpandSource(shuffle);
    if (!expandSrc) {
      return;
    }
    // Check if the expand source is a collapse result.
    Value collapseSource = matchCollapseSource(expandSrc);
    if (!collapseSource) {
      return;
    }
    expandOps.push_back(shuffle);
  });

  for (vector::ShuffleOp expandOp : expandOps) {
    Value expandSrc = matchExpandSource(expandOp);
    Value collapseSource = matchCollapseSource(expandSrc);
    if (!collapseSource) {
      continue;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "SMFMAC acc: eliminating expand(collapse()) pair\n");
    rewriter.replaceAllUsesWith(expandOp.getResult(), collapseSource);
    ++count;
  }

  return count;
}

/// Emits the collapse operation:
///   evens = shuffle %acc, %acc [0, 2]
///   odds = shuffle %acc, %acc [1, 3]
///   result = addf(evens, odds)  or  addi(evens, odds)
static Value emitCollapse(IRRewriter &rewriter, Location loc, Value acc) {
  auto accType = cast<VectorType>(acc.getType());
  Type elementType = accType.getElementType();
  Value evens = vector::ShuffleOp::create(rewriter, loc, acc, acc,
                                          ArrayRef<int64_t>{0, 2});
  Value odds = vector::ShuffleOp::create(rewriter, loc, acc, acc,
                                         ArrayRef<int64_t>{1, 3});
  if (isa<FloatType>(elementType)) {
    return arith::AddFOp::create(rewriter, loc, evens, odds);
  }
  return arith::AddIOp::create(rewriter, loc, evens, odds);
}

/// Emits the expand operation: vector.shuffle %acc, %zero [0, 2, 1, 3].
static Value emitExpand(IRRewriter &rewriter, Location loc, Value acc) {
  auto accType = cast<VectorType>(acc.getType());
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(accType));
  return vector::ShuffleOp::create(rewriter, loc, acc, zero,
                                   ArrayRef<int64_t>{0, 2, 1, 3});
}

/// Information about an expand→...→collapse chain connected to a loop
/// iter_arg through extract/shape_cast operations.
struct LoopAccCandidate {
  int64_t iterArgIndex;
  // Extract chain: iter_arg → extract → shape_cast → expand
  vector::ExtractOp extractOp;
  vector::ShapeCastOp extractShapeCast;
  vector::ShuffleOp expandOp;
  // Collapse chain: collapse → shape_cast → insert_strided_slice
  Operation *collapseAdd;
  vector::ShapeCastOp insertShapeCast;
  vector::InsertStridedSliceOp insertOp;
  // The 4-element value that feeds the collapse.
  Value expandedResult;
  // The extract indices for reinserting after the loop.
  SmallVector<int64_t> extractIndices;
};

/// Try to match the extract→shape_cast→expand chain from the iter_arg to a
/// particular element. Returns the candidate if matched.
static std::optional<LoopAccCandidate>
matchLoopAccCandidate(int64_t iterArgIndex, BlockArgument iterArg,
                      vector::ShuffleOp expandOp) {
  // The expand source should be a shape_cast.
  Value expandSrc = expandOp.getV1();
  auto shapeCast =
      dyn_cast_if_present<vector::ShapeCastOp>(expandSrc.getDefiningOp());
  if (!shapeCast) {
    return std::nullopt;
  }

  // The shape_cast source should be a vector.extract from the iter_arg.
  auto extractOp = dyn_cast_if_present<vector::ExtractOp>(
      shapeCast.getSource().getDefiningOp());
  if (!extractOp || extractOp.getSource() != iterArg) {
    return std::nullopt;
  }

  // Get static extract indices.
  SmallVector<int64_t> indices;
  for (OpFoldResult ofr : extractOp.getMixedPosition()) {
    auto constIdx = getConstantIntValue(ofr);
    if (!constIdx) {
      return std::nullopt;
    }
    indices.push_back(*constIdx);
  }

  LoopAccCandidate candidate;
  candidate.iterArgIndex = iterArgIndex;
  candidate.extractOp = extractOp;
  candidate.extractShapeCast = shapeCast;
  candidate.expandOp = expandOp;
  candidate.extractIndices = std::move(indices);
  return candidate;
}

/// Walk backward through the insert_strided_slice chain from the yield value,
/// matching each insert's source to a collapse→shape_cast pattern. Pairs each
/// matched insert with a candidate by comparing extract indices to insert
/// offsets.
///
/// Expected IR at the yield end (per N-tile):
///   %col = arith.addf(shuffle %src [0,2], shuffle %src [1,3])
///   %shaped = vector.shape_cast %col : vector<2xf32> to vector<1x1x2x1xf32>
///   %ins = vector.insert_strided_slice %shaped, %base {offsets = [0, i, 0, 0]}
static bool matchCollapseChains(scf::ForOp forOp, int64_t iterArgIndex,
                                SmallVectorImpl<LoopAccCandidate> &candidates) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldVal = yieldOp.getOperand(iterArgIndex);

  // Walk backward through insert_strided_slice chain.
  SmallVector<vector::InsertStridedSliceOp> inserts;
  while (auto insertOp =
             yieldVal.getDefiningOp<vector::InsertStridedSliceOp>()) {
    inserts.push_back(insertOp);
    yieldVal = insertOp.getDest();
  }

  // The base should be a zero constant (the chain rebuilds from scratch each
  // iteration, not modifying the iter_arg in-place).
  if (!isZeroSplatConstant(yieldVal)) {
    return false;
  }

  // Must have the same number of inserts as candidates.
  if (inserts.size() != candidates.size()) {
    return false;
  }

  for (vector::InsertStridedSliceOp insertOp : inserts) {
    // The source should be shape_cast(collapse_result).
    auto shapeCast =
        insertOp.getValueToStore().getDefiningOp<vector::ShapeCastOp>();
    if (!shapeCast) {
      return false;
    }

    Value collapseResult = shapeCast.getSource();
    Value collapseSource = matchCollapseSource(collapseResult);
    if (!collapseSource) {
      return false;
    }

    // Get insert offsets. Match against candidate extract indices:
    // extract indices [i0, i1] ↔ insert offsets [i0, i1, 0, 0].
    SmallVector<int64_t> offsets = getI64SubArray(insertOp.getOffsets());
    bool matched = false;
    for (LoopAccCandidate &cand : candidates) {
      if (cand.expandedResult) {
        continue; // Already matched.
      }

      ArrayRef<int64_t> indices = cand.extractIndices;
      if (offsets.size() < indices.size()) {
        continue;
      }

      bool isMatch = true;
      for (size_t i = 0; i < indices.size(); ++i) {
        if (offsets[i] != indices[i]) {
          isMatch = false;
          break;
        }
      }
      for (size_t i = indices.size(); i < offsets.size(); ++i) {
        if (offsets[i] != 0) {
          isMatch = false;
          break;
        }
      }

      if (isMatch) {
        cand.collapseAdd = collapseResult.getDefiningOp();
        cand.insertShapeCast = shapeCast;
        cand.insertOp = insertOp;
        cand.expandedResult = collapseSource;
        matched = true;
        break;
      }
    }

    if (!matched) {
      return false;
    }
  }

  return llvm::all_of(
      candidates, [](const LoopAccCandidate &c) { return !!c.expandedResult; });
}

/// Hoist expand/collapse across the loop boundary by replacing the packed
/// iter_arg (vector<1x2x2x1xf32>) with individual expanded iter_args
/// (vector<4xf32>, one per N-tile). The accumulator stays in 4-element
/// expanded form throughout the loop; expand happens once before the loop
/// and collapse happens once after.
///
/// Before:
///   scf.for iter_args(%packed = %init) -> (vector<1x2x2x1xf32>) {
///     %ext = extract %packed[0,i] → shape_cast → expand → SMFMAC chain
///     → collapse → shape_cast → insert_strided_slice
///     yield %packed_result
///   }
///
/// After:
///   %exp0 = expand(extract %init[0,0])
///   %exp1 = expand(extract %init[0,1])
///   %r:2 = scf.for iter_args(%a0 = %exp0, %a1 = %exp1) -> (v4, v4) {
///     SMFMAC chain using %a0, %a1 directly
///     yield %last_smfmac_0, %last_smfmac_1
///   }
///   %packed = collapse(%r#0) → shape_cast → insert [0,0,0,0]
///             collapse(%r#1) → shape_cast → insert [0,1,0,0]
static LogicalResult tryHoistLoopBoundary(IRRewriter &rewriter,
                                          scf::ForOp forOp) {
  // Only handle loops with exactly one iter_arg for now.
  if (forOp.getInitArgs().size() != 1) {
    return failure();
  }

  BlockArgument iterArg = forOp.getRegionIterArg(0);
  VectorType packedType = dyn_cast<VectorType>(iterArg.getType());
  if (!packedType) {
    return failure();
  }

  // Step 1: Every use of the iter_arg must be an extract→shape_cast→expand
  // chain. If any use doesn't match, bail.
  SmallVector<LoopAccCandidate> candidates;
  for (Operation *user : iterArg.getUsers()) {
    auto extractOp = dyn_cast<vector::ExtractOp>(user);
    if (!extractOp) {
      return failure();
    }
    if (!extractOp->hasOneUse()) {
      return failure();
    }
    auto shapeCast = dyn_cast<vector::ShapeCastOp>(*extractOp->user_begin());
    if (!shapeCast || !shapeCast->hasOneUse()) {
      return failure();
    }
    auto shuffleOp = dyn_cast<vector::ShuffleOp>(*shapeCast->user_begin());
    if (!shuffleOp || !matchExpandSource(shuffleOp)) {
      return failure();
    }

    std::optional<LoopAccCandidate> candidate =
        matchLoopAccCandidate(/*iterArgIndex=*/0, iterArg, shuffleOp);
    if (!candidate) {
      return failure();
    }
    candidates.push_back(*candidate);
  }

  if (candidates.empty()) {
    return failure();
  }

  // Step 2: Match collapse chains at the yield end.
  if (!matchCollapseChains(forOp, /*iterArgIndex=*/0, candidates)) {
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "SMFMAC acc: hoisting " << candidates.size()
                          << " expand/collapse pairs out of loop\n");

  Location loc = forOp.getLoc();

  // Snapshot candidate info before IR mutation.
  struct CandidateInfo {
    SmallVector<int64_t> extractIndices;
    Value expandedResult;
    VectorType insertSrcType;
    SmallVector<int64_t> insertOffsets;
    SmallVector<int64_t> insertStrides;
  };
  SmallVector<CandidateInfo> infos;
  for (LoopAccCandidate &cand : candidates) {
    CandidateInfo info;
    info.extractIndices = cand.extractIndices;
    info.expandedResult = cand.expandedResult;
    info.insertSrcType = cand.insertOp.getSourceVectorType();
    info.insertOffsets = getI64SubArray(cand.insertOp.getOffsets());
    info.insertStrides = getI64SubArray(cand.insertOp.getStrides());
    infos.push_back(std::move(info));
  }

  // Step 3: Build expanded init values before the loop.
  rewriter.setInsertionPoint(forOp);
  Value initArg = forOp.getInitArgs()[0];
  SmallVector<Value> expandedInits;
  for (CandidateInfo &info : infos) {
    Value ext =
        vector::ExtractOp::create(rewriter, loc, initArg, info.extractIndices);
    VectorType extType = cast<VectorType>(ext.getType());
    VectorType flatType =
        VectorType::get({extType.getNumElements()}, extType.getElementType());
    Value flat = vector::ShapeCastOp::create(rewriter, loc, flatType, ext);
    expandedInits.push_back(emitExpand(rewriter, loc, flat));
  }

  // Step 4: Create new loop with expanded iter_args.
  auto newForOp =
      scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), expandedInits);

  // Step 5: Transplant the old loop body into the new loop.
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();

  // Replace old block arg uses with new values. This temporarily creates
  // cross-block SSA references, resolved when ops are spliced below.
  rewriter.replaceAllUsesWith(oldBody->getArgument(0),
                              newForOp.getInductionVar());

  // The old packed iter_arg becomes dead after expand RAUW below. Provide a
  // dummy zero so any residual extract→shape_cast ops become trivially dead.
  rewriter.setInsertionPointToStart(newBody);
  Value dummy = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getZeroAttr(packedType));
  rewriter.replaceAllUsesWith(oldBody->getArgument(1), dummy);

  // Replace expand results with new block args (the expanded accumulators).
  for (auto [i, cand] : llvm::enumerate(candidates)) {
    rewriter.replaceAllUsesWith(cand.expandOp.getResult(),
                                newForOp.getRegionIterArg(i));
  }

  // Splice all non-terminator ops from old body into new body (which is
  // currently empty — ForOp::build does not create a terminator when
  // initArgs are provided). Leave old body's yield in place so the old
  // forOp remains valid when erased.
  auto &newOps = newBody->getOperations();
  auto &oldOps = oldBody->getOperations();
  Operation *oldYield = oldBody->getTerminator();
  newOps.splice(newOps.end(), oldOps, oldOps.begin(), oldYield->getIterator());

  // Step 6: Create a new yield with the expanded (4-element) SMFMAC results.
  rewriter.setInsertionPointToEnd(newBody);
  SmallVector<Value> newYieldValues;
  for (CandidateInfo &info : infos) {
    newYieldValues.push_back(info.expandedResult);
  }
  scf::YieldOp::create(rewriter, loc, newYieldValues);

  // Step 7: After the loop, collapse each result and repack.
  rewriter.setInsertionPointAfter(newForOp);
  Value packedResult = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(packedType));

  for (auto [i, info] : llvm::enumerate(infos)) {
    Value result = newForOp.getResult(i);
    Value collapsed = emitCollapse(rewriter, loc, result);
    Value shaped = vector::ShapeCastOp::create(rewriter, loc,
                                               info.insertSrcType, collapsed);
    packedResult = vector::InsertStridedSliceOp::create(
        rewriter, loc, shaped, packedResult, info.insertOffsets,
        info.insertStrides);
  }

  // Step 8: Replace old loop result and erase old loop.
  rewriter.replaceAllUsesWith(forOp.getResult(0), packedResult);
  rewriter.eraseOp(forOp);

  return success();
}

struct ROCDLHoistSMFMACAccExpandPass final
    : impl::ROCDLHoistSMFMACAccExpandPassBase<ROCDLHoistSMFMACAccExpandPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Phase 1: Eliminate intra-iteration expand(collapse(X)) → X pairs.
    int64_t pairsEliminated = eliminateExpandCollapsePairs(rewriter, funcOp);

    if (pairsEliminated > 0) {
      LLVM_DEBUG(llvm::dbgs() << "SMFMAC acc: eliminated " << pairsEliminated
                              << " expand(collapse()) pairs\n");
    }

    // Phase 2: Try loop boundary hoisting (future enhancement).
    SmallVector<scf::ForOp> loops;
    funcOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      (void)tryHoistLoopBoundary(rewriter, forOp);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
