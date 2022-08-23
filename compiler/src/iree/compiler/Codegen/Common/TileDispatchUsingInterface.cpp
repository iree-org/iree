// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "tile-dispatch-using-interface"

namespace mlir {
namespace iree_compiler {

/// Method to check if a value is zero
static bool isZero(Value val) {
  OpFoldResult ofr = getAsOpFoldResult(val);
  return isConstantIntValue(ofr, 0);
}

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<unsigned> fillInterchangeVector(
    ArrayRef<unsigned> interchangeVector, size_t iterationDomainSize) {
  SmallVector<unsigned> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<unsigned>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Helper method to apply permutation to a vector
template <typename T>
static SmallVector<T> applyPermutationToVector(const SmallVector<T> &vector,
                                               ArrayRef<unsigned> interchange) {
  assert(interchange.size() == vector.size());
  return llvm::to_vector(
      llvm::map_range(interchange, [&](unsigned val) { return vector[val]; }));
}
/// Helper method to apply to invert a permutation.
static SmallVector<unsigned> invertPermutationVector(
    ArrayRef<unsigned> interchange) {
  SmallVector<unsigned> inversion(interchange.size());
  for (auto pos : llvm::enumerate(interchange)) {
    inversion[pos.value()] = pos.index();
  }
  return inversion;
}
/// Method to check if an interchange vector is a permutation.
static bool isPermutation(ArrayRef<unsigned> interchange) {
  llvm::SmallDenseSet<unsigned, 4> seenVals;
  for (auto val : interchange) {
    if (seenVals.count(val)) return false;
    seenVals.insert(val);
  }
  return seenVals.size() == interchange.size();
}

/// Given the `lb` and `step` of a loop, return the lower bound and step to use
/// for a distributed loop. Replace the iteration domain to
/// - lb_partitioned = lb + procId * step
/// - step_partitioned = step * nprocs
static std::tuple<Value, Value> getDistributeLBAndStep(OpBuilder &b,
                                                       Location loc, Value lb,
                                                       Value step, Value procId,
                                                       Value nprocs) {
  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  auto offsetMap = AffineMap::get(0, 3, {s0 + s1 * s2});
  auto stepMap = AffineMap::get(0, 2, {s0 * s1});
  Value distributeLB =
      makeComposedAffineApply(b, loc, offsetMap, ValueRange{lb, procId, step});
  Value distributeStep =
      makeComposedAffineApply(b, loc, stepMap, ValueRange{step, nprocs});
  return {distributeLB, distributeStep};
}

//===----------------------------------------------------------------------===//
// TileDispatchUsingSCFForOp pattern implementation.
//===----------------------------------------------------------------------===//

/// This implementation mirrors the implementation in
/// include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h and
/// lib/Dialect/SCF/Transforms/TileUsingInterface.cpp. It is adapted to do
/// distribution and also use `flow.dispatch.tensor.load/store` instead of
/// `tensor.extract_slice/insert_slice`.

namespace {
// Till `scf.for` can take an `OpFoldResult` for lb, ub and step, use of `Range`
// with `OpFoldResult` causes lots of `OpFoldResult` -> `Value` conversion.
// This struct is a mirror of `Range` with `Value` type fields.
struct RangeVal {
  Value offset;
  Value size;
  Value stride;
};
}  // namespace

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the
///   tile processed within the inner most loop.
static SmallVector<scf::ForOp> generateTileLoopNest(
    OpBuilder &builder, Location loc, ArrayRef<RangeVal> loopRanges,
    ArrayRef<Value> tileSizeVals,
    ArrayRef<linalg::DistributionMethod> distributionMethod,
    ArrayRef<linalg::ProcInfo> procInfo, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  assert(loopRanges.size() == distributionMethod.size() &&
         "expected as many entries in distribution method list as number of "
         "loop ranges");
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp> loops;
  offsets.resize(loopRanges.size());
  sizes.resize(loopRanges.size());

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable
  // of the tiled loop.
  AffineExpr s0, s1, d0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());
  auto createBoundedTileSize = [&](Value iv, Value tileSize,
                                   Value size) -> OpFoldResult {
    if (isConstantIntValue(getAsOpFoldResult(tileSize), 1)) {
      return builder.getIndexAttr(1);
    }
    return makeComposedFoldedAffineMin(
        builder, loc, minMap, ArrayRef<OpFoldResult>{iv, tileSize, size});
  };

  unsigned procDim = 0;
  for (auto loopRange : llvm::enumerate(loopRanges)) {
    auto index = loopRange.index();
    Value lb = loopRange.value().offset;
    Value ub = loopRange.value().size;
    Value step = tileSizeVals[index];

    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (matchPattern(tileSizeVals[index], m_Zero())) {
      offsets[index] = lb;
      sizes[index] = ub;
      continue;
    }

    auto method = distributionMethod[index];
    if (method != linalg::DistributionMethod::None) {
      std::tie(lb, step) = getDistributeLBAndStep(builder, loc, lb, step,
                                                  procInfo[procDim].procId,
                                                  procInfo[procDim].nprocs);
      procDim++;
    }

    if (method == linalg::DistributionMethod::CyclicNumProcsEqNumIters) {
      offsets[index] = getAsOpFoldResult(lb);
      sizes[index] = createBoundedTileSize(lb, tileSizeVals[index], ub);
      continue;
    }

    auto loop = builder.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          sizes[index] = createBoundedTileSize(iv, tileSizeVals[index], ub);
          builder.create<scf::YieldOp>(loc);
        });
    offsets[index] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return loops;
}

namespace {

/// Result of the tiled operation.
struct TilingResult {
  Operation *tiledOp;
  SmallVector<scf::ForOp> loops;
};

/// Pattern to tile an op that implements the `TilingInterface` using
/// `scf.for`+`flow.dispatch.tensor.load/store` for iterating over the tiles.
struct TileDispatchUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {
  /// Construct a generic pattern applied to all TilingInterface ops.
  TileDispatchUsingSCFForOp(MLIRContext *context,
                            linalg::LinalgTilingOptions options,
                            linalg::LinalgTransformationFilter filter,
                            PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        options(std::move(options)),
        filter(std::move(filter)) {}

  /// Construct a generic pattern applied to `opName`.
  TileDispatchUsingSCFForOp(StringRef opName, MLIRContext *context,
                            linalg::LinalgTilingOptions options,
                            linalg::LinalgTransformationFilter filter,
                            PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        options(std::move(options)),
        filter(std::move(filter)) {}

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<TilingResult> returningMatchAndRewrite(
      TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

 private:
  /// Options to control tiling.
  linalg::LinalgTilingOptions options;

  /// Filter to control transformation.
  linalg::LinalgTransformationFilter filter;
};
}  // namespace

FailureOr<TilingResult> TileDispatchUsingSCFForOp::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  // Check for the filter and abort if needed.
  if (failed(filter.checkAndNotify(rewriter, op))) {
    return failure();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomainOfr = op.getIterationDomain(rewriter);
  Location loc = op.getLoc();
  size_t numLoops = iterationDomainOfr.size();
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }
  auto iterationDomain =
      llvm::to_vector(llvm::map_range(iterationDomainOfr, [&](Range r) {
        return RangeVal{
            getValueOrCreateConstantIndexOp(rewriter, loc, r.offset),
            getValueOrCreateConstantIndexOp(rewriter, loc, r.size),
            getValueOrCreateConstantIndexOp(rewriter, loc, r.stride)};
      }));

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<Value> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < numLoops) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }
  tileSizeVector.resize(numLoops);

  TilingResult tilingResult;
  SmallVector<OpFoldResult> offsets, sizes;
  {
    // If there is an interchange specified, permute the iteration domain and
    // the tile sizes.
    SmallVector<unsigned> interchangeVector;
    if (!options.interchangeVector.empty()) {
      interchangeVector = fillInterchangeVector(options.interchangeVector,
                                                iterationDomain.size());
    }
    if (!interchangeVector.empty()) {
      if (!isPermutation(interchangeVector)) {
        return rewriter.notifyMatchFailure(
            op,
            "invalid intechange vector, not a permutation of the entire "
            "iteration space");
      }

      iterationDomain =
          applyPermutationToVector(iterationDomain, interchangeVector);
      tileSizeVector =
          applyPermutationToVector(tileSizeVector, interchangeVector);
    }

    // If there is distribution specified, adjust the loop ranges. Note that
    // permutation happens *before* distribution.
    SmallVector<linalg::DistributionMethod> distributionMethods(
        iterationDomain.size(), linalg::DistributionMethod::None);
    SmallVector<linalg::ProcInfo> procInfo;
    if (options.distribution) {
      SmallVector<StringRef> iteratorTypes = op.getLoopIteratorTypes();

      // The parallel loops that are tiled are partitionable loops.
      SmallVector<Range> parallelLoopRanges;
      SmallVector<unsigned> partitionedLoopIds;
      for (auto iteratorType : llvm::enumerate(iteratorTypes)) {
        if (iteratorType.value() == getParallelIteratorTypeName() &&
            !isZero(tileSizeVector[iteratorType.index()])) {
          parallelLoopRanges.push_back(
              iterationDomainOfr[iteratorType.index()]);
          partitionedLoopIds.push_back(iteratorType.index());
        }
      }

      // Query the callback to get the {procId, nprocs} to use.
      procInfo =
          options.distribution->procInfo(rewriter, loc, parallelLoopRanges);

      for (auto it : llvm::enumerate(partitionedLoopIds)) {
        distributionMethods[it.value()] =
            procInfo[it.index()].distributionMethod;
      }
    }

    // 3. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    tilingResult.loops =
        generateTileLoopNest(rewriter, loc, iterationDomain, tileSizeVector,
                             distributionMethods, procInfo, offsets, sizes);

    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      offsets = applyPermutationToVector(offsets, inversePermutation);
      sizes = applyPermutationToVector(sizes, inversePermutation);
    }

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "LoopNest shell :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });

    // 4. Generate the tiled implementation within the inner most loop.
    if (!tilingResult.loops.empty())
      rewriter.setInsertionPoint(
          tilingResult.loops.back().getBody()->getTerminator());
    SmallVector<Operation *> tiledImplementation = op.getTiledImplementation(
        rewriter, op.getDestinationOperands(rewriter), offsets, sizes, true);
    if (tiledImplementation.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected tiled implementation to return a single op");
    }
    tilingResult.tiledOp = tiledImplementation[0];

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "After tiled implementation :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });
  }

  // Update the filter.
  filter.replaceLinalgTransformationFilter(rewriter, tilingResult.tiledOp);

  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
    return tilingResult;
  }

  // Rewrite all `flow.dispatch.workgroups`. First find all the stores.
  SmallVector<IREE::Flow::DispatchTensorStoreOp> storeOps;
  for (OpOperand &use : op->getUses()) {
    auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(use.getOwner());
    if (storeOp && storeOp.getValue() == use.get()) {
      storeOps.push_back(storeOp);
    }
  }
  for (auto storeOp : storeOps) {
    OpResult result = storeOp.getValue().cast<OpResult>();
    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(op.getResultTilePosition(rewriter, result.getResultNumber(),
                                        offsets, sizes, resultOffsets,
                                        resultSizes))) {
      return rewriter.notifyMatchFailure(
          op, "failed to rewrite destructive update");
    }
    SmallVector<OpFoldResult> resultStrides(resultOffsets.size(),
                                            rewriter.getIndexAttr(1));

    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(IREE::Flow::foldOffsetsSizesAndStrides(
            rewriter, loc, storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
            storeOp.getMixedStrides(), storeOp.getDroppedDims(), resultOffsets,
            resultSizes, resultStrides, combinedOffsets, combinedSizes,
            combinedStrides))) {
      return rewriter.notifyMatchFailure(
          op, "failed to create tiled flow.dispatch.tensor.store op");
    }

    rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
        loc, tilingResult.tiledOp->getResult(result.getResultNumber()),
        storeOp.getTarget(), storeOp.getTargetDims(), combinedOffsets,
        combinedSizes, combinedStrides);
  }
  for (auto storeOp : storeOps) {
    rewriter.eraseOp(storeOp);
  }
  return tilingResult;
}

namespace {

//===----------------------------------------------------------------------===//
// SwapExtractSliceWithTiledProducer
//===----------------------------------------------------------------------===//

/// Pattern to swap a `tilinginterface op` -> `tensor.extract_slice` with
/// `tensor.extract_slice` of operands of the op -> tiled `tilinginterface op`.
struct SwapExtractSliceWithTiledProducer
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    OpResult producer = sliceOp.getSource().dyn_cast<OpResult>();
    if (!producer) {
      return rewriter.notifyMatchFailure(sliceOp, "source uses bb arg");
    }
    FailureOr<Value> tiledProducer =
        tensor::replaceExtractSliceWithTiledProducer(rewriter, sliceOp,
                                                     producer);
    if (failed(tiledProducer)) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, tiledProducer.value());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SwapExtractSliceWithDispatchTensorLoad
//===----------------------------------------------------------------------===//

/// Pattern to swap `flow.dispatch.tensor.load` -> `tensor.extract_slice` with
/// `flow.dispatch.tensor.load` of the slice.
struct SwapExtractSliceWithDispatchTensorLoad
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto loadOp =
        sliceOp.getSource().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp) return failure();

    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(IREE::Flow::foldOffsetsSizesAndStrides(
            rewriter, loadOp.getLoc(), loadOp, sliceOp, loadOp.getDroppedDims(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return rewriter.notifyMatchFailure(
          sliceOp, "failed to fold offsets, sizes and strides with load op");
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        sliceOp, sliceOp.getType(), loadOp.getSource(), loadOp.getSourceDims(),
        combinedOffsets, combinedSizes, combinedStrides);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SwapExtractSliceWithInitTensor
//===----------------------------------------------------------------------===//

/// Pattern to swap `init_tensor` -> `tensor.extract_slice` with
/// `init_tensor` of the slice.
struct SwapExtractSliceWithInitTensor
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto initTensorOp =
        sliceOp.getSource().getDefiningOp<linalg::InitTensorOp>();
    if (!initTensorOp) return failure();

    SmallVector<OpFoldResult> mixedSizes = sliceOp.getMixedSizes();
    if (mixedSizes.size() != sliceOp.getType().getRank()) {
      SmallVector<OpFoldResult> rankReducedMixedSizes;
      rankReducedMixedSizes.reserve(sliceOp.getType().getRank());
      auto droppedDims = sliceOp.getDroppedDims();
      for (auto size : llvm::enumerate(mixedSizes)) {
        if (droppedDims.test(size.index())) continue;
        rankReducedMixedSizes.push_back(size.value());
      }
      std::swap(mixedSizes, rankReducedMixedSizes);
    }
    rewriter.replaceOpWithNewOp<linalg::InitTensorOp>(
        sliceOp, mixedSizes, sliceOp.getType().getElementType());
    return success();
  }
};

}  // namespace

void populateTileAndDistributeToWorkgroupsPatterns(
    RewritePatternSet &patterns, linalg::LinalgTilingOptions options,
    linalg::LinalgTransformationFilter filter) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<TileDispatchUsingSCFForOp>(context, std::move(options),
                                             std::move(filter));
  patterns.insert<SwapExtractSliceWithDispatchTensorLoad,
                  SwapExtractSliceWithInitTensor,
                  SwapExtractSliceWithTiledProducer>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
