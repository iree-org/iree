// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <deque>

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "tile-dispatch-using-interface"

namespace mlir::iree_compiler {

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<unsigned> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector;
  for (auto v : interchangeVector)
    filledVector.push_back(v);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<unsigned>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

/// Given the `lb` and `step` of a loop, return the lower bound and step to use
/// for a distributed loop. Replace the iteration domain to
/// - lb_partitioned = lb + procId * step
/// - step_partitioned = step * nprocs
static std::tuple<OpFoldResult, OpFoldResult>
getDistributeLBAndStep(OpBuilder &b, Location loc, OpFoldResult lb,
                       OpFoldResult step, OpFoldResult procId,
                       OpFoldResult nprocs) {
  AffineExpr s0, s1, s2;
  bindSymbols(b.getContext(), s0, s1, s2);
  auto offsetMap = AffineMap::get(0, 3, {s0 + s1 * s2});
  auto stepMap = AffineMap::get(0, 2, {s0 * s1});
  OpFoldResult distributeLB = affine::makeComposedFoldedAffineApply(
      b, loc, offsetMap, ArrayRef<OpFoldResult>{lb, procId, step});
  OpFoldResult distributeStep = affine::makeComposedFoldedAffineApply(
      b, loc, stepMap, ArrayRef<OpFoldResult>{step, nprocs});
  return {distributeLB, distributeStep};
}

// Helper function to change arith.constant to i64 attribute.
static void changeArithCstToI64Attr(OpBuilder &b,
                                    MutableArrayRef<OpFoldResult> constants) {
  for (OpFoldResult &val : constants) {
    if (auto dyn_cast = llvm::dyn_cast_if_present<Value>(val)) {
      APInt intVal;
      if (matchPattern(dyn_cast, m_ConstantInt(&intVal))) {
        val = b.getI64IntegerAttr(intVal.getSExtValue());
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// TileDispatchUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

/// This implementation mirrors the implementation in
/// include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h and
/// lib/Dialect/SCF/Transforms/TileUsingInterface.cpp. It is adapted to do
/// distribution and also use `iree_tensor_ext.dispatch.tensor.load/store`
/// instead of `tensor.extract_slice/insert_slice`.

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the
///   tile processed within the inner most loop.
static SmallVector<scf::ForOp> generateTileLoopNest(
    OpBuilder &builder, Location loc, ArrayRef<Range> loopRanges,
    ArrayRef<OpFoldResult> tileSizeVals,
    ArrayRef<linalg::DistributionMethod> distributionMethod,
    ArrayRef<linalg::ProcInfo> procInfo, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes, SmallVector<Value> &workgroupCount) {
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
  AffineExpr s0, s1, d0, s2;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());
  auto createBoundedTileSize = [&](OpFoldResult iv, OpFoldResult tileSize,
                                   OpFoldResult size) -> OpFoldResult {
    if (isConstantIntValue(tileSize, 1)) {
      return tileSize;
    }
    return affine::makeComposedFoldedAffineMin(
        builder, loc, minMap, ArrayRef<OpFoldResult>{iv, tileSize, size});
  };

  unsigned procDim = 0;
  AffineMap numIterationsMap =
      AffineMap::get(0, 3, (s1 - s0).ceilDiv(s2), builder.getContext());
  for (auto [idx, loopRange] : llvm::enumerate(loopRanges)) {
    // Capturing structured bindings in lambdas is a c++20 feature, so we have
    // to declare a local variable for it.
    int index = idx;
    OpFoldResult lb = loopRange.offset;
    OpFoldResult ub = loopRange.size;
    OpFoldResult step = tileSizeVals[index];
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    if (isConstantIntValue(tileSizeVals[index], 0)) {
      offsets[index] = lb;
      sizes[index] = ub;
      continue;
    }

    auto method = distributionMethod[index];
    if (method != linalg::DistributionMethod::None) {
      Value numWorkgroups = getValueOrCreateConstantIndexOp(
          builder, loc,
          affine::makeComposedFoldedAffineApply(
              builder, loc, numIterationsMap,
              {loopRange.offset, loopRange.size, tileSizeVals[index]}));
      workgroupCount.push_back(numWorkgroups);
      std::tie(lb, step) = getDistributeLBAndStep(builder, loc, lb, step,
                                                  procInfo[procDim].procId,
                                                  procInfo[procDim].nprocs);
      procDim++;
    }

    if (method == linalg::DistributionMethod::CyclicNumProcsEqNumIters) {
      offsets[index] = lb;
      sizes[index] = createBoundedTileSize(lb, tileSizeVals[index], ub);
      continue;
    }

    Value lbVal = getValueOrCreateConstantIndexOp(builder, loc, lb);
    Value ubVal = getValueOrCreateConstantIndexOp(builder, loc, ub);
    Value stepVal = getValueOrCreateConstantIndexOp(builder, loc, step);
    auto loop = builder.create<scf::ForOp>(
        loc, lbVal, ubVal, stepVal, ValueRange{},
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
            ValueRange /*iterArgs*/) {
          sizes[index] = createBoundedTileSize(iv, tileSizeVals[index], ub);
          builder.create<scf::YieldOp>(loc);
        });
    offsets[index] = loop.getInductionVar();
    loops.push_back(loop);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
  }

  // Update the sizes if it contains arith.index with i64 attrs.
  // TODO: tensor.extract_slice is unable to determine the
  // result type if arith.constant is present. This is a workaround
  // to ensure that the result type is determined.
  changeArithCstToI64Attr(builder, sizes);

  return loops;
}

/// Replace the `iree_tensor_ext.dispatch.tensor.store` of the `untiledValue`
/// with a tiled `iree_tensor_ext.dispatch.tensor.store` that writes only a tile
/// of the result at offsets given by `tiledOffsets` and sizes given by
/// `tiledSizes`, using `tiledValue` as the source.
static LogicalResult replaceStoresWithTiledVersion(
    RewriterBase &rewriter, OpResult untiledValue, Value tiledValue,
    ArrayRef<OpFoldResult> tileOffsets, ArrayRef<OpFoldResult> tileSizes,
    Block *innerLoopBody) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(innerLoopBody->getTerminator());
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> storeOps;
  for (OpOperand &use : untiledValue.getUses()) {
    auto storeOp =
        dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(use.getOwner());
    if (storeOp && storeOp.getValue() == use.get()) {
      storeOps.push_back(storeOp);
    }
  }
  if (storeOps.empty())
    return success();
  if (storeOps.size() != 1) {
    return rewriter.notifyMatchFailure(untiledValue.getOwner(),
                                       "expected a single store for the op");
  }
  auto storeOp = storeOps.front();

  SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                        rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
  SliceAndDynamicDims clonedSliceAndVals =
      cloneOffsetsSizesAndStrides(rewriter, storeOp);

  if (failed(affine::mergeOffsetsSizesAndStrides(
          rewriter, storeOp.getLoc(), clonedSliceAndVals.offsets,
          clonedSliceAndVals.sizes, clonedSliceAndVals.strides,
          storeOp.getDroppedDims(), tileOffsets, tileSizes, tileStrides,
          combinedOffsets, combinedSizes, combinedStrides))) {
    return rewriter.notifyMatchFailure(
        storeOp,
        "failed to create tiled iree_tensor_ext.dispatch.tensor.store op");
  }

  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      storeOp.getLoc(), tiledValue, storeOp.getTarget(),
      clonedSliceAndVals.dynamicDims, combinedOffsets, combinedSizes,
      combinedStrides);
  rewriter.eraseOp(storeOp);
  return success();
}

/// Replace all `iree_tensor_ext.dispatch.tensor.store` operations that use
/// values produced by `untiledOp` as source, with tiled stores, with tiled
/// values produced by `tiledOp`.
static LogicalResult replaceAllStoresWithTiledVersion(
    RewriterBase &rewriter, TilingInterface untiledOp,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ValueRange tiledValues, Block *innerLoopBody) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(innerLoopBody->getTerminator());
  for (auto [index, result] : llvm::enumerate(untiledOp->getResults())) {
    SmallVector<OpFoldResult> resultOffsets, resultSizes;
    if (failed(untiledOp.getResultTilePosition(rewriter, index, offsets, sizes,
                                               resultOffsets, resultSizes))) {
      return rewriter.notifyMatchFailure(
          untiledOp, "failed to rewrite destructive update");
    }
    if (failed(replaceStoresWithTiledVersion(
            rewriter, llvm::cast<OpResult>(result), tiledValues[index],
            resultOffsets, resultSizes, innerLoopBody))) {
      return failure();
    }
  }
  return success();
}

FailureOr<IREETilingResult>
tileDispatchUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
                          linalg::LinalgTilingOptions options) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  Location loc = op.getLoc();
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) {
    return IREETilingResult();
  }

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(options.tileSizeComputationFunction(rewriter, op));
  tileSizes.resize(numLoops, rewriter.getIndexAttr(0));

  IREETilingResult tilingResult;
  tilingResult.tiledLoops.resize(numLoops, false);
  AffineExpr s0, s1, s2, s3; // lb, ub, step, tileSize
  bindSymbols(rewriter.getContext(), s0, s1, s2, s3);
  AffineExpr numTilesExprs = (s1 - s0).ceilDiv(s2 * s3);
  for (auto [index, iteratorType, range, tileSize] :
       llvm::enumerate(op.getLoopIteratorTypes(), iterationDomain, tileSizes)) {
    // If distribution is specified, only parallel loops are tiled.
    if (options.distribution && iteratorType != utils::IteratorType::parallel) {
      continue;
    }
    // If tile size is 0, it isnt tiled.
    if (isConstantIntValue(tileSize, 0)) {
      continue;
    }
    // If number of tiles is statically know to be 1, the loop isnt tiled.
    OpFoldResult numTiles = affine::makeComposedFoldedAffineApply(
        rewriter, loc, numTilesExprs,
        {range.offset, range.size, range.stride, tileSize});
    if (isConstantIntValue(numTiles, 1)) {
      continue;
    }

    tilingResult.tiledLoops.set(index);
  }

  if (!tilingResult.tiledLoops.any()) {
    return IREETilingResult();
  }

  {
    SmallVector<OpFoldResult> offsets, sizes;
    SmallVector<Value> workgroupCount;
    // If there is an interchange specified, permute the iteration domain and
    // the tile sizes.
    SmallVector<int64_t> interchangeVector;
    if (!options.interchangeVector.empty()) {
      interchangeVector = fillInterchangeVector(options.interchangeVector,
                                                iterationDomain.size());
    }
    if (!interchangeVector.empty()) {
      if (!isPermutationVector(interchangeVector)) {
        return rewriter.notifyMatchFailure(
            op, "invalid intechange vector, not a permutation of the entire "
                "iteration space");
      }

      applyPermutationToVector(iterationDomain, interchangeVector);
      applyPermutationToVector(tileSizes, interchangeVector);
    }

    // If there is distribution specified, adjust the loop ranges. Note that
    // permutation happens *before* distribution.
    SmallVector<linalg::DistributionMethod> distributionMethods(
        iterationDomain.size(), linalg::DistributionMethod::None);
    SmallVector<linalg::ProcInfo> procInfo;
    if (options.distribution) {
      SmallVector<Range> parallelLoopRanges;
      for (auto loopIdx : llvm::seq<unsigned>(0, numLoops)) {
        if (tilingResult.tiledLoops.test(loopIdx)) {
          AffineExpr s0, s1;
          bindSymbols(rewriter.getContext(), s0, s1);
          OpFoldResult parallelLoopStep = affine::makeComposedFoldedAffineApply(
              rewriter, loc, s0 * s1,
              {iterationDomain[loopIdx].stride, tileSizes[loopIdx]});
          Range r = {iterationDomain[loopIdx].offset,
                     iterationDomain[loopIdx].size, parallelLoopStep};
          parallelLoopRanges.emplace_back(std::move(r));
        }
      }

      procInfo =
          options.distribution->procInfo(rewriter, loc, parallelLoopRanges);

      unsigned partitionedLoopIdx = 0;
      for (auto loopIdx : llvm::seq<unsigned>(0, numLoops)) {
        if (!tilingResult.tiledLoops.test(loopIdx)) {
          continue;
        }
        distributionMethods[loopIdx] =
            procInfo[partitionedLoopIdx++].distributionMethod;
      }
    }

    // 3. Materialize an empty loop nest that iterates over the tiles. These
    // loops for now do not return any values even if the original operation has
    // results.
    tilingResult.loops = generateTileLoopNest(
        rewriter, loc, iterationDomain, tileSizes, distributionMethods,
        procInfo, offsets, sizes, workgroupCount);

    if (!interchangeVector.empty()) {
      auto inversePermutation = invertPermutationVector(interchangeVector);
      applyPermutationToVector(offsets, inversePermutation);
      applyPermutationToVector(sizes, inversePermutation);
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
    FailureOr<TilingResult> tiledImplementation =
        op.getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tiledImplementation)) {
      return rewriter.notifyMatchFailure(
          op, "failed to generate tiled implementation");
    }
    std::swap(tilingResult.tiledOps, tiledImplementation->tiledOps);
    std::swap(tilingResult.tiledValues, tiledImplementation->tiledValues);

    LLVM_DEBUG({
      if (!tilingResult.loops.empty()) {
        llvm::errs() << "After tiled implementation :\n";
        tilingResult.loops.front().dump();
        llvm::errs() << "\n";
      }
    });
    std::swap(tilingResult.tileOffsets, offsets);
    std::swap(tilingResult.tileSizes, sizes);
    std::swap(tilingResult.workgroupCount, workgroupCount);
  }

  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
    return tilingResult;
  }
  Block *body = tilingResult.loops.empty()
                    ? op->getBlock()
                    : tilingResult.loops.back().getBody();
  // Rewrite all `iree_tensor_ext.dispatch.tensor.store` operation with tiled
  // version of the store. Its valid to this for all stores of the root untiled
  // op.
  if (failed(replaceAllStoresWithTiledVersion(
          rewriter, op, tilingResult.tileOffsets, tilingResult.tileSizes,
          tilingResult.tiledValues, body))) {
    return failure();
  }
  return tilingResult;
}

//===----------------------------------------------------------------------===//
// tileAndFuseDispatchUsingSCFForOp implementation.
//===----------------------------------------------------------------------===//

/// Find all producers to fuse and return them in sorted order;
static SmallVector<Operation *> getAllFusableProducers(TilingInterface op) {
  llvm::SetVector<Operation *> producers;
  std::deque<Operation *> worklist;
  worklist.push_back(op);

  while (!worklist.empty()) {
    Operation *currOp = worklist.front();
    worklist.pop_front();
    for (OpOperand &operand : currOp->getOpOperands()) {
      Operation *definingOp = operand.get().getDefiningOp();
      auto tilingInterfaceProducer =
          dyn_cast_or_null<TilingInterface>(definingOp);
      if (!tilingInterfaceProducer || isa<tensor::PadOp>(definingOp) ||
          producers.count(tilingInterfaceProducer)) {
        continue;
      }
      worklist.push_back(tilingInterfaceProducer);
      producers.insert(tilingInterfaceProducer);
    }
  }

  SmallVector<Operation *> sortedOps(producers.begin(), producers.end());
  mlir::computeTopologicalSorting(sortedOps);
  return sortedOps;
}

/// Return all slices that are used to access a tile of the producer. Assume
/// that `tiledOps` are in "reverse" order of their appearance in the IR.
static SmallVector<tensor::ExtractSliceOp>
getAllFusableProducerUses(Operation *untiledOp,
                          ArrayRef<Operation *> tiledOps) {
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  for (auto tiledOp : llvm::reverse(tiledOps)) {
    for (OpOperand &operand : llvm::reverse(tiledOp->getOpOperands())) {
      auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
      if (!sliceOp || sliceOp.getSource().getDefiningOp() != untiledOp)
        continue;
      sliceOps.push_back(sliceOp);
    }
  }
  return sliceOps;
}

FailureOr<IREETileAndFuseResult>
tileAndFuseDispatchUsingSCFForOp(RewriterBase &rewriter, TilingInterface op,
                                 linalg::LinalgTilingOptions tilingOptions) {
  IREETileAndFuseResult tileAndFuseResult;
  auto fusableProducers = getAllFusableProducers(op);
  // Apply the tiling pattern.
  FailureOr<IREETilingResult> tilingResult =
      tileDispatchUsingSCFForOp(rewriter, op, tilingOptions);
  if (failed(tilingResult)) {
    return failure();
  }
  tileAndFuseResult.tiledAndFusedOps = tilingResult->tiledOps;
  tileAndFuseResult.loops = tilingResult->loops;
  tileAndFuseResult.workgroupCount = tilingResult->workgroupCount;

  // If there is no tiling then there is nothing to do for fusion.
  if (!tilingResult->tiledLoops.any()) {
    return tileAndFuseResult;
  }

  // Get all ops to tile and fuse.
  auto fusableProducersRef = llvm::ArrayRef(fusableProducers);
  while (!fusableProducersRef.empty()) {
    auto fusableProducer = cast<TilingInterface>(fusableProducersRef.back());
    fusableProducersRef = fusableProducersRef.drop_back();

    // Find a slice that is used to access the producer. Get all the slice ops.
    // It is assumed that the slice ops are returned in-order, so that you
    // could use the first slice as the insertion point.
    auto sliceOps = getAllFusableProducerUses(
        fusableProducer, tileAndFuseResult.tiledAndFusedOps);

    for (auto sliceOp : sliceOps) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(sliceOp);

      // Generate the tiled implementation of the producer.
      OpResult untiledValue = llvm::cast<OpResult>(sliceOp.getSource());
      FailureOr<TilingResult> swapSliceResult =
          tensor::replaceExtractSliceWithTiledProducer(rewriter, sliceOp,
                                                       untiledValue);
      if (failed(swapSliceResult) || swapSliceResult->tiledValues.size() != 1) {
        return rewriter.notifyMatchFailure(sliceOp,
                                           "fusion along slice op failed");
      }

      Block *body = tileAndFuseResult.loops.empty()
                        ? op->getBlock()
                        : tileAndFuseResult.loops.back().getBody();
      // 2c. Finally replace any `iree_tensor_ext.dispatch.tensor.store`
      // operation with
      //     tiled version of the operation. It is only valid to do this under
      //     the above assumption that the producer and consumer share the loops
      //     that can be tiled.
      if (failed(replaceStoresWithTiledVersion(
              rewriter, untiledValue, swapSliceResult->tiledValues[0],
              sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(), body))) {
        return failure();
      }
      rewriter.replaceOp(sliceOp, swapSliceResult->tiledValues[0]);
      tileAndFuseResult.tiledAndFusedOps.append(swapSliceResult->tiledOps);
    }
  }

  return tileAndFuseResult;
}

namespace {

//===----------------------------------------------------------------------===//
// SwapExtractSliceWithDispatchTensorLoad
//===----------------------------------------------------------------------===//

/// Pattern to swap `iree_tensor_ext.dispatch.tensor.load` ->
/// `tensor.extract_slice` with `iree_tensor_ext.dispatch.tensor.load` of the
/// slice.
struct SwapExtractSliceWithDispatchTensorLoad
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto loadOp = sliceOp.getSource()
                      .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp)
      return failure();

    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(affine::mergeOffsetsSizesAndStrides(
            rewriter, loadOp.getLoc(), loadOp, sliceOp, loadOp.getDroppedDims(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return rewriter.notifyMatchFailure(
          sliceOp, "failed to fold offsets, sizes and strides with load op");
    }
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        sliceOp, sliceOp.getType(), loadOp.getSource(), loadOp.getSourceDims(),
        combinedOffsets, combinedSizes, combinedStrides);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SwapExtractSliceWithTensorEmpty
//===----------------------------------------------------------------------===//

/// Pattern to swap `empty` -> `tensor.extract_slice` with
/// `empty` of the slice.
struct SwapExtractSliceWithTensorEmpty
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto emptyTensorOp = sliceOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyTensorOp)
      return failure();

    SmallVector<OpFoldResult> mixedSizes = sliceOp.getMixedSizes();
    if (mixedSizes.size() != sliceOp.getType().getRank()) {
      SmallVector<OpFoldResult> rankReducedMixedSizes;
      rankReducedMixedSizes.reserve(sliceOp.getType().getRank());
      auto droppedDims = sliceOp.getDroppedDims();
      for (auto [index, size] : llvm::enumerate(mixedSizes)) {
        if (droppedDims.test(index))
          continue;
        rankReducedMixedSizes.push_back(size);
      }
      std::swap(mixedSizes, rankReducedMixedSizes);
    }
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
        sliceOp, mixedSizes, sliceOp.getType().getElementType());
    return success();
  }
};

} // namespace

void populateTileAndDistributeToWorkgroupsCleanupPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<SwapExtractSliceWithDispatchTensorLoad,
                  SwapExtractSliceWithTensorEmpty>(context);
}

} // namespace mlir::iree_compiler
