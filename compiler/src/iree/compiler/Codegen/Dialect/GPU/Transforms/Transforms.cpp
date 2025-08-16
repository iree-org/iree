// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include <cstdint>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-codegen-gpu-transforms"

namespace mlir::iree_compiler::IREE::GPU {

//===---------------------------------------------------------------------===//
// Forall Fusion
//===---------------------------------------------------------------------===//

static FailureOr<SmallVector<scf::ForallOp>>
getEquivalentMappingConsumerLoopNest(scf::ForallOp producer,
                                     scf::ForallOp consumer) {
  auto compareMappingTypes = [&](ArrayRef<Attribute> l, ArrayRef<Attribute> r) {
    return (llvm::all_of(l, llvm::IsaPred<gpu::GPUThreadMappingAttr>) &&
            llvm::all_of(r, llvm::IsaPred<gpu::GPUThreadMappingAttr>)) ||
           (llvm::all_of(l, llvm::IsaPred<gpu::GPUWarpMappingAttr>) &&
            llvm::all_of(r, llvm::IsaPred<gpu::GPUWarpMappingAttr>));
  };

  ArrayRef<Attribute> producerMapping = producer.getMappingAttr().getValue();
  ArrayRef<Attribute> consumerMapping = consumer.getMappingAttr().getValue();

  if (producerMapping.empty() || consumerMapping.empty()) {
    return failure();
  }

  // Require descending relative indices so that the linearization and
  // delinearization done in subsequent steps are valid.
  if (!isDescendingRelativeMappingIndices(producerMapping) ||
      !isDescendingRelativeMappingIndices(consumerMapping)) {
    return failure();
  }

  // If both loops share the same kind of mapping, return the sole consumer.
  if (compareMappingTypes(producerMapping, consumerMapping)) {
    return SmallVector<scf::ForallOp>({consumer});
  }

  // The only other supported case is fusing a thread mapped loop into a nest
  // of a warp and lane forall.
  if (!llvm::all_of(producerMapping,
                    llvm::IsaPred<gpu::GPUThreadMappingAttr>) ||
      !llvm::all_of(consumerMapping, llvm::IsaPred<IREE::GPU::LaneIdAttr>)) {
    return failure();
  }
  auto outerWarpLoop = consumer->getParentOfType<scf::ForallOp>();
  if (!outerWarpLoop || !llvm::all_of(outerWarpLoop.getMappingAttr().getValue(),
                                      llvm::IsaPred<gpu::GPUWarpMappingAttr>)) {
    return failure();
  }
  return SmallVector<scf::ForallOp>({outerWarpLoop, consumer});
}

static FailureOr<Value> createSharedAllocDestination(RewriterBase &rewriter,
                                                     scf::ForallOp forallOp) {
  if (forallOp->getNumResults() != 1) {
    return failure();
  }

  auto empty = forallOp.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  // Fail if the destination is not a `tensor.empty` op and cannot be trivially
  // converted to a `bufferization.alloc_tensor`.
  if (!empty) {
    return failure();
  }

  // Create a `bufferization.alloc_tensor` op with memory space
  // `#gpu.address_space<workgroup>`.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(empty);
  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocTensor = rewriter.create<bufferization::AllocTensorOp>(
      empty->getLoc(), cast<TensorType>(empty.getResult().getType()),
      empty.getDynamicSizes(),
      /*copy=*/Value(), /*size_hint=*/Value(),
      /*memory_space=*/sharedMemoryAddrSpace);
  return allocTensor.getResult();
}

LogicalResult fuseForallIntoConsumer(RewriterBase &rewriter,
                                     scf::ForallOp producer,
                                     scf::ForallOp consumer,
                                     SmallVector<Operation *> consumerChain) {
  // TODO: Support multi-result producer loops.
  if (producer->getNumResults() != 1) {
    return failure();
  }

  FailureOr<SmallVector<scf::ForallOp>> consumerLoopNest =
      getEquivalentMappingConsumerLoopNest(producer, consumer);
  if (failed(consumerLoopNest)) {
    return failure();
  }

  // Verify that all loops are normalized.
  if (!producer.isNormalized() ||
      !llvm::all_of(consumerLoopNest.value(), [](scf::ForallOp forall) {
        return forall.isNormalized();
      })) {
    return failure();
  }

  // Step 1. Get the destination of the producer loop as a shared memory
  // allocation.
  rewriter.setInsertionPointToStart(consumer.getBody());
  FailureOr<Value> maybeDest = createSharedAllocDestination(rewriter, producer);
  if (failed(maybeDest)) {
    return failure();
  }
  Value sharedDest = maybeDest.value();

  // Step 2. Move the consumer chain to right before the last user in the
  // chain.
  if (!consumerChain.empty()) {
    Operation *base = consumerChain.back();
    for (Operation *op : consumerChain) {
      if (op == base) {
        continue;
      }
      rewriter.moveOpBefore(op, base);
    }
  }

  // Step 3. Create the `iree_gpu.barrier_region` to wrap the fused producer.
  auto barrierOp = rewriter.create<IREE::GPU::BarrierRegionOp>(
      producer.getLoc(), sharedDest.getType(), sharedDest);
  rewriter.setInsertionPointToStart(barrierOp.getBody());

  // Step 4. Compute the producer IDs in terms of the consumer IDs.
  // The producer IDs are computed as follows:
  //
  // producer = [p0, ..., pn] ∈ [0, ..., 0] to [P0, ..., Pn]
  // consumer = [c0, ..., cn] ∈ [0, ..., 0] to [C0, ..., Cn]
  //
  //                   Not a real op
  //                         |
  // %ub = P0 * ... * Pn     |
  // %step = C0 * ... * Cn   v
  // %flatc = affine.linearize_index %c0, ..., %cn
  // scf.for %id = %flatc to %ub step %step {
  //   %p:n = affine.delinearize_index %id into [%P0, ..., %Pn]
  //   ...
  // }
  //
  // Note: We use 0 as the loop lower bound instead of the linearized consumer
  // loop ID if possible to make later loop promotion patterns easier.

  MLIRContext *context = rewriter.getContext();
  Location loc = producer.getLoc();

  // Compute the linearize consumer loop ID and total consumer loop worker
  // count (C0 * ... * Cn).
  AffineExpr d0, d1, d2;
  bindDims(context, d0, d1, d2);
  AffineExpr mulAdd = d0 * d1 + d2;
  OpFoldResult linearId = rewriter.getIndexAttr(0);
  OpFoldResult consumerWorkerCount = rewriter.getIndexAttr(1);
  for (auto loop : *consumerLoopNest) {
    for (auto [inductionVar, workerCount] :
         llvm::zip_equal(getAsOpFoldResult(loop.getInductionVars()),
                         loop.getMixedUpperBound())) {
      linearId = affine::makeComposedFoldedAffineApply(
          rewriter, loc, mulAdd, {linearId, workerCount, inductionVar});
      consumerWorkerCount = affine::makeComposedFoldedAffineApply(
          rewriter, loc, d0 * d1, {consumerWorkerCount, workerCount});
    }
  }

  // Compute the total producer loop worker count (P0 * ... * Pn).
  Value linearConsumerIdVal =
      getValueOrCreateConstantIndexOp(rewriter, loc, linearId);
  SmallVector<OpFoldResult> producerRanges;
  OpFoldResult producerWorkerCount = rewriter.getIndexAttr(1);
  for (auto workerCount : producer.getMixedUpperBound()) {
    producerRanges.push_back(workerCount);
    producerWorkerCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, d0 * d1, {producerWorkerCount, workerCount});
  }

  std::optional<int64_t> staticProducerCount =
      getConstantIntValue(producerWorkerCount);
  std::optional<int64_t> staticConsumerCount =
      getConstantIntValue(consumerWorkerCount);
  bool perfectlyDivides =
      staticConsumerCount && staticProducerCount &&
      staticProducerCount.value() % staticConsumerCount.value() == 0;

  // Step 5. Create the `scf.for` loop for the producer.
  // If the consumer worker count perfectly divides the producer worker count,
  // then we can use a lower bound of 0 and keep the loop bounds static.
  Value lb = perfectlyDivides ? rewriter.create<arith::ConstantIndexOp>(loc, 0)
                              : linearConsumerIdVal;
  Value ub =
      getValueOrCreateConstantIndexOp(rewriter, loc, producerWorkerCount);
  Value step =
      getValueOrCreateConstantIndexOp(rewriter, loc, consumerWorkerCount);
  auto newProducer = rewriter.create<scf::ForOp>(
      loc, lb, ub, step, barrierOp.getBody()->getArgument(0));
  setLoopUnrollMarker(newProducer);
  Block *loopBody = newProducer.getBody();

  // Get the replacement IDs for the producer loop.
  rewriter.setInsertionPointToStart(loopBody);
  Value newFlatProducerId =
      perfectlyDivides
          ? affine::makeComposedAffineApply(
                rewriter, loc, d0 + d1,
                {newProducer.getInductionVar(), linearConsumerIdVal})
          : newProducer.getInductionVar();

  // We require a descending relative mapping and scf.forall loop ranges are
  // listed from outer most to inner most, so we can use the ranges directly
  // for the delinearization basis.
  auto delinearize = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, newFlatProducerId, llvm::to_vector(producerRanges));

  SmallVector<Value> newBlockArgs = delinearize.getResults();
  newBlockArgs.append(newProducer.getRegionIterArgs().begin(),
                      newProducer.getRegionIterArgs().end());

  // Step 6. Inline the region of the producer and replace the terminator.
  scf::InParallelOp terminator = producer.getTerminator();
  rewriter.mergeBlocks(producer.getBody(), loopBody, newBlockArgs);

  rewriter.setInsertionPointAfter(terminator);
  auto parallelInsert =
      cast<tensor::ParallelInsertSliceOp>(*terminator.getYieldingOps().begin());

  // Create an insert_slice to yield from the loop body.
  SmallVector<OpFoldResult, 4> sourceOffsets = parallelInsert.getMixedOffsets();
  SmallVector<OpFoldResult, 4> sourceSizes = parallelInsert.getMixedSizes();
  SmallVector<OpFoldResult, 4> sourceStrides = parallelInsert.getMixedStrides();
  Value insertedSlice = rewriter.create<tensor::InsertSliceOp>(
      loc, parallelInsert.getSource(), parallelInsert.getDest(),
      parallelInsert.getMixedOffsets(), parallelInsert.getMixedSizes(),
      parallelInsert.getMixedStrides());
  rewriter.create<scf::YieldOp>(loc, insertedSlice);
  rewriter.eraseOp(parallelInsert);
  rewriter.eraseOp(terminator);

  // Step 7. Yield the result of the loop from the barrier op and replace the
  // producer.
  rewriter.setInsertionPointToEnd(barrierOp.getBody());
  rewriter.create<IREE::GPU::YieldOp>(loc, newProducer.getResults());

  rewriter.replaceOp(producer, barrierOp);
  return success();
}

FailureOr<scf::ForallOp>
fuseNestedLaneAndWarpForalls(RewriterBase &rewriter, scf::ForallOp warpForallOp,
                             scf::ForallOp laneForallOp) {
  // Verify mappings.
  if (!warpForallOp.getMapping() ||
      !llvm::all_of(*warpForallOp.getMapping(), [](Attribute mappingAttr) {
        return isa<gpu::GPUWarpMappingAttr>(mappingAttr);
      })) {
    return rewriter.notifyMatchFailure(warpForallOp, "not a warp forall op");
  }
  if (!laneForallOp.getMapping() || laneForallOp.getMapping()->size() != 1 ||
      !isa<IREE::GPU::LaneIdAttr>(laneForallOp.getMapping()->getValue()[0])) {
    return rewriter.notifyMatchFailure(
        laneForallOp, "inner forall op is not mapped to a single lane id");
  }
  // Verify the lane forall is the only nested forall op.
  SmallVector<scf::ForallOp> innerForallOps(
      warpForallOp.getBody()->getOps<scf::ForallOp>());
  if (innerForallOps.size() != 1) {
    return rewriter.notifyMatchFailure(warpForallOp,
                                       "expected a single inner forall op");
  }
  if (warpForallOp.getOperation() != laneForallOp->getParentOp()) {
    return rewriter.notifyMatchFailure(
        laneForallOp, "expected warp forall op to be the lane forall's parent");
  }
  // Only allow arith/affine ops and tensor.extract_slice. We would want other
  // ops to be fused into the lane forall before this transformation happens.
  for (Operation &op : warpForallOp.getBody()->getOperations()) {
    if (!isa_and_nonnull<arith::ArithDialect, affine::AffineDialect>(
            op.getDialect()) &&
        !isa<tensor::ExtractSliceOp, scf::ForallOp, scf::InParallelOp>(op)) {
      return rewriter.notifyMatchFailure(
          warpForallOp,
          "Warp forall body has non arith, affine, or extract_slice ops");
    }
  }
  if (!warpForallOp.isNormalized() || !laneForallOp.isNormalized()) {
    return rewriter.notifyMatchFailure(warpForallOp,
                                       "forall ops are not normalized");
  }
  if (laneForallOp.getNumResults() != warpForallOp.getNumResults()) {
    return rewriter.notifyMatchFailure(
        laneForallOp,
        "lane and warp foralls have a different number of results");
  }
  auto hasSingleNonStridedFullRankParallelInsert = [](scf::ForallOp forall) {
    for (BlockArgument initArg : forall.getRegionOutArgs()) {
      SmallVector<Operation *> combiningOps = forall.getCombiningOps(initArg);
      if (combiningOps.size() != 1) {
        return false;
      }
      auto parallelInsertOp =
          dyn_cast<tensor::ParallelInsertSliceOp>(combiningOps[0]);
      if (!parallelInsertOp ||
          !areAllConstantIntValue(parallelInsertOp.getMixedStrides(), 1)) {
        return false;
      }
      if (parallelInsertOp.getSourceType().getRank() !=
          parallelInsertOp.getDestType().getRank()) {
        return false;
      }
    }
    return true;
  };
  if (!hasSingleNonStridedFullRankParallelInsert(warpForallOp) ||
      !hasSingleNonStridedFullRankParallelInsert(laneForallOp)) {
    return rewriter.notifyMatchFailure(warpForallOp,
                                       "forall op has strided combining op");
  }
  // Verify that the source of all combining ops of the warpForallOp are
  // produced by the laneForallOp, and that the laneForallOp combining ops
  // have the same rank as the corresponding combining op in the warpForallOp.
  // This is the requirement for composing the combining ops into a single
  // thread distributed combining op.
  for (auto [warpResult, laneResult] :
       llvm::zip_equal(warpForallOp.getResults(), laneForallOp.getResults())) {
    unsigned int resultIdx = warpResult.getResultNumber();
    BlockArgument warpInitArg = warpForallOp.getTiedBlockArgument(
        &warpForallOp.getOutputsMutable()[resultIdx]);
    auto warpInsertOp = cast<tensor::ParallelInsertSliceOp>(
        warpForallOp.getCombiningOps(warpInitArg)[0]);
    if (warpInsertOp.getSource() != laneResult) {
      return rewriter.notifyMatchFailure(
          warpForallOp, "combining op source is not inner lane forall result");
    }
  }

  // Create the thread mapped forall to replace the nested foralls.
  SmallVector<Attribute> threadMappings = llvm::map_to_vector(
      llvm::seq<unsigned>(warpForallOp.getRank() + laneForallOp.getRank()),
      [&](unsigned mappingDim) -> Attribute {
        unsigned mappingId =
            static_cast<unsigned>(gpu::MappingId::LinearDim0) + mappingDim;
        return gpu::GPUThreadMappingAttr::get(
            rewriter.getContext(), static_cast<gpu::MappingId>(mappingId));
      });
  std::reverse(threadMappings.begin(), threadMappings.end());
  SmallVector<OpFoldResult> lbs(threadMappings.size(),
                                rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> steps(threadMappings.size(),
                                  rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> ubs = llvm::to_vector(llvm::concat<OpFoldResult>(
      warpForallOp.getMixedUpperBound(), laneForallOp.getMixedUpperBound()));
  scf::ForallOp threadForallOp = rewriter.create<scf::ForallOp>(
      warpForallOp.getLoc(), lbs, ubs, steps, warpForallOp.getOutputs(),
      ArrayAttr::get(rewriter.getContext(), threadMappings));

  // Collect pairs of combining ops to compose after inlining everything.
  SmallVector<
      std::pair<tensor::ParallelInsertSliceOp, tensor::ParallelInsertSliceOp>>
      insertPairs;
  for (auto [warpArg, laneArg] : llvm::zip_equal(
           warpForallOp.getRegionOutArgs(), laneForallOp.getRegionOutArgs())) {
    auto warpInsert = cast<tensor::ParallelInsertSliceOp>(
        warpForallOp.getCombiningOps(warpArg)[0]);
    auto laneInsert = cast<tensor::ParallelInsertSliceOp>(
        laneForallOp.getCombiningOps(laneArg)[0]);
    insertPairs.push_back({warpInsert, laneInsert});
  }
  // The lane forall's terminator also needs to be erased after inlining.
  Operation *laneForallTerminator = laneForallOp.getBody()->getTerminator();
  // Inline the warp and lane forall bodies into the thread forall, and remap
  // the induction variables.
  // The terminator will be replaced with the terminator of the inlined block.
  rewriter.eraseOp(threadForallOp.getTerminator());
  SmallVector<Value> replacementArgs(threadForallOp.getInductionVars());
  replacementArgs.pop_back();
  replacementArgs.append(threadForallOp.getRegionOutArgs().begin(),
                         threadForallOp.getRegionOutArgs().end());
  rewriter.mergeBlocks(warpForallOp.getBody(), threadForallOp.getBody(),
                       replacementArgs);
  SmallVector<Value> innerReplacementArgs(laneForallOp.getOutputs());
  innerReplacementArgs.insert(innerReplacementArgs.begin(),
                              threadForallOp.getInductionVars().back());
  rewriter.inlineBlockBefore(laneForallOp.getBody(), laneForallOp,
                             innerReplacementArgs);
  for (auto [warpInsert, laneInsert] : insertPairs) {
    SmallVector<OpFoldResult> composedOffsets;
    for (auto [warpOffset, laneOffset] : llvm::zip_equal(
             warpInsert.getMixedOffsets(), laneInsert.getMixedOffsets())) {
      SmallVector<Value> offsets;
      if (auto warpOffsetVal = dyn_cast<Value>(warpOffset)) {
        offsets.push_back(warpOffsetVal);
      }
      if (auto laneOffsetVal = dyn_cast<Value>(laneOffset)) {
        offsets.push_back(laneOffsetVal);
      }
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(warpInsert);
      if (!offsets.empty()) {
        (void)setInsertionPointAfterLastValue(rewriter, offsets);
      }
      composedOffsets.push_back(IREE::LinalgExt::addOfrs(
          rewriter, laneInsert.getLoc(), warpOffset, laneOffset));
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(warpInsert);
    rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
        warpInsert, laneInsert.getSource(), warpInsert.getDest(),
        composedOffsets, laneInsert.getMixedSizes(),
        laneInsert.getMixedStrides());
    rewriter.eraseOp(laneInsert);
  }
  rewriter.eraseOp(laneForallTerminator);
  rewriter.eraseOp(laneForallOp);
  return threadForallOp;
}

/// Return whether a parallel insert slice operation can be collapsed with
/// the given reassociation indices. For a slice to be collapsible, each group
/// of collapsed dimensions must be fully contiguous in the destination type.
/// For example, the following parallel_insert_slice can be collapsed:
/// ```
///   tensor.parallel_insert_slice ...
///     tensor<1x2x3x4x5xf32> into tensor<?x?x3x?x5xf32>
///   ...
///   // Collapsed by:
///   tensor.collapse_shape %result [[0, 1, 2], [3, 4]]
///     tensor<?x?x3x?x5> into tensor<?x?x?xf32>
/// ```
/// Since the slice <1x2x3> is contiguous in <?x?x3> for the first group, and
/// the slice <4x5> is contiguous in <?x5> for the second group.
static LogicalResult
collapsibleSlicePrecondition(RewriterBase &rewriter,
                             tensor::ParallelInsertSliceOp sliceOp,
                             SmallVector<ReassociationIndices> reassociations) {
  if (!areAllConstantIntValue(sliceOp.getMixedStrides(), 1)) {
    return rewriter.notifyMatchFailure(sliceOp, "strides are not all 1");
  }
  RankedTensorType sliceType = sliceOp.getSourceType();
  if (sliceOp.getMixedSizes().size() != sliceType.getRank()) {
    return rewriter.notifyMatchFailure(
        sliceOp, "parallel insert slice is rank reducing");
  }

  SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
  RankedTensorType fullTensorType = sliceOp.getDestType();
  ArrayRef<int64_t> destShape = fullTensorType.getShape();
  for (auto group : reassociations) {
    bool isFullSlice = true;
    for (auto idx : llvm::reverse(group)) {
      std::optional<int64_t> constSize = getConstantIntValue(sizes[idx]);
      // If the size is dynamic, then conservatively assume it is not full.
      if (!constSize.has_value()) {
        if (!isFullSlice) {
          return rewriter.notifyMatchFailure(
              sliceOp,
              "parallel insert slice is not contiguous in the destination");
        }
        isFullSlice = false;
        continue;
      }
      if (constSize.value() == 1) {
        if (destShape[idx] == 1) {
          continue;
        }
        // Unit slices are okay as long as they are the outermost sliced dims.
        // Any other non-unit sliced dimension that is more outer than the unit
        // slice would be invalid, so we set `isFullSlice` to false.
        isFullSlice = false;
        continue;
      }
      // If the size is not unit, then the slice must be full so far.
      if (!isFullSlice) {
        return rewriter.notifyMatchFailure(
            sliceOp,
            "parallel insert slice is not contiguous in the destination");
      }
      if (constSize.value() != destShape[idx]) {
        isFullSlice = false;
      }
    }
  }

  return success();
}

/// Collapse all `ops` with the given `reassociations`. All `ops` are expected
/// to have equivalent offsets, sizes, and strides. All strides are expected to
/// be 1. This function assumes that the parallelInsertOp passes the
/// collapsibleSlicePrecondition.
static tensor::ParallelInsertSliceOp
collapseParallelInsertOp(RewriterBase &rewriter,
                         tensor::ParallelInsertSliceOp parallelInsertOp,
                         SmallVector<ReassociationIndices> reassociations) {
  OpBuilder::InsertionGuard g(rewriter);
  // Compute the collapsed offsets, sizes, and strides.
  auto subsetOp =
      cast<SubsetInsertionOpInterface>(parallelInsertOp.getOperation());
  Operation *lastOp = setInsertionPointAfterLastNeededValue(rewriter, subsetOp);
  Location loc = lastOp->getLoc();
  int64_t resultIdx = parallelInsertOp.getTiedOpResult().getResultNumber();
  auto forallOp = parallelInsertOp->getParentOfType<scf::ForallOp>();
  Value loopInit = forallOp.getOutputs()[resultIdx];
  SmallVector<OpFoldResult> mixedInitSizes =
      tensor::getMixedSizes(rewriter, loc, loopInit);
  auto prod = [&](ArrayRef<OpFoldResult> vals) -> OpFoldResult {
    auto mulMap = AffineMap::get(
        2, 0, {rewriter.getAffineDimExpr(0) * rewriter.getAffineDimExpr(1)});
    OpFoldResult product = rewriter.getIndexAttr(1);
    for (OpFoldResult val : vals) {
      product = affine::makeComposedFoldedAffineApply(rewriter, loc, mulMap,
                                                      {product, val});
    }
    return product;
  };
  SmallVector<OpFoldResult> offsets = parallelInsertOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = parallelInsertOp.getMixedSizes();
  SmallVector<OpFoldResult> newSizes, newOffsets;
  for (auto group : reassociations) {
    if (group.size() == 1) {
      newOffsets.push_back(offsets[group[0]]);
      newSizes.push_back(sizes[group[0]]);
      continue;
    }
    ArrayRef<OpFoldResult> basis(mixedInitSizes.begin() + group.front(),
                                 mixedInitSizes.begin() + group.back() + 1);
    ArrayRef<OpFoldResult> groupOffsets(offsets.begin() + group.front(),
                                        offsets.begin() + group.back() + 1);
    SmallVector<Value> offsetVals =
        llvm::map_to_vector(groupOffsets, [&](OpFoldResult ofr) {
          return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
        });
    OpFoldResult collapsedOffset =
        rewriter
            .create<affine::AffineLinearizeIndexOp>(loc, offsetVals, basis,
                                                    /*disjoint=*/true)
            .getResult();
    ArrayRef<OpFoldResult> groupSizes(sizes.begin() + group.front(),
                                      sizes.begin() + group.back() + 1);
    OpFoldResult collapsedSize = prod(groupSizes);
    newOffsets.push_back(collapsedOffset);
    newSizes.push_back(collapsedSize);
  }
  SmallVector<OpFoldResult> newStrides(newSizes.size(),
                                       rewriter.getIndexAttr(1));

  // Collapse the slice source.
  loc = parallelInsertOp.getParallelCombiningParent()->getLoc();
  rewriter.setInsertionPoint(parallelInsertOp.getParallelCombiningParent());
  auto newCollapse = rewriter.create<tensor::CollapseShapeOp>(
      loc, parallelInsertOp.getSource(), reassociations);

  // Collapse the parallel insert slice.
  rewriter.setInsertionPoint(parallelInsertOp);
  auto newInsertOp = rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
      parallelInsertOp, newCollapse, parallelInsertOp.getDest(), newOffsets,
      newSizes, newStrides);
  return newInsertOp;
}

FailureOr<scf::ForallOp>
fuseCollapseShapeIntoProducerForall(RewriterBase &rewriter,
                                    scf::ForallOp forallOp,
                                    tensor::CollapseShapeOp collapseOp) {
  // Check that there is a single user of the collapsed result.
  auto forallResult = cast<OpResult>(collapseOp.getSrc());
  if (!forallResult.hasOneUse()) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "forall result has multiple uses");
  }

  // Get the result's corresponding parallel_insert_slice op.
  SmallVector<Operation *> parallelInsertOps = forallOp.getCombiningOps(
      forallOp.getRegionIterArgs()[forallResult.getResultNumber()]);
  if (parallelInsertOps.size() != 1) {
    return rewriter.notifyMatchFailure(
        forallOp, "Expected a single parallel_insert_slice");
  }

  auto parallelInsertOp =
      dyn_cast<tensor::ParallelInsertSliceOp>(parallelInsertOps.front());
  if (!parallelInsertOp) {
    return rewriter.notifyMatchFailure(
        forallOp, "Expected parallel_insert_slice combining op");
  }

  // Collapse the parallel insert slice op.
  SmallVector<ReassociationIndices> reassociations =
      collapseOp.getReassociationIndices();
  if (failed(collapsibleSlicePrecondition(rewriter, parallelInsertOp,
                                          reassociations))) {
    return failure();
  }
  tensor::ParallelInsertSliceOp newParallelInsertOp =
      collapseParallelInsertOp(rewriter, parallelInsertOp, reassociations);

  // At this point, the newParallelInsertOp still has the destination of the
  // original parallel insert op, so the destination is the original expanded
  // init block argument, and we can use it to get the sizes for the expand.
  // The block argument will be corrected later, when the forall op is replaced.
  Value initArg = newParallelInsertOp.getDest();
  Value forallOutput = forallOp.getOutputs()[forallResult.getResultNumber()];
  Location loc = forallOutput.getLoc();
  rewriter.setInsertionPointAfterValue(forallOutput);
  SmallVector<OpFoldResult> initSizes =
      tensor::getMixedSizes(rewriter, loc, forallOutput);
  loc = initArg.getLoc();
  rewriter.setInsertionPointToStart(forallOp.getBody());
  auto expandedInitArg = rewriter.create<tensor::ExpandShapeOp>(
      loc, initArg.getType(), initArg, reassociations, initSizes);

  // The new parallel insert slice is collapsed, so don't use the expanded init.
  // Also don't replace the expand shape src with its own result.
  rewriter.replaceUsesWithIf(
      initArg, expandedInitArg.getResult(), [&](OpOperand &operand) {
        return operand != expandedInitArg.getSrcMutable() &&
               operand != newParallelInsertOp.getDestMutable();
      });

  // Now create a new scf::Forall with a collapsed loop init.
  loc = forallOp->getLoc();
  rewriter.setInsertionPoint(forallOp);
  SmallVector<Value> newForallOutputs(forallOp.getOutputs());
  Value collapsedLoopInit = rewriter.create<tensor::CollapseShapeOp>(
      loc, newForallOutputs[forallResult.getResultNumber()], reassociations);
  newForallOutputs[forallResult.getResultNumber()] = collapsedLoopInit;

  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newForallOutputs, forallOp.getMappingAttr());

  SmallVector<Value> argReplacements(newForallOp.getInductionVars());
  argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                         newForallOp.getRegionIterArgs().end());
  newForallOp.getTerminator()->erase();
  rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                       argReplacements);

  // Replaces the uses of the old scf.forall with the new scf.forall
  rewriter.replaceOp(collapseOp,
                     newForallOp.getResult(forallResult.getResultNumber()));
  for (int idx = 0; idx < forallOp->getNumResults(); ++idx) {
    if (idx == forallResult.getResultNumber()) {
      continue;
    }
    forallOp->getResult(idx).replaceAllUsesWith(newForallOp->getResult(idx));
  }
  rewriter.eraseOp(forallOp);
  return newForallOp;
}

/// Return whether the `parallelInsertOp` can be clamped along the sliced
/// dimensions of `extractSliceOp`. The dimensions of the extractSliceOp source
/// are expected to match the dimensions of the parallelInsertOp destination.
/// This function checks that the parallelInsertOp is not rank reducing along
/// any of the sliced dimensions of the extractSliceOp.
static LogicalResult canClampParallelInsertSlice(
    RewriterBase &rewriter, tensor::ParallelInsertSliceOp parallelInsertOp,
    tensor::ExtractSliceOp extractSliceOp,
    llvm::SmallDenseSet<unsigned int> insertRankReductionMask) {
  // Find the dimensions that are sliced by the extractSliceOp
  llvm::SmallDenseSet<unsigned int> slicedDims;
  ArrayRef<int64_t> sliceStaticSizes = extractSliceOp.getStaticSizes();
  ArrayRef<int64_t> sliceSourceSizes =
      extractSliceOp.getSourceType().getShape();
  for (int dim = 0; dim < sliceStaticSizes.size(); ++dim) {
    if (ShapedType::isDynamic(sliceStaticSizes[dim]) ||
        sliceStaticSizes[dim] != sliceSourceSizes[dim]) {
      slicedDims.insert(dim);
    }
  }
  for (int dim = 0; dim < parallelInsertOp.getDestType().getRank(); ++dim) {
    if (insertRankReductionMask.contains(dim) && slicedDims.contains(dim)) {
      return rewriter.notifyMatchFailure(
          parallelInsertOp, "parallel insert reduces sliced dimensions");
    }
  }
  return success();
}

/// Clamps the source of a parallel_insert_slice op to fit within the
/// `upperBoundSizes`. This function computes the upper bound sizes, and creates
/// an extract slice op on the parallel insert source, which is then used in a
/// new parallel insert slice to replace the old one. This function assumes that
/// the parallel insert op passes `canClampParallelInsertSlice` precondition.
static FailureOr<tensor::ParallelInsertSliceOp>
clampParallelInsertSliceOp(RewriterBase &rewriter,
                           tensor::ParallelInsertSliceOp parallelInsertOp,
                           SmallVector<OpFoldResult> upperBoundSizes) {
  OpBuilder::InsertionGuard g(rewriter);
  auto subsetOp =
      cast<SubsetInsertionOpInterface>(parallelInsertOp.getOperation());
  Operation *lastOp = setInsertionPointAfterLastNeededValue(rewriter, subsetOp);
  Location loc = lastOp->getLoc();

  // Clamp the parallel_insert_slice sizes to fit within the full result tensor.
  SmallVector<OpFoldResult> offsets = parallelInsertOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = parallelInsertOp.getMixedSizes();
  SmallVector<OpFoldResult> clampedSizes;
  for (auto [offset, size, ub] :
       llvm::zip_equal(offsets, sizes, upperBoundSizes)) {
    AffineExpr d0, d1, d2;
    MLIRContext *ctx = rewriter.getContext();
    bindDims(ctx, d0, d1, d2);
    auto lbClampMap = AffineMap::get(3, 0, {d0 - d1, d2}, ctx);
    auto ubClampMap = rewriter.getMultiDimIdentityMap(2);
    OpFoldResult lbClamped = affine::makeComposedFoldedAffineMax(
        rewriter, loc, lbClampMap, {ub, offset, rewriter.getIndexAttr(0)});
    OpFoldResult ubClamped = affine::makeComposedFoldedAffineMin(
        rewriter, loc, ubClampMap, {lbClamped, size});
    clampedSizes.push_back(ubClamped);
  }

  // Compute the clamped type. This could be rank reduced, but rank reduced
  // dimensions will never be potentially zero by construction. The earlier
  // matchers ensure that all sliceable users are not rank reduced along a
  // dimensions that is being sliced by the loop consumer.
  llvm::SmallDenseSet<unsigned int> rankReductionMask =
      computeRankReductionMask(parallelInsertOp.getStaticSizes(),
                               parallelInsertOp.getSourceType().getShape(),
                               /*matchDynamic=*/true)
          .value();
  SmallVector<int64_t> clampedShape;
  SmallVector<OpFoldResult> rankReducedClampedSizes;
  SmallVector<Value> d;
  for (auto [idx, clampedSize] : llvm::enumerate(clampedSizes)) {
    if (rankReductionMask.contains(idx)) {
      continue;
    }
    dispatchIndexOpFoldResult(clampedSize, d, clampedShape);
    rankReducedClampedSizes.push_back(clampedSize);
  }
  RankedTensorType clampedType =
      parallelInsertOp.getSourceType().clone(clampedShape);
  // Create an extract_slice to extract the correct size from the parallel
  // insert source.
  SmallVector<OpFoldResult> zeros(clampedType.getRank(),
                                  rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> ones(clampedType.getRank(),
                                 rewriter.getIndexAttr(1));
  Operation *combiningOp =
      parallelInsertOp.getParallelCombiningParent().getOperation();
  rewriter.setInsertionPoint(combiningOp);
  loc = combiningOp->getLoc();
  auto extractOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, clampedType, parallelInsertOp.getSource(), zeros,
      rankReducedClampedSizes, ones);

  // Replace the parallel insert op with the clamped version, and return the
  // new parallel insert slice.
  rewriter.setInsertionPoint(parallelInsertOp);
  loc = parallelInsertOp->getLoc();
  return rewriter.replaceOpWithNewOp<tensor::ParallelInsertSliceOp>(
      parallelInsertOp, extractOp.getResult(), parallelInsertOp.getDest(),
      parallelInsertOp.getMixedOffsets(), clampedSizes,
      parallelInsertOp.getMixedStrides());
}

FailureOr<scf::ForallOp>
fuseExtractSliceIntoProducerForall(RewriterBase &rewriter,
                                   scf::ForallOp forallOp,
                                   tensor::ExtractSliceOp extractSliceOp) {
  auto forallResult = cast<OpResult>(extractSliceOp.getSource());
  if (!forallResult.hasOneUse()) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "forall result has multiple uses");
  }
  BlockArgument initBbarg =
      forallOp.getRegionIterArgs()[forallResult.getResultNumber()];
  SmallVector<Operation *> parallelInsertOps =
      forallOp.getCombiningOps(initBbarg);
  if (parallelInsertOps.size() != 1) {
    return rewriter.notifyMatchFailure(
        forallOp, "Expected a single parallel_insert_slice");
  }

  auto parallelInsertOp =
      dyn_cast<tensor::ParallelInsertSliceOp>(parallelInsertOps.front());
  if (!parallelInsertOp) {
    return rewriter.notifyMatchFailure(
        forallOp, "Expected parallel_insert_slice combining op");
  }

  // Only zero offset extract_slice ops are supported.
  if (!areAllConstantIntValue(extractSliceOp.getMixedOffsets(), 0)) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "extract_slice has non-zero offsets");
  }

  // The extract_slice index operands must dominate the forall loop in order
  // to extract a slice of the init operand later.
  DominanceInfo domInfo;
  int64_t indexOperandStartIdx =
      extractSliceOp.getOffsetSizeAndStrideStartOperandIndex();
  SmallVector<Value> indexOperands(extractSliceOp->getOperands().begin() +
                                       indexOperandStartIdx,
                                   extractSliceOp->getOperands().end());
  if (!llvm::all_of(indexOperands,
                    [&](Value v) { return domInfo.dominates(v, forallOp); })) {
    return rewriter.notifyMatchFailure(
        extractSliceOp,
        "Extract slice index operands do not dominate the forall op");
  }

  // Compute the rank reduction mask of the extract_slice for resolving rank
  // reduction at the end. For rank reducing slices, the extract_slice is
  // fused into the loop as a non rank reducing slice, and then a collapse
  // shape is added on the result of the loop. This simplifies the logic in
  // this pattern, and other patterns for collapse shape fusion can then fuse
  // this collapse shape into the loop if needed.
  auto maybeRankReductionMask = computeRankReductionMask(
      extractSliceOp.getStaticSizes(), extractSliceOp.getType().getShape(),
      /*matchDynamic=*/true);
  if (!maybeRankReductionMask) {
    return rewriter.notifyMatchFailure(extractSliceOp,
                                       "Could not compute rank reduction mask");
  }

  std::optional<llvm::SmallDenseSet<unsigned int>>
      maybeInsertRankReductionMask =
          computeRankReductionMask(parallelInsertOp.getStaticSizes(),
                                   parallelInsertOp.getSourceType().getShape(),
                                   /*matchDynamic=*/true);
  if (!maybeInsertRankReductionMask) {
    return rewriter.notifyMatchFailure(parallelInsertOp,
                                       "Could not compute rank reduction mask");
  }
  llvm::SmallDenseSet<unsigned int> insertRankReductionMask =
      maybeInsertRankReductionMask.value();

  // Verify that the parallelInsertOp can be clamped to the sizes of the
  // extractSliceOp.
  if (failed(canClampParallelInsertSlice(rewriter, parallelInsertOp,
                                         extractSliceOp,
                                         insertRankReductionMask))) {
    return failure();
  }
  int64_t resultIdx = forallResult.getResultNumber();

  // Clamp the parallel insert slice source to fit within the extracted slice.
  SmallVector<OpFoldResult> newInitSizes = extractSliceOp.getMixedSizes();
  FailureOr<tensor::ParallelInsertSliceOp> maybeClampedParallelInsertSliceOp =
      clampParallelInsertSliceOp(rewriter, parallelInsertOp, newInitSizes);
  if (failed(maybeClampedParallelInsertSliceOp)) {
    return failure();
  }
  tensor::ParallelInsertSliceOp clampedParallelInsertSliceOp =
      maybeClampedParallelInsertSliceOp.value();

  // Now replace users of the forall loop init argument with the output operand
  // from outside the loop. Do not replace the clamped parallel insert dest.
  Value forallOutput = forallOp.getOutputs()[forallResult.getResultNumber()];
  rewriter.replaceUsesWithIf(initBbarg, forallOutput, [&](OpOperand &operand) {
    return operand != clampedParallelInsertSliceOp.getDestMutable();
  });

  // Clone the extract_slice, and replace the source with the forall init
  // operand.
  Value forallInit = forallOp.getOutputs()[resultIdx];
  rewriter.setInsertionPoint(forallOp);
  auto extractedInit = rewriter.create<tensor::ExtractSliceOp>(
      forallOp->getLoc(), forallInit, extractSliceOp.getMixedOffsets(),
      extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());

  // Clone the forall op with the extracted init operand to replace the
  // original forall op.
  Location loc = forallOp->getLoc();
  rewriter.setInsertionPoint(forallOp);
  SmallVector<Value> newForallOutputs(forallOp.getOutputs());
  newForallOutputs[resultIdx] = extractedInit.getResult();

  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newForallOutputs, forallOp.getMappingAttr());

  SmallVector<Value> argReplacements(newForallOp.getInductionVars());
  argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                         newForallOp.getRegionIterArgs().end());
  newForallOp.getTerminator()->erase();
  rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(),
                       argReplacements);

  // Create a collapse_shape to handle rank reduction.
  Value extractedResult = newForallOp->getResult(resultIdx);
  auto forallResultType = cast<ShapedType>(extractedResult.getType());
  SmallVector<ReassociationIndices> reassociations;
  ReassociationIndices reassociation;
  for (int i = 0; i < forallResultType.getRank(); ++i) {
    if (maybeRankReductionMask->contains(i)) {
      reassociation.push_back(i);
      continue;
    }
    reassociation.push_back(i);
    reassociations.push_back(reassociation);
    reassociation = {};
  }
  auto collapseShape = rewriter.create<tensor::CollapseShapeOp>(
      extractSliceOp->getLoc(), extractedResult, reassociations);

  // Replace forall and extract_slice ops with the new operations.
  rewriter.replaceAllOpUsesWith(extractSliceOp, collapseShape);
  rewriter.replaceOp(forallOp, newForallOp);
  return newForallOp;
}

//===----------------------------------------------------------------------===//
// InnerTiledOp lowering to underlying operation
//===----------------------------------------------------------------------===//

namespace {
struct LowerInnerTiledPattern
    : public OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using OpRewritePattern<IREE::Codegen::InnerTiledOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "lowering to concrete op requires vector semantics");
    }
    SmallVector<int64_t> bounds;
    tiledOp.getIterationBounds(bounds);
    if (!bounds.empty()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "must be a single inner tiled operation");
    }

    SmallVector<Value> operands = tiledOp.getOperands();
    SmallVector<VectorType> regTypes;
    tiledOp.getKind().getDistributedTileTypes(regTypes);

    for (auto [operand, regType] : llvm::zip_equal(operands, regTypes)) {
      if (operand.getType() != regType) {
        operand = rewriter.create<vector::ShapeCastOp>(tiledOp.getLoc(),
                                                       regType, operand);
      }
    }

    SmallVector<Value> concreteResults;
    int64_t numInputs = tiledOp.getNumInputs();
    LogicalResult couldLower = tiledOp.getKind().buildUnderlyingOperations(
        rewriter, tiledOp.getLoc(), ValueRange{operands}.take_front(numInputs),
        ValueRange{operands}.drop_front(numInputs), concreteResults);
    if (failed(couldLower)) {
      tiledOp.emitOpError(
          "failed to lower to concrete inner tiled operations.");
      return failure();
    }
    for (auto [result, externalShape] :
         llvm::zip_equal(concreteResults, tiledOp.getResultTypes())) {
      result = rewriter.create<vector::ShapeCastOp>(tiledOp.getLoc(),
                                                    externalShape, result);
    }
    rewriter.replaceOp(tiledOp, concreteResults);
    return success();
  }
};
} // namespace

void populateIREEGPULowerInnerTiledPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerInnerTiledPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Conversion to MultiMmaOp
//===----------------------------------------------------------------------===//

AffineMap dropDims(MLIRContext *context, int64_t newDimCount, AffineMap map,
                   llvm::SmallDenseMap<int64_t, int64_t> &oldDimsToNewDimsMap) {
  assert(map.isProjectedPermutation() && "expected projected permutation");

  SmallVector<AffineExpr> newResults;
  for (auto expr : map.getResults()) {
    int64_t dimPos = cast<AffineDimExpr>(expr).getPosition();
    if (!oldDimsToNewDimsMap.contains(dimPos)) {
      continue;
    }
    newResults.push_back(
        getAffineDimExpr(oldDimsToNewDimsMap[dimPos], context));
  }
  return AffineMap::get(/*dimCount=*/newDimCount, /*symbolCount=*/0, newResults,
                        context);
}

FailureOr<IREE::Codegen::InnerTiledOp> convertScaledContractionToInnerTiledMma(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp,
    IREE::Codegen::InnerTileDescAttrInterface kind) {
  auto smmaKind = dyn_cast<IREE::GPU::ScaledMMAAttr>(kind);
  FailureOr<IREE::LinalgExt::ScaledContractionDimensions> contractionDims =
      IREE::LinalgExt::inferScaledContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return failure();
  }

  if (contractionDims->m.empty() || contractionDims->n.empty() ||
      contractionDims->k.empty() || contractionDims->kB.empty()) {
    return failure();
  }

  MLIRContext *context = rewriter.getContext();

  int64_t innerM = contractionDims->m.back();
  int64_t innerN = contractionDims->n.back();
  int64_t innerK = contractionDims->k.back();
  int64_t innerKb = contractionDims->kB.back();

  AffineExpr mExpr = rewriter.getAffineDimExpr(innerM);
  AffineExpr nExpr = rewriter.getAffineDimExpr(innerN);
  AffineExpr kExpr = rewriter.getAffineDimExpr(innerK);
  AffineExpr kBExpr = rewriter.getAffineDimExpr(innerKb);

  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  AffineMap lhsMap = indexingMaps[0];
  AffineMap rhsMap = indexingMaps[1];
  AffineMap sc1Map = indexingMaps[2];
  AffineMap sc2Map = indexingMaps[3];
  AffineMap accMap = indexingMaps[4];

  auto getNormalizedPermutation =
      [&](AffineMap map,
          ArrayRef<AffineExpr> expectedDimOrder) -> SmallVector<int64_t> {
    llvm::SmallDenseMap<AffineExpr, int64_t> dimMap;
    for (auto [i, expr] : llvm::enumerate(expectedDimOrder)) {
      dimMap[expr] = i;
    }
    SmallVector<int64_t> permutation;
    for (AffineExpr resExpr : map.getResults()) {
      if (!dimMap.contains(resExpr)) {
        return {};
      }
      permutation.push_back(dimMap[resExpr]);
    }
    return permutation;
  };

  // TODO: Enable batched intrinsics and get the appropriate sub-map here.
  SmallVector<int64_t> lhsInnerPerm = getNormalizedPermutation(
      lhsMap.getMinorSubMap(3), {mExpr, kExpr, kBExpr});
  SmallVector<int64_t> sc1InnerPerm =
      getNormalizedPermutation(sc1Map.getMinorSubMap(2), {mExpr, kExpr});
  SmallVector<int64_t> rhsInnerPerm = getNormalizedPermutation(
      rhsMap.getMinorSubMap(3), {kExpr, kBExpr, nExpr});
  SmallVector<int64_t> sc2InnerPerm =
      getNormalizedPermutation(sc2Map.getMinorSubMap(2), {nExpr, kExpr});
  SmallVector<int64_t> accInnerPerm =
      getNormalizedPermutation(accMap.getMinorSubMap(2), {mExpr, nExpr});
  if (lhsInnerPerm.empty() || sc1InnerPerm.empty() || rhsInnerPerm.empty() ||
      sc2InnerPerm.empty() || accInnerPerm.empty()) {
    return failure();
  }

  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  auto [intrinsicM, intrinsicN, intrinsicK, intrinsicKB] =
      smmaKind.getScaledMNKShape();
  if (intrinsicM != bounds[innerM] || intrinsicN != bounds[innerN] ||
      intrinsicK != bounds[innerK]) {
    return failure();
  }

  auto reorderInputs = [](SmallVector<Value> inputs) {
    SmallVector<Value> res;
    int numInputs = inputs.size() - 1;
    int offset = numInputs / 2;
    for (int i = 0; i < numInputs; i++) {
      res.push_back(inputs[i / 2 + (i % 2) * offset]);
    }
    res.push_back(inputs.back());
    return res;
  };
  SmallVector<Value> inputs = reorderInputs(linalgOp->getOperands());

  SmallVector<Type> eltTypes;
  smmaKind.getElementTypes(eltTypes);
  if (cast<RankedTensorType>(inputs[0].getType()).getElementType() !=
          eltTypes[0] ||
      cast<RankedTensorType>(inputs[2].getType()).getElementType() !=
          eltTypes[2] ||
      cast<RankedTensorType>(inputs[4].getType()).getElementType() !=
          eltTypes[4]) {
    return failure();
  }

  SmallVector<utils::IteratorType> linalgIteratorTypes =
      linalgOp.getIteratorTypesArray();
  llvm::SmallDenseSet<int64_t> droppedDims = {innerM, innerN, innerK, innerKb};
  llvm::SmallDenseMap<int64_t, int64_t> oldDimsToNewDimsMap;
  int64_t currentDim = 0;
  int64_t numDims = lhsMap.getNumDims();
  SmallVector<utils::IteratorType> iteratorTypes;
  for (int64_t dim = 0, e = numDims; dim < e; ++dim) {
    if (droppedDims.contains(dim)) {
      continue;
    }
    iteratorTypes.push_back(linalgIteratorTypes[dim]);
    oldDimsToNewDimsMap[dim] = currentDim++;
  }
  AffineMap outerLhsMap =
      dropDims(context, numDims - 4, lhsMap, oldDimsToNewDimsMap);
  AffineMap outerRhsMap =
      dropDims(context, numDims - 4, rhsMap, oldDimsToNewDimsMap);
  AffineMap outerSc1Map =
      dropDims(context, numDims - 4, sc1Map, oldDimsToNewDimsMap);
  AffineMap outerSc2Map =
      dropDims(context, numDims - 4, sc2Map, oldDimsToNewDimsMap);
  AffineMap outerAccMap =
      dropDims(context, numDims - 4, accMap, oldDimsToNewDimsMap);
  std::optional<SmallVector<SmallVector<int64_t>>> perms =
      SmallVector<SmallVector<int64_t>>{
          lhsInnerPerm, sc1InnerPerm, rhsInnerPerm, sc2InnerPerm, accInnerPerm};
  SmallVector<int64_t> identityPerm = {0, 1};
  if (lhsInnerPerm == identityPerm && rhsInnerPerm == identityPerm &&
      accInnerPerm == identityPerm)
    perms = std::nullopt;

  IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
      getLoweringConfig(linalgOp);
  auto newMmaOp = rewriter.replaceOpWithNewOp<IREE::Codegen::InnerTiledOp>(
      linalgOp, /*inputs=*/ValueRange{inputs}.drop_back(),
      /*inits=*/ValueRange{inputs}.back(),
      ArrayRef<AffineMap>{outerLhsMap, outerSc1Map, outerRhsMap, outerSc2Map,
                          outerAccMap},
      iteratorTypes, smmaKind, perms);
  if (maybeLoweringConfig) {
    setLoweringConfig(newMmaOp, maybeLoweringConfig);
  }
  return newMmaOp;
}

// Helper to convert a contraction-like linalg op to an iree_codegen.inner_tiled
// op with a MMA-like intrinsic descriptor.
FailureOr<IREE::Codegen::InnerTiledOp> convertContractionToInnerTiledMma(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp,
    IREE::Codegen::InnerTileDescAttrInterface kind) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

  FailureOr<IREE::LinalgExt::ScaledContractionDimensions> maybeScaledContrDims =
      IREE::LinalgExt::inferScaledContractionDims(linalgOp);
  if (succeeded(maybeScaledContrDims)) {
    return convertScaledContractionToInnerTiledMma(rewriter, linalgOp, kind);
  }

  IREE::GPU::MmaInterfaceAttr mmaKind =
      dyn_cast<IREE::GPU::MmaInterfaceAttr>(kind);
  FailureOr<linalg::ContractionDimensions> maybeContractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(maybeContractionDims)) {
    return failure();
  }

  linalg::ContractionDimensions contractionDims = *maybeContractionDims;
  if (contractionDims.m.empty() || contractionDims.n.empty() ||
      contractionDims.k.empty()) {
    return failure();
  }

  MLIRContext *context = rewriter.getContext();

  int64_t innerM = contractionDims.m.back();
  int64_t innerN = contractionDims.n.back();
  int64_t innerK = contractionDims.k.back();

  AffineExpr d0, d1, d2;
  bindDims(context, d0, d1, d2);
  llvm::SmallDenseMap<AffineExpr, AffineExpr> newDims;
  AffineExpr mExpr = rewriter.getAffineDimExpr(innerM);
  AffineExpr nExpr = rewriter.getAffineDimExpr(innerN);
  AffineExpr kExpr = rewriter.getAffineDimExpr(innerK);

  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  AffineMap lhsMap = indexingMaps[0];
  AffineMap rhsMap = indexingMaps[1];
  AffineMap accMap = indexingMaps[2];

  auto getNormalizedPermutation =
      [&](AffineMap map,
          ArrayRef<AffineExpr> expectedDimOrder) -> SmallVector<int64_t> {
    llvm::SmallDenseMap<AffineExpr, int64_t> dimMap;
    for (auto [i, expr] : llvm::enumerate(expectedDimOrder)) {
      dimMap[expr] = i;
    }
    SmallVector<int64_t> permutation;
    for (AffineExpr resExpr : map.getResults()) {
      if (!dimMap.contains(resExpr)) {
        return {};
      }
      permutation.push_back(dimMap[resExpr]);
    }
    return permutation;
  };

  // TODO: Enable batched intrinsics and get the appropriate sub-map here.
  SmallVector<int64_t> lhsInnerPerm =
      getNormalizedPermutation(lhsMap.getMinorSubMap(2), {mExpr, kExpr});
  SmallVector<int64_t> rhsInnerPerm =
      getNormalizedPermutation(rhsMap.getMinorSubMap(2), {kExpr, nExpr});
  SmallVector<int64_t> accInnerPerm =
      getNormalizedPermutation(accMap.getMinorSubMap(2), {mExpr, nExpr});

  if (lhsInnerPerm.empty() || rhsInnerPerm.empty() || accInnerPerm.empty()) {
    return failure();
  }

  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();

  auto [intrinsicM, intrinsicN, intrinsicK] = mmaKind.getMNKShape();
  if (intrinsicM != bounds[innerM] || intrinsicN != bounds[innerN] ||
      intrinsicK != bounds[innerK]) {
    return failure();
  }

  SmallVector<Value> inputs = linalgOp->getOperands();
  auto [lhsElementType, rhsElementType, accElementType] =
      mmaKind.getABCElementTypes();
  if (cast<RankedTensorType>(inputs[0].getType()).getElementType() !=
          lhsElementType ||
      cast<RankedTensorType>(inputs[1].getType()).getElementType() !=
          rhsElementType ||
      cast<RankedTensorType>(inputs[2].getType()).getElementType() !=
          accElementType) {
    return failure();
  }

  SmallVector<utils::IteratorType> linalgIteratorTypes =
      linalgOp.getIteratorTypesArray();
  llvm::SmallDenseSet<int64_t> droppedDims = {innerM, innerN, innerK};
  llvm::SmallDenseMap<int64_t, int64_t> oldDimsToNewDimsMap;
  int64_t currentDim = 0;
  int64_t numDims = lhsMap.getNumDims();
  SmallVector<utils::IteratorType> iteratorTypes;
  for (int64_t dim = 0, e = numDims; dim < e; ++dim) {
    if (droppedDims.contains(dim)) {
      continue;
    }
    iteratorTypes.push_back(linalgIteratorTypes[dim]);
    oldDimsToNewDimsMap[dim] = currentDim++;
  }

  AffineMap outerLhsMap =
      dropDims(context, numDims - 3, lhsMap, oldDimsToNewDimsMap);
  AffineMap outerRhsMap =
      dropDims(context, numDims - 3, rhsMap, oldDimsToNewDimsMap);
  AffineMap outerAccMap =
      dropDims(context, numDims - 3, accMap, oldDimsToNewDimsMap);

  std::optional<SmallVector<SmallVector<int64_t>>> perms =
      SmallVector<SmallVector<int64_t>>{lhsInnerPerm, rhsInnerPerm,
                                        accInnerPerm};
  SmallVector<int64_t> identityPerm = {0, 1};

  if (lhsInnerPerm == identityPerm && rhsInnerPerm == identityPerm &&
      accInnerPerm == identityPerm)
    perms = std::nullopt;

  IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
      getLoweringConfig(linalgOp);

  auto newMmaOp = rewriter.replaceOpWithNewOp<IREE::Codegen::InnerTiledOp>(
      linalgOp, /*inputs=*/ValueRange{inputs}.drop_back(),
      /*inits=*/ValueRange{inputs}.back(),
      ArrayRef<AffineMap>{outerLhsMap, outerRhsMap, outerAccMap}, iteratorTypes,
      mmaKind, perms);
  if (maybeLoweringConfig) {
    setLoweringConfig(newMmaOp, maybeLoweringConfig);
  }
  return newMmaOp;
}

//===----------------------------------------------------------------------===//
// InnerTiledOp Distribution
//===----------------------------------------------------------------------===//

FailureOr<Operation *>
distributeInnerTiledOp(RewriterBase &rewriter,
                       IREE::Codegen::InnerTiledOp tiledOp) {
  if (!tiledOp.hasTensorSemantics() || tiledOp.hasThreadSemantics()) {
    return rewriter.notifyMatchFailure(
        tiledOp, "tiledOp must have vector and subgroup for distribution.");
  }

  RewriterBase::InsertionGuard g(rewriter);

  Location loc = tiledOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);

  // Step 1. Create the new scf.forall op with a lane id mapping.
  Attribute mappingType = tiledOp.getKind().getDistributionMappingKind();
  if (!mappingType) {
    return rewriter.notifyMatchFailure(
        tiledOp, "doesn't specify how it's to be distributed");
  }
  OpFoldResult ub =
      tiledOp.getKind().getDistributionWorkerCount(rewriter, loc, tiledOp);
  if (!ub) {
    return tiledOp.emitOpError("failed to specify a worker count for the "
                               "forall it's to be distributed into.");
  }

  auto newForallOp = rewriter.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>{zero}, ArrayRef<OpFoldResult>{ub},
      ArrayRef<OpFoldResult>{one}, tiledOp.getOutputs(),
      ArrayAttr::get(context, {mappingType}));

  rewriter.setInsertionPointToStart(newForallOp.getBody());

  // Step 2. Compute the offsets/sizes/strides for each of the operands.
  Value id = newForallOp.getInductionVar(0);
  SmallVector<Value> inputSlices, initSlices;
  SmallVector<tensor::ExtractSliceOp> initSliceOps;
  std::optional<ArrayAttr> maybePerms = tiledOp.getPermutations();
  int64_t firstOutIdx = tiledOp.getNumInputs();

  for (auto [opIndex, operand] : llvm::enumerate(tiledOp.getOperands())) {
    int64_t outerRank = tiledOp.getOperandOuterRank(opIndex);
    SmallVector<OpFoldResult> offsets(outerRank, zero);
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0, e = outerRank; i < e; ++i) {
      sizes.push_back(tensor::getMixedSize(rewriter, loc, operand, i));
    }
    ArrayRef<int64_t> innerShape = tiledOp.getOperandInnerShape(opIndex);
    SmallVector<int64_t> permutation;
    if (maybePerms) {
      permutation = llvm::to_vector(
          cast<DenseI64ArrayAttr>((*maybePerms)[opIndex]).asArrayRef());
    } else {
      permutation = llvm::to_vector(llvm::seq(
          static_cast<int64_t>(0), static_cast<int64_t>(innerShape.size())));
    }
    // Slice offsets.
    SmallVector<OpFoldResult> strides(outerRank, one);
    if (failed(tiledOp.getKind().populateOperandOffsetsSizesStrides(
            rewriter, loc, opIndex, id, permutation, offsets, sizes,
            strides))) {
      return tiledOp->emitOpError("failed to populate offsets for operand " +
                                  Twine(opIndex));
    }
    // Extract the rank-reduced slice of the operand based on the expected inner
    // vector shape. If we're slicing the accumulator, extract from the loop
    // variable, since the accumulator operand has been used to create the
    // forall.
    if (opIndex >= firstOutIdx) {
      auto sliceOp = rewriter.create<tensor::ExtractSliceOp>(
          loc, newForallOp.getRegionIterArgs()[opIndex - firstOutIdx], offsets,
          sizes, strides);
      initSliceOps.push_back(sliceOp);
      initSlices.push_back(sliceOp);
    } else {
      Value slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, operand, offsets, sizes, strides);
      inputSlices.push_back(slice);
    }
  }

  // Step 3. Create the new inner_tiled op.
  auto newTiledOp = rewriter.create<IREE::Codegen::InnerTiledOp>(
      loc, inputSlices, initSlices, tiledOp.getIndexingMaps(),
      tiledOp.getIteratorTypes(), tiledOp.getKind());

  newTiledOp->setDiscardableAttrs(tiledOp->getDiscardableAttrDictionary());

  // Step 4. Insert the result of the inner_tiled using the same offsets/sizes
  // as the accumulator slice.
  scf::InParallelOp terminator = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(terminator.getBody());
  for (auto [newResult, extractOp] :
       llvm::zip_equal(newTiledOp.getResults(), initSliceOps)) {
    rewriter.create<tensor::ParallelInsertSliceOp>(
        loc, newResult, extractOp.getSource(), extractOp.getMixedOffsets(),
        extractOp.getMixedSizes(), extractOp.getMixedStrides());
  }

  rewriter.replaceOp(tiledOp, newForallOp);

  return &*newForallOp;
}

//===----------------------------------------------------------------------===//
// InnerTiledOp Unit Dim Folding
//===----------------------------------------------------------------------===//

namespace {
struct DropInnerTiledUnitDimsPattern
    : public OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using OpRewritePattern<IREE::Codegen::InnerTiledOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "unimplemented: unit dim dropping for tensor mma ops");
    }
    SmallVector<int64_t> bounds;
    tiledOp.getIterationBounds(bounds);
    if (bounds.empty()) {
      return rewriter.notifyMatchFailure(tiledOp, "no dimensions to fold");
    }

    // TODO: Generalize to allow only some iteration bounds to be unit. This
    // pattern currently only supports the most common case of unrolling to the
    // intrinsic shape.
    if (!llvm::all_of(bounds, [](int64_t b) { return b == 1; })) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "not all iteration bounds are unit");
    }

    Location loc = tiledOp.getLoc();
    SmallVector<Value> newOperands;
    for (auto [opIndex, operand] : llvm::enumerate(tiledOp.getOperands())) {
      int64_t outerRank = tiledOp.getOperandOuterRank(opIndex);
      if (outerRank == 0) {
        newOperands.push_back(operand);
        continue;
      }
      SmallVector<int64_t> droppedDimIndices(outerRank, 0);
      Value slice =
          rewriter.create<vector::ExtractOp>(loc, operand, droppedDimIndices);
      newOperands.push_back(slice);
    }

    SmallVector<AffineMap> emptyMaps(tiledOp.getNumOperands(),
                                     AffineMap::get(rewriter.getContext()));
    auto newTiledOp = rewriter.create<IREE::Codegen::InnerTiledOp>(
        loc, ValueRange{newOperands}.take_front(tiledOp.getNumInputs()),
        ValueRange{newOperands}.drop_front(tiledOp.getNumInputs()),
        rewriter.getAffineMapArrayAttr(emptyMaps), rewriter.getArrayAttr({}),
        tiledOp.getKind());

    SmallVector<Value> newResults(newTiledOp.getResults());
    for (auto [newResult, externalShape] :
         llvm::zip_equal(newResults, tiledOp.getResultTypes())) {
      newResult =
          rewriter.create<vector::BroadcastOp>(loc, externalShape, newResult);
    }
    rewriter.replaceOp(tiledOp, newResults);
    return success();
  }
};
} // namespace

void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns) {
  patterns.add<DropInnerTiledUnitDimsPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// InnerTiledOp Unrolling
//===----------------------------------------------------------------------===//

static SmallVector<int64_t>
getUnrollOrder(unsigned numLoops, Operation *op,
               const vector::UnrollVectorOptions &options) {
  SmallVector<int64_t> loopOrder =
      llvm::to_vector(llvm::seq<int64_t>(0, static_cast<int64_t>(numLoops)));
  if (options.traversalOrderCallback != nullptr) {
    std::optional<SmallVector<int64_t>> order =
        options.traversalOrderCallback(op);
    if (order) {
      loopOrder = std::move(*order);
    }
  }
  return loopOrder;
}

namespace {

/// Helper structure to track partially accumulated values while unrolling.
struct OffsetMapInfo {
  static SmallVector<int64_t> getEmptyKey() { return {int64_t(-1)}; }

  static SmallVector<int64_t> getTombstoneKey() { return {int64_t(-2)}; }

  static unsigned getHashValue(const SmallVector<int64_t> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
  }

  static bool isEqual(const SmallVector<int64_t> &lhs,
                      const SmallVector<int64_t> &rhs) {
    return lhs == rhs;
  }
};

struct UnrollInnerTiledPattern
    : public OpRewritePattern<Codegen::InnerTiledOp> {
  UnrollInnerTiledPattern(MLIRContext *context,
                          const vector::UnrollVectorOptions &options,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<Codegen::InnerTiledOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          tiledOp, "don't know how to unroll multiple accumulators yet");
    }
    if (options.filterConstraint && failed(options.filterConstraint(tiledOp))) {
      return rewriter.notifyMatchFailure(tiledOp, "unrolling filter");
    }
    assert(options.nativeShape &&
           "vector unrolling expects the native shape or native shape call "
           "back function to be set");
    std::optional<SmallVector<int64_t, 4>> maybeUnrollShape =
        tiledOp.getShapeForUnroll();
    if (!maybeUnrollShape) {
      return rewriter.notifyMatchFailure(
          tiledOp, "unexpected failure to get unroll shape");
    }

    std::optional<SmallVector<int64_t>> targetShape =
        options.nativeShape(tiledOp);
    if (!targetShape) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "unspecified native unroll shape");
    }

    auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
    if (!maybeShapeRatio) {
      return rewriter.notifyMatchFailure(
          tiledOp, "operation unroll shape not divisible by target shape");
    }

    // Early exit if unrolling has no effect.
    if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
      return rewriter.notifyMatchFailure(
          tiledOp, "operation already unrolled to native shape");
    }

    auto dstVecType = cast<VectorType>(tiledOp.getResultTypes().front());
    SmallVector<int64_t, 4> originalSize = *maybeUnrollShape;

    Location loc = tiledOp.getLoc();
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(tiledOp.getIteratorTypes().size(), tiledOp, options);

    SmallVector<AffineMap> permutationMaps = tiledOp.getIndexingMapsArray();
    int64_t accIndex = permutationMaps.size() - 1;
    AffineMap accPermutationMap = permutationMaps.back();
    ArrayRef<int64_t> innerAccShape = tiledOp.getOperandInnerShape(accIndex);

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<Value> slicesOperands(tiledOp.getNumOperands());

      // Helper to compute the new shape of each operand and extract the slice.
      auto extractOperand = [&](unsigned index, Value operand,
                                AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffets) {
        SmallVector<int64_t> operandShape = applyPermutationMap(
            permutationMap, ArrayRef<int64_t>(*targetShape));
        SmallVector<int64_t> operandStrides(operandOffets.size(), 1);
        slicesOperands[index] = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, operand, operandOffets, operandShape, operandStrides);
      };
      // Extract the new input operands.
      for (auto [inputIndex, input] : llvm::enumerate(tiledOp.getInputs())) {
        SmallVector<int64_t> inOffsets = applyPermutationMap(
            permutationMaps[inputIndex], ArrayRef<int64_t>(offsets));
        extractOperand(inputIndex, input, permutationMaps[inputIndex],
                       inOffsets);
      }

      SmallVector<int64_t> accOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // If a version of the accumulator has already been computed, use it
      // otherwise extract the first version from the original operand.
      auto *accIt = accCache.find(accOffets);
      if (accIt != accCache.end()) {
        slicesOperands[accIndex] = accIt->second;
      } else {
        extractOperand(accIndex, tiledOp.getOutputs().front(),
                       accPermutationMap, accOffets);
      }

      SmallVector<int64_t> dstShape = applyPermutationMap(
          accPermutationMap, ArrayRef<int64_t>(*targetShape));
      dstShape.append(innerAccShape.begin(), innerAccShape.end());
      auto targetType = VectorType::get(dstShape, dstVecType.getElementType());

      // Clone the inner tiled op with the new operands and result type.
      IREE::Codegen::InnerTiledOp newOp =
          mlir::clone(rewriter, tiledOp, targetType, slicesOperands);

      SmallVector<int64_t> dstOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // Save the accumulated value until all the loops are unrolled since
      // reduction loop keep updating the accumulator.
      accCache[dstOffets] = newOp.getResults().front();
    }
    // Assemble back the accumulator into a single vector.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstVecType, rewriter.getZeroAttr(dstVecType));
    for (const auto &[offsets, partialResult] : accCache) {
      SmallVector<int64_t> dstStrides(offsets.size() + innerAccShape.size(), 1);
      SmallVector<int64_t> fullOffsets(offsets);
      fullOffsets.append(innerAccShape.size(), 0);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, partialResult, result, fullOffsets, dstStrides);
    }
    rewriter.replaceOp(tiledOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};
} // namespace

void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options) {
  patterns.add<UnrollInnerTiledPattern>(patterns.getContext(), options);
}

static bool isReductionIterator(Attribute attr) {
  return cast<linalg::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<linalg::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register, assuming that the LHS
/// register is the first argument.
static std::optional<SmallVector<int64_t>>
gpuMatmulLikeUnrollOrder(Operation *op) {
  IREE::Codegen::InnerTiledOp tiledOp =
      dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!tiledOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dimsInLhs;
  for (AffineExpr expr : tiledOp.getIndexingMapsArray()[0].getResults()) {
    dimsInLhs.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>>
getInnerTiledUnitShape(Operation *op) {
  auto tiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!tiledOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(tiledOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void populateIREEGPUVectorUnrollPatterns(RewritePatternSet &patterns) {
  populateIREEGPUVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getInnerTiledUnitShape)
                    .setUnrollTraversalOrderFn(gpuMatmulLikeUnrollOrder));
}

//===---------------------------------------------------------------------===//
// Resolving lane mapped forall ops
//===---------------------------------------------------------------------===//

static bool isLaneMappableForall(scf::ForallOp forallOp) {
  if (forallOp.getNumResults() > 0)
    return false;
  if (forallOp.getRank() != 1)
    return false;
  if (!forallOp.getMapping().has_value())
    return false;
  Attribute mapping = *forallOp.getMapping()->getValue().begin();
  if (mapping != IREE::GPU::LaneIdAttr::get(forallOp.getContext(), 0)) {
    return false;
  }
  return true;
}

static void rewriteForallToLanes(RewriterBase &rewriter, scf::ForallOp forallOp,
                                 bool insertBarrier) {
  Location loc = forallOp->getLoc();
  assert(isLaneMappableForall(forallOp) && "mapping non-lane forall op");

  auto upperBounds = forallOp.getLoopUpperBounds();
  std::optional<IntegerAttr> upperBound;
  if (upperBounds && upperBounds->size() > 0) {
    if (auto upperBoundAttr = (*upperBounds)[0].dyn_cast<Attribute>()) {
      upperBound = dyn_cast<IntegerAttr>(upperBoundAttr);
    }
  }
  Value laneId = rewriter.create<gpu::LaneIdOp>(
      loc, upperBound ? rewriter.getIndexAttr(upperBound->getInt()) : nullptr);
  rewriter.eraseOp(forallOp.getTerminator());
  rewriter.setInsertionPoint(forallOp);
  rewriter.inlineBlockBefore(forallOp.getBody(), forallOp, {laneId});
  if (insertBarrier) {
    rewriter.create<gpu::BarrierOp>(loc);
  }
  rewriter.eraseOp(forallOp);
}

void mapLaneForalls(RewriterBase &rewriter, Operation *funcOp,
                    bool insertBarrier) {
  SmallVector<scf::ForallOp> foralls;
  OpBuilder::InsertionGuard g(rewriter);
  funcOp->walk([&](scf::ForallOp forallOp) {
    if (isLaneMappableForall(forallOp)) {
      foralls.push_back(forallOp);
    }
  });
  for (auto forall : foralls) {
    rewriter.setInsertionPoint(forall);
    rewriteForallToLanes(rewriter, forall, insertBarrier);
  }
}

//===---------------------------------------------------------------------===//
// BarrierRegion Lowering
//===---------------------------------------------------------------------===//

namespace {
struct LowerBarrierRegion
    : public OpRewritePattern<IREE::GPU::BarrierRegionOp> {
  using OpRewritePattern<IREE::GPU::BarrierRegionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::BarrierRegionOp barrierRegionOp,
                                PatternRewriter &rewriter) const final {
    Location loc = barrierRegionOp.getLoc();

    // Step 1. Synchronize the workers on the shared dest.
    auto writeBarrier = rewriter.create<IREE::GPU::ValueBarrierOp>(
        loc, barrierRegionOp.getInputs());

    // Step 2. Inline the barrier op region.
    auto terminator = barrierRegionOp.getBody()->getTerminator();
    rewriter.inlineBlockBefore(barrierRegionOp.getBody(), barrierRegionOp,
                               writeBarrier.getResults());
    rewriter.setInsertionPoint(terminator);

    // Step 3. Synchronize the result value.
    auto barrier = rewriter.create<IREE::GPU::ValueBarrierOp>(
        loc, terminator->getOperands());
    rewriter.replaceAllUsesWith(barrierRegionOp.getResults(),
                                barrier.getResults());
    rewriter.eraseOp(terminator);
    return success();
  }
};
} // namespace

void populateIREEGPULowerBarrierRegionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerBarrierRegion>(patterns.getContext());
}

//===---------------------------------------------------------------------===//
// InnerTiledOp Vectorization
//===---------------------------------------------------------------------===//

static LogicalResult
vectorizeStaticInnerTiledOp(RewriterBase &rewriter,
                            IREE::Codegen::InnerTiledOp tiledOp) {
  if (!tiledOp.hasTensorSemantics()) {
    return failure();
  }
  SmallVector<ShapedType> argTypes = tiledOp.getOperandShapedTypes();
  if (!llvm::all_of(argTypes, [](auto st) { return st.hasStaticShape(); })) {
    return rewriter.notifyMatchFailure(tiledOp,
                                       "non-static shape for vectorization");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(tiledOp);

  Location loc = tiledOp.getLoc();

  // Construct the (never used) zero padding value for each operand.
  SmallVector<Value> padValues =
      llvm::map_to_vector(argTypes, [&](ShapedType argType) -> Value {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(argType.getElementType()));
      });

  SmallVector<Value> newOperands = tiledOp.getOperands();
  for (auto [operand, type, padValue] :
       llvm::zip_equal(newOperands, argTypes, padValues)) {
    operand = vector::createReadOrMaskedRead(
        rewriter, loc, operand, type.getShape(), padValue,
        /*useInBoundsInsteadOfMasking=*/true);
  }
  auto newTiledOp = rewriter.create<IREE::Codegen::InnerTiledOp>(
      loc, ValueRange{newOperands}.take_front(tiledOp.getNumInputs()),
      ValueRange{newOperands}.take_back(tiledOp.getNumOutputs()),
      tiledOp.getIndexingMaps(), tiledOp.getIteratorTypes(), tiledOp.getKind());

  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> transferWrites;
  for (auto [result, tensorAcc] :
       llvm::zip_equal(newTiledOp.getResults(), tiledOp.getOutputs())) {
    // Create the write back to a tensor.
    int64_t rank = cast<RankedTensorType>(tensorAcc.getType()).getRank();
    auto write = rewriter.create<vector::TransferWriteOp>(
        loc,
        /*vector=*/result,
        /*source=*/tensorAcc,
        /*indices=*/SmallVector<Value>(rank, zero),
        /*inBounds=*/SmallVector<bool>(rank, true));
    transferWrites.push_back(write.getResults().front());
  }
  rewriter.replaceOp(tiledOp, transferWrites);
  return success();
}

namespace {
struct VectorizeStaticInnerTiledOpPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using OpRewritePattern<IREE::Codegen::InnerTiledOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    return vectorizeStaticInnerTiledOp(rewriter, tiledOp);
  }
};
} // namespace

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeStaticInnerTiledOpPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// VectorBarrierOp Lowering
//===----------------------------------------------------------------------===//

namespace {
struct LowerValueBarrierPattern
    : public OpRewritePattern<IREE::GPU::ValueBarrierOp> {
  using OpRewritePattern<IREE::GPU::ValueBarrierOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::ValueBarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    if (barrier.hasTensorSemantics()) {
      return failure();
    }
    rewriter.create<gpu::BarrierOp>(barrier.getLoc());
    for (auto [result, input] :
         llvm::zip_equal(barrier.getResults(), barrier.getInputs())) {
      rewriter.replaceAllUsesWith(result, input);
    }
    return success();
  }
};

struct LowerGlobalLoadDMAPattern
    : public OpRewritePattern<IREE::GPU::GlobalLoadDMAOp> {
  using OpRewritePattern<IREE::GPU::GlobalLoadDMAOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::GlobalLoadDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    Type transferType = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<amdgpu::GatherToLDSOp>(
        dmaOp, dmaOp.getSource(), dmaOp.getSourceIndices(), dmaOp.getTarget(),
        dmaOp.getTargetIndices(), transferType);
    return success();
  }
};
} // namespace

void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerValueBarrierPattern>(patterns.getContext());
}

void populateIREEGPULowerGlobalLoadDMAPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerGlobalLoadDMAPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
