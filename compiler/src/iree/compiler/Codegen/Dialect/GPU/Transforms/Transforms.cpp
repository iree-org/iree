// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
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
      empty->getLoc(), empty->getResultTypes()[0], empty.getDynamicSizes());
  allocTensor.setMemorySpaceAttr(sharedMemoryAddrSpace);
  return allocTensor.getResult();
}

static void replaceConsumerChain(RewriterBase &rewriter, Location loc,
                                 Value source, Value replacement,
                                 SmallVector<Operation *> consumerChain) {
  auto extractSlice = cast<tensor::ExtractSliceOp>(consumerChain.back());
  OpBuilder::InsertionGuard g(rewriter);

  auto barrierRegionOp = rewriter.create<IREE::GPU::BarrierRegionOp>(
      loc, extractSlice.getType(), replacement);
  rewriter.setInsertionPointToStart(barrierRegionOp.getBody());
  auto terminator =
      rewriter.create<IREE::GPU::YieldOp>(loc, extractSlice.getResult());
  for (auto consumer : consumerChain) {
    rewriter.moveOpBefore(consumer, terminator);
  }
  (*consumerChain.begin())
      ->replaceUsesOfWith(source, barrierRegionOp.getBody()->getArgument(0));
  rewriter.replaceAllUsesExcept(extractSlice.getResult(), barrierRegionOp,
                                terminator);
}

LogicalResult fuseForallIntoSlice(RewriterBase &rewriter,
                                  scf::ForallOp producer,
                                  scf::ForallOp consumer,
                                  SmallVector<Operation *> consumerChain) {
  if (consumerChain.empty()) {
    return failure();
  }

  auto slice = dyn_cast<tensor::ExtractSliceOp>(consumerChain.back());
  if (!slice) {
    return failure();
  }

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

  rewriter.setInsertionPoint(slice);

  // Step 1. Get the destination of the producer loop as a shared memory
  // allocation.
  FailureOr<Value> sharedDest =
      createSharedAllocDestination(rewriter, producer);
  if (failed(sharedDest)) {
    return failure();
  }

  // Step 2. Compute the producer IDs in terms of the consumer IDs.
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
  SmallVector<Value> producerRanges;
  OpFoldResult producerWorkerCount = rewriter.getIndexAttr(1);
  for (auto workerCount : producer.getMixedUpperBound()) {
    producerRanges.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, workerCount));
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

  // Step 3. Create the `scf.for` loop for the producer.
  // If the consumer worker count perfectly divides the producer worker count,
  // then we can use a lower bound of 0 and keep the loop bounds static.
  Value lb = perfectlyDivides ? rewriter.create<arith::ConstantIndexOp>(loc, 0)
                              : linearConsumerIdVal;
  Value ub =
      getValueOrCreateConstantIndexOp(rewriter, loc, producerWorkerCount);
  Value step =
      getValueOrCreateConstantIndexOp(rewriter, loc, consumerWorkerCount);
  auto newProducer =
      rewriter.create<scf::ForOp>(loc, lb, ub, step, *sharedDest);
  Block *loopBody = newProducer.getBody();

  // Get the replacement IDs for the producer loop.
  rewriter.setInsertionPointToStart(loopBody);
  Value newFlatProducerId =
      perfectlyDivides
          ? affine::makeComposedAffineApply(
                rewriter, loc, d0 + d1,
                {newProducer.getInductionVar(), linearConsumerIdVal})
          : newProducer.getInductionVar();

  // We require a descending relative mapping, so delinearize in reverse order.
  auto delinearize = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, newFlatProducerId, llvm::to_vector(llvm::reverse(producerRanges)));

  SmallVector<Value> newBlockArgs =
      llvm::map_to_vector(llvm::reverse(delinearize.getResults()),
                          [](OpResult r) -> Value { return r; });
  newBlockArgs.append(newProducer.getRegionIterArgs().begin(),
                      newProducer.getRegionIterArgs().end());

  // Step 4. Inline the region of the producer and replace the terminator.
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

  // Step 5. Replace the extract slice with a `barrier_region` op to indicate
  // synchronization of the shared tensor.
  rewriter.setInsertionPointAfter(newProducer);
  replaceConsumerChain(rewriter, loc, producer.getResult(0),
                       newProducer.getResult(0), consumerChain);

  rewriter.eraseOp(producer);
  return success();
}

//===----------------------------------------------------------------------===//
// MultiMmaOp Lowering
//===----------------------------------------------------------------------===//

namespace {
struct LowerMultiMmaPattern : public OpRewritePattern<IREE::GPU::MultiMmaOp> {
  using OpRewritePattern<IREE::GPU::MultiMmaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    if (mmaOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          mmaOp, "lowering to concrete op requires vector semantics");
    }
    SmallVector<int64_t> bounds;
    mmaOp.getIterationBounds(bounds);
    if (!bounds.empty()) {
      return rewriter.notifyMatchFailure(mmaOp,
                                         "must be a single mma operation");
    }

    auto [lhsVectorType, rhsVectorType, accVectorType] =
        mmaOp.getKind().getABCVectorTypes();

    Value aCast = mmaOp.getLhs();
    Value bCast = mmaOp.getRhs();
    Value cCast = mmaOp.getAcc();
    if (aCast.getType() != lhsVectorType) {
      aCast = rewriter.create<vector::ShapeCastOp>(mmaOp.getLoc(),
                                                   lhsVectorType, aCast);
    }
    if (bCast.getType() != rhsVectorType) {
      bCast = rewriter.create<vector::ShapeCastOp>(mmaOp.getLoc(),
                                                   rhsVectorType, bCast);
    }
    if (cCast.getType() != accVectorType) {
      cCast = rewriter.create<vector::ShapeCastOp>(mmaOp.getLoc(),
                                                   accVectorType, cCast);
    }

    FailureOr<Value> concreteMmaOp = mmaOp.getKind().buildMmaOperation(
        rewriter, mmaOp.getLoc(), cCast.getType(), aCast, bCast, cCast);
    assert(succeeded(concreteMmaOp) && "Failed to create mma op");
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        mmaOp, mmaOp.getAcc().getType(), *concreteMmaOp);
    return success();
  }
};
} // namespace

void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerMultiMmaPattern>(patterns.getContext());
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

// Helper to convert a contraction-like linalg op to an iree_gpu.multi_mma.
FailureOr<IREE::GPU::MultiMmaOp>
convertContractionToMultiMma(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             IREE::GPU::MmaInterfaceAttr mmaKind) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return failure();
  }

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

  SmallVector<int64_t> identityPerm = {0, 1};

  std::optional<SmallVector<int64_t>> lhsPerm = std::nullopt;
  if (lhsInnerPerm != identityPerm) {
    lhsPerm = lhsInnerPerm;
  }
  std::optional<SmallVector<int64_t>> rhsPerm = std::nullopt;
  if (rhsInnerPerm != identityPerm) {
    rhsPerm = rhsInnerPerm;
  }
  std::optional<SmallVector<int64_t>> accPerm = std::nullopt;
  if (accInnerPerm != identityPerm) {
    accPerm = accInnerPerm;
  }

  IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
      getLoweringConfig(linalgOp);

  auto newMmaOp = rewriter.replaceOpWithNewOp<IREE::GPU::MultiMmaOp>(
      linalgOp, inputs[0], inputs[1], inputs[2],
      ArrayRef<AffineMap>{outerLhsMap, outerRhsMap, outerAccMap}, iteratorTypes,
      mmaKind, lhsPerm, rhsPerm, accPerm);
  if (maybeLoweringConfig) {
    setLoweringConfig(newMmaOp, maybeLoweringConfig);
  }
  return newMmaOp;
}

//===----------------------------------------------------------------------===//
// MultiMmaOp Distribution
//===----------------------------------------------------------------------===//

FailureOr<Operation *> distributeMultiMmaOp(RewriterBase &rewriter,
                                            IREE::GPU::MultiMmaOp mmaOp) {
  if (!mmaOp.hasTensorSemantics() || mmaOp.hasThreadSemantics()) {
    return rewriter.notifyMatchFailure(
        mmaOp, "mmaOp must have vector and subgroup for distribution.");
  }

  OpBuilder::InsertionGuard g(rewriter);

  Location loc = mmaOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);

  // Step 1. Create the new scf.forall op with a lane id mapping.
  auto newForallOp = rewriter.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>{zero},
      ArrayRef<OpFoldResult>{
          rewriter.getIndexAttr(mmaOp.getKind().getSubgroupSize())},
      ArrayRef<OpFoldResult>{one}, mmaOp.getAcc(),
      ArrayAttr::get(context, {IREE::GPU::LaneIdAttr::get(context, 0)}));

  rewriter.setInsertionPointToStart(newForallOp.getBody());

  // Step 2. Compute the offsets/sizes/strides for each of the operands.
  auto getOrInferPermutationOfRank =
      [](std::optional<ArrayRef<int64_t>> maybePerm,
         int64_t rank) -> SmallVector<int64_t> {
    if (maybePerm) {
      return SmallVector<int64_t>(*maybePerm);
    }
    return llvm::to_vector(llvm::seq(static_cast<int64_t>(0), rank));
  };
  Value laneId = newForallOp.getInductionVar(0);

  // LHS slice offsets.
  int64_t lhsOuterRank = mmaOp.getLhsOuterRank();
  SmallVector<OpFoldResult> lhsOffsets(lhsOuterRank, zero);
  SmallVector<OpFoldResult> lhsSizes;
  for (int64_t i = 0, e = lhsOuterRank; i < e; ++i) {
    lhsSizes.push_back(tensor::getMixedSize(rewriter, loc, mmaOp.getLhs(), i));
  }
  SmallVector<OpFoldResult> lhsStrides(lhsOuterRank, one);
  SmallVector<int64_t> lhsPermutation = getOrInferPermutationOfRank(
      mmaOp.getLhsPermutation(), mmaOp.getLhsInnerShape().size());
  if (failed(mmaOp.getKind().populateOperandOffsetsSizesStrides(
          rewriter, loc, IREE::GPU::MMAFragment::Lhs, laneId, lhsPermutation,
          lhsOffsets, lhsSizes, lhsStrides))) {
    return mmaOp->emitOpError("failed to populate lhs offsets");
  }
  // Extract the rank-reduced slice of the lhs based on the expected inner
  // vector shape.
  Value lhsSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, mmaOp.getLhs(), lhsOffsets, lhsSizes, lhsStrides);

  // RHS slice offsets.
  int64_t rhsOuterRank = mmaOp.getRhsOuterRank();
  SmallVector<OpFoldResult> rhsOffsets(rhsOuterRank, zero);
  SmallVector<OpFoldResult> rhsSizes;
  for (int64_t i = 0, e = rhsOuterRank; i < e; ++i) {
    rhsSizes.push_back(tensor::getMixedSize(rewriter, loc, mmaOp.getRhs(), i));
  }
  SmallVector<OpFoldResult> rhsStrides(rhsOuterRank, one);
  SmallVector<int64_t> rhsPermutation = getOrInferPermutationOfRank(
      mmaOp.getRhsPermutation(), mmaOp.getRhsInnerShape().size());
  if (failed(mmaOp.getKind().populateOperandOffsetsSizesStrides(
          rewriter, loc, IREE::GPU::MMAFragment::Rhs, laneId, rhsPermutation,
          rhsOffsets, rhsSizes, rhsStrides))) {
    return mmaOp->emitOpError("failed to populate rhs offsets");
  }
  // Extract the rank-reduced slice of the rhs based on the expected inner
  // vector shape.
  Value rhsSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, mmaOp.getRhs(), rhsOffsets, rhsSizes, rhsStrides);

  // Accumulator slice offsets.
  int64_t accOuterRank = mmaOp.getAccOuterRank();
  SmallVector<OpFoldResult> accOffsets(accOuterRank, zero);
  SmallVector<OpFoldResult> accSizes;
  for (int64_t i = 0, e = accOuterRank; i < e; ++i) {
    accSizes.push_back(tensor::getMixedSize(rewriter, loc, mmaOp.getAcc(), i));
  }
  SmallVector<OpFoldResult> accStrides(accOuterRank, one);
  SmallVector<int64_t> accPermutation = getOrInferPermutationOfRank(
      mmaOp.getAccPermutation(), mmaOp.getAccInnerShape().size());
  if (failed(mmaOp.getKind().populateOperandOffsetsSizesStrides(
          rewriter, loc, IREE::GPU::MMAFragment::Acc, laneId, accPermutation,
          accOffsets, accSizes, accStrides))) {
    return mmaOp->emitOpError("failed to populate acc offsets");
  }
  // Extract the rank-reduced slice of the accumulator based on the expected
  // inner vector shape.
  Value accSlice = rewriter.create<tensor::ExtractSliceOp>(
      loc, newForallOp.getRegionIterArgs()[0], accOffsets, accSizes,
      accStrides);

  // Step 3. Create the new multi_mma op.
  auto newMmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
      loc, lhsSlice, rhsSlice, accSlice, mmaOp.getIndexingMaps(),
      mmaOp.getIteratorTypes(), mmaOp.getKind());

  // Step 4. Insert the result of the multi_mma using the same offsets/sizes as
  // the accumulator slice.
  scf::InParallelOp terminator = newForallOp.getTerminator();
  rewriter.setInsertionPointToStart(terminator.getBody());
  rewriter.create<tensor::ParallelInsertSliceOp>(
      loc, newMmaOp.getResult(), newForallOp.getRegionIterArgs()[0], accOffsets,
      accSizes, accStrides);

  rewriter.replaceOp(mmaOp, newForallOp);

  return &*newForallOp;
}

//===----------------------------------------------------------------------===//
// MultiMmaOp Unit Dim Folding
//===----------------------------------------------------------------------===//

namespace {
struct DropMultiMmaUnitDimsPattern
    : public OpRewritePattern<IREE::GPU::MultiMmaOp> {
  using OpRewritePattern<IREE::GPU::MultiMmaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    if (mmaOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          mmaOp, "unimplemented: unit dim dropping for tensor mma ops");
    }
    SmallVector<int64_t> bounds;
    mmaOp.getIterationBounds(bounds);
    if (bounds.empty()) {
      return rewriter.notifyMatchFailure(mmaOp, "no dimensions to fold");
    }

    // TODO: Generalize to allow only some iteration bounds to be unit. This
    // pattern currently only supports the most common case of unrolling to the
    // intrinsic shape.
    if (!llvm::all_of(bounds, [](int64_t b) { return b == 1; })) {
      return rewriter.notifyMatchFailure(mmaOp,
                                         "not all iteration bounds are unit");
    }

    Location loc = mmaOp.getLoc();
    auto dropLeadUnitDims = [&](Value operand, int64_t numDims) -> Value {
      if (numDims == 0) {
        return operand;
      }
      SmallVector<int64_t> droppedDimIndices(numDims, 0);
      return rewriter.create<vector::ExtractOp>(loc, operand,
                                                droppedDimIndices);
    };

    Value newLhs = dropLeadUnitDims(mmaOp.getLhs(), mmaOp.getLhsOuterRank());
    Value newRhs = dropLeadUnitDims(mmaOp.getRhs(), mmaOp.getRhsOuterRank());
    Value newAcc = dropLeadUnitDims(mmaOp.getAcc(), mmaOp.getAccOuterRank());

    AffineMap empty = AffineMap::get(rewriter.getContext());
    auto newMmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
        loc, newLhs, newRhs, newAcc,
        rewriter.getAffineMapArrayAttr({empty, empty, empty}),
        rewriter.getArrayAttr({}), mmaOp.getKind());

    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        mmaOp, mmaOp.getResultType(), newMmaOp);
    return success();
  }
};
} // namespace

void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns) {
  patterns.add<DropMultiMmaUnitDimsPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// MultiMmaOp Unrolling
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

struct UnrollMultiMmaPattern : public OpRewritePattern<GPU::MultiMmaOp> {
  UnrollMultiMmaPattern(MLIRContext *context,
                        const vector::UnrollVectorOptions &options,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<GPU::MultiMmaOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    if (options.filterConstraint && failed(options.filterConstraint(mmaOp))) {
      return rewriter.notifyMatchFailure(mmaOp, "unrolling filter");
    }
    assert(options.nativeShape &&
           "vector unrolling expects the native shape or native shape call "
           "back function to be set");
    std::optional<SmallVector<int64_t, 4>> maybeUnrollShape =
        mmaOp.getShapeForUnroll();
    if (!maybeUnrollShape) {
      return rewriter.notifyMatchFailure(
          mmaOp, "unexpected failure to get unroll shape");
    }

    std::optional<SmallVector<int64_t>> targetShape =
        options.nativeShape(mmaOp);
    if (!targetShape) {
      return rewriter.notifyMatchFailure(mmaOp,
                                         "unspecified native unroll shape");
    }

    auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
    if (!maybeShapeRatio) {
      return rewriter.notifyMatchFailure(
          mmaOp, "operation unroll shape not divisible by target shape");
    }

    // Early exit if unrolling has no effect.
    if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
      return rewriter.notifyMatchFailure(
          mmaOp, "operation already unrolled to native shape");
    }

    auto dstVecType = cast<VectorType>(mmaOp.getResultType());
    SmallVector<int64_t, 4> originalSize = *maybeUnrollShape;

    Location loc = mmaOp.getLoc();
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(mmaOp.getIteratorTypes().size(), mmaOp, options);

    AffineMap lhsPermutationMap = mmaOp.getIndexingMapsArray()[0];
    AffineMap rhsPermutationMap = mmaOp.getIndexingMapsArray()[1];
    AffineMap accPermutationMap = mmaOp.getIndexingMapsArray()[2];

    ArrayRef<int64_t> innerAccShape = mmaOp.getAccInnerShape();

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<Value> slicesOperands(mmaOp.getNumOperands());

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

      // Extract the new lhs operand.
      SmallVector<int64_t> lhsOffets =
          applyPermutationMap(lhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(0, mmaOp.getLhs(), lhsPermutationMap, lhsOffets);

      // Extract the new rhs operand.
      SmallVector<int64_t> rhsOffets =
          applyPermutationMap(rhsPermutationMap, ArrayRef<int64_t>(offsets));
      extractOperand(1, mmaOp.getRhs(), rhsPermutationMap, rhsOffets);

      SmallVector<int64_t> accOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // If a version of the accumulator has already been computed, use it
      // otherwise extract the first version from the original operand.
      auto *accIt = accCache.find(accOffets);
      if (accIt != accCache.end()) {
        slicesOperands[2] = accIt->second;
      } else {
        extractOperand(2, mmaOp.getAcc(), accPermutationMap, accOffets);
      }

      SmallVector<int64_t> dstShape = applyPermutationMap(
          accPermutationMap, ArrayRef<int64_t>(*targetShape));
      dstShape.append(innerAccShape.begin(), innerAccShape.end());
      auto targetType = VectorType::get(dstShape, dstVecType.getElementType());

      // Clone the mma op with the new operands and result type.
      IREE::GPU::MultiMmaOp newOp =
          mlir::clone(rewriter, mmaOp, targetType, slicesOperands);

      SmallVector<int64_t> dstOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // Save the accumulated value until all the loops are unrolled since
      // reduction loop keep updating the accumulator.
      accCache[dstOffets] = newOp.getResult();
    }
    // Assemble back the accumulator into a single vector.
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstVecType, rewriter.getZeroAttr(dstVecType));
    for (const auto &[offsets, partialResult] : accCache) {
      SmallVector<int64_t> dstStrides(offsets.size() + innerAccShape.size(), 1);
      SmallVector<int64_t> fullOffsets(offsets.begin(), offsets.end());
      fullOffsets.append(innerAccShape.size(), 0);
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, partialResult, result, fullOffsets, dstStrides);
    }
    rewriter.replaceOp(mmaOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};
} // namespace

void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options) {
  patterns.add<UnrollMultiMmaPattern>(patterns.getContext(), options);
}

static bool isReductionIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register.
static std::optional<SmallVector<int64_t>>
gpuMultiMmaUnrollOrder(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dimsInLhs;
  for (AffineExpr expr : mmaOp.getIndexingMapsArray()[0].getResults()) {
    dimsInLhs.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>> getMultiMmaUnitShape(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(mmaOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void populateIREEGPUVectorUnrollPatterns(RewritePatternSet &patterns) {
  populateIREEGPUVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getMultiMmaUnitShape)
                    .setUnrollTraversalOrderFn(gpuMultiMmaUnrollOrder));
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

  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, /*upperBound=*/nullptr);
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
        loc, barrierRegionOp.getDest());

    // Step 2. Inline the barrier op region.
    auto terminator = barrierRegionOp.getBody()->getTerminator();
    Value replacement = terminator->getOperand(0);
    rewriter.inlineBlockBefore(barrierRegionOp.getBody(), barrierRegionOp,
                               {writeBarrier.getResult(0)});
    rewriter.setInsertionPointAfterValue(replacement);
    Value barrier;

    // Step 3. Synchronize the result value.
    barrier = rewriter.create<IREE::GPU::ValueBarrierOp>(loc, replacement)
                  .getResult(0);
    rewriter.replaceAllUsesWith(barrierRegionOp.getResult(), barrier);
    rewriter.eraseOp(terminator);
    return success();
  }
};
} // namespace

void populateIREEGPULowerBarrierRegionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerBarrierRegion>(patterns.getContext());
}

//===---------------------------------------------------------------------===//
// MultiMmaOp Vectorization
//===---------------------------------------------------------------------===//

static LogicalResult vectorizeStaticMultiMmaOp(RewriterBase &rewriter,
                                               IREE::GPU::MultiMmaOp mmaOp) {
  if (!mmaOp.hasTensorSemantics()) {
    return failure();
  }
  if (!mmaOp.getLhsType().hasStaticShape() ||
      !mmaOp.getRhsType().hasStaticShape() ||
      !mmaOp.getAccType().hasStaticShape()) {
    return rewriter.notifyMatchFailure(mmaOp,
                                       "non-static shape for vectorization");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mmaOp);

  Location loc = mmaOp.getLoc();

  // Construct the (never used) zero padding value for each operand.
  auto lhsPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(mmaOp.getLhsType().getElementType()));
  auto rhsPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(mmaOp.getRhsType().getElementType()));
  Type resultElementType = mmaOp.getResultType().getElementType();
  auto accPadValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultElementType));

  auto lhs = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getLhs(), mmaOp.getLhsType().getShape(), lhsPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto rhs = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getRhs(), mmaOp.getRhsType().getShape(), rhsPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto acc = vector::createReadOrMaskedRead(
      rewriter, loc, mmaOp.getAcc(), mmaOp.getAccType().getShape(), accPadValue,
      /*useInBoundsInsteadOfMasking=*/true);
  auto newMmaOp = rewriter.create<IREE::GPU::MultiMmaOp>(
      loc, lhs, rhs, acc, mmaOp.getIndexingMaps(), mmaOp.getIteratorTypes(),
      mmaOp.getKind());

  // Create the write back to a tensor.
  int64_t rank = mmaOp.getResultType().getRank();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      mmaOp,
      /*vector=*/newMmaOp,
      /*source=*/mmaOp.getAcc(),
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/SmallVector<bool>(rank, true));
  return success();
}

namespace {
struct VectorizeStaticMultiMmaOpPattern final
    : OpRewritePattern<IREE::GPU::MultiMmaOp> {
  using OpRewritePattern<IREE::GPU::MultiMmaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {
    return vectorizeStaticMultiMmaOp(rewriter, mmaOp);
  }
};
} // namespace

static LogicalResult
vectorizeStaticBarrierRegionResult(RewriterBase &rewriter,
                                   IREE::GPU::BarrierRegionOp barrier) {
  auto tensorResultType =
      dyn_cast<RankedTensorType>(barrier.getResult().getType());
  if (!tensorResultType || !tensorResultType.hasStaticShape()) {
    return failure();
  }

  VectorType newResultType = VectorType::get(tensorResultType.getShape(),
                                             tensorResultType.getElementType());

  auto paddingValue = rewriter.create<arith::ConstantOp>(
      barrier.getLoc(), rewriter.getZeroAttr(newResultType.getElementType()));

  auto newBarrier = rewriter.create<IREE::GPU::BarrierRegionOp>(
      barrier.getLoc(), newResultType, barrier.getDest());

  auto currentTerminator =
      cast<IREE::GPU::YieldOp>(barrier.getBody()->getTerminator());
  rewriter.mergeBlocks(barrier.getBody(), newBarrier.getBody(),
                       newBarrier.getBody()->getArguments());
  rewriter.setInsertionPointToEnd(newBarrier.getBody());

  auto innerRead = vector::createReadOrMaskedRead(
      rewriter, currentTerminator.getLoc(), currentTerminator->getOperand(0),
      newResultType.getShape(), paddingValue,
      /*useInBoundsInsteadOfMasking=*/true);
  rewriter.create<IREE::GPU::YieldOp>(currentTerminator->getLoc(), innerRead);
  rewriter.eraseOp(currentTerminator);

  rewriter.setInsertionPointAfter(newBarrier);

  // Create the write back to a tensor.
  auto empty = rewriter.create<tensor::EmptyOp>(
      barrier.getLoc(), tensorResultType.getShape(),
      tensorResultType.getElementType());
  int64_t rank = tensorResultType.getRank();
  auto zero = rewriter.create<arith::ConstantIndexOp>(barrier.getLoc(), 0);
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      barrier,
      /*vector=*/newBarrier,
      /*source=*/empty,
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/SmallVector<bool>(rank, true));
  return success();
}

namespace {
struct VectorizeStaticBarrierRegionResultPattern
    : public OpRewritePattern<IREE::GPU::BarrierRegionOp> {
  using OpRewritePattern<IREE::GPU::BarrierRegionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::BarrierRegionOp shuffle,
                                PatternRewriter &rewriter) const override {
    return vectorizeStaticBarrierRegionResult(rewriter, shuffle);
  }
};
} // namespace

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeStaticMultiMmaOpPattern>(patterns.getContext());
  patterns.add<VectorizeStaticBarrierRegionResultPattern>(
      patterns.getContext());
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
} // namespace

void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerValueBarrierPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
