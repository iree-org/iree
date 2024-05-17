// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"

#define DEBUG_TYPE "iree-codegen-gpu-transforms"

namespace mlir::iree_compiler::IREE::GPU {

//===---------------------------------------------------------------------===//
// Forall Fusion
//===---------------------------------------------------------------------===//

static FailureOr<int64_t> getTripCount(scf::ForallOp loop) {
  ArrayRef<int64_t> lbs = loop.getStaticLowerBound();
  ArrayRef<int64_t> ubs = loop.getStaticUpperBound();
  ArrayRef<int64_t> steps = loop.getStaticStep();

  if (ShapedType::isDynamicShape(lbs) || ShapedType::isDynamicShape(ubs) ||
      ShapedType::isDynamicShape(steps)) {
    return failure();
  }

  int64_t tripCount = 1;
  for (auto [lb, ub, step] : llvm::zip_equal(lbs, ubs, steps)) {
    tripCount *= mlir::ceilDiv((ub - lb), step);
  }
  return tripCount;
}

static LogicalResult compareWorkerCountsAndTypes(scf::ForallOp producer,
                                                 scf::ForallOp consumer) {
  FailureOr<int64_t> producerTripCount = getTripCount(producer);
  FailureOr<int64_t> consumerTripCount = getTripCount(consumer);
  if (failed(producerTripCount) || failed(consumerTripCount) ||
      *producerTripCount != *consumerTripCount) {
    return failure();
  }

  auto checkMappingTypes = [&](ArrayAttr array) {
    return llvm::all_of(array.getValue(),
                        llvm::IsaPred<gpu::GPUThreadMappingAttr>) ||
           llvm::all_of(array.getValue(),
                        llvm::IsaPred<gpu::GPUWarpMappingAttr>);
  };

  if (producer.getMappingAttr() != consumer.getMappingAttr() ||
      !checkMappingTypes(producer.getMappingAttr()) ||
      !checkMappingTypes(consumer.getMappingAttr())) {
    return failure();
  }
  return success();
}

static void replaceExtractSlice(RewriterBase &rewriter, Location loc,
                                tensor::ParallelInsertSliceOp parallelInsert,
                                tensor::ExtractSliceOp extractSlice) {
  OpBuilder::InsertionGuard g(rewriter);
  auto shuffleOp = rewriter.create<IREE::GPU::ShuffleTensorOp>(
      loc, extractSlice.getType(), parallelInsert.getSource(),
      parallelInsert.getOffsets(), parallelInsert.getSizes(),
      parallelInsert.getStrides(), parallelInsert.getStaticOffsets(),
      parallelInsert.getStaticSizes(), parallelInsert.getStaticStrides(),
      parallelInsert.getDest());
  Region *region = &shuffleOp.getRegion();
  rewriter.createBlock(region, region->end(),
                       ArrayRef<Type>{parallelInsert.getDestType()},
                       ArrayRef<Location>{loc});
  rewriter.setInsertionPointToStart(shuffleOp.getBody());
  auto terminator =
      rewriter.create<IREE::GPU::YieldOp>(loc, extractSlice.getResult());
  rewriter.moveOpBefore(extractSlice, terminator);
  extractSlice.getSourceMutable().assign(shuffleOp.getBody()->getArgument(0));
  rewriter.replaceAllUsesExcept(extractSlice.getResult(), shuffleOp,
                                terminator);
}

LogicalResult fuseForallIntoSlice(RewriterBase &rewriter,
                                  scf::ForallOp producer,
                                  scf::ForallOp consumer,
                                  tensor::ExtractSliceOp slice) {
  if (producer->getNumResults() != 1) {
    return failure();
  }

  if (failed(compareWorkerCountsAndTypes(producer, consumer))) {
    return failure();
  }

  auto isAll = [](ArrayRef<OpFoldResult> array, int64_t cmp) {
    return llvm::all_of(array, [cmp](OpFoldResult val) {
      return isConstantIntValue(val, cmp);
    });
  };

  if (!isAll(producer.getMixedStep(), 1) ||
      !isAll(producer.getMixedLowerBound(), 0) ||
      !isAll(consumer.getMixedStep(), 1) ||
      !isAll(consumer.getMixedLowerBound(), 0)) {
    return failure();
  }

  rewriter.setInsertionPoint(slice);

  // Step 1. Compute the producer IDs in terms of the consumer IDs.

  MLIRContext *context = rewriter.getContext();
  Location loc = producer.getLoc();

  AffineExpr d0, d1, d2;
  bindDims(context, d0, d1, d2);
  AffineExpr mulAdd = d0 * d1 + d2;
  OpFoldResult linearId = rewriter.getIndexAttr(0);
  for (auto [inductionVar, workerCount] :
       llvm::zip_equal(getAsOpFoldResult(consumer.getInductionVars()),
                       consumer.getMixedUpperBound())) {
    linearId = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulAdd, {linearId, workerCount, inductionVar});
  }

  Value linearThreadIdVal =
      getValueOrCreateConstantIndexOp(rewriter, loc, linearId);
  SmallVector<Value> ranges;
  for (auto workerCount : producer.getStaticUpperBound()) {
    ranges.push_back(rewriter.create<arith::ConstantIndexOp>(loc, workerCount));
  }
  ValueRange newIds = rewriter
                          .create<affine::AffineDelinearizeIndexOp>(
                              loc, linearThreadIdVal, ranges)
                          .getResults();

  // Step 2. Inline the region of the producer.
  SmallVector<Value> bbArgReplacements(newIds);
  bbArgReplacements.append(producer.getOutputs().begin(),
                           producer.getOutputs().end());

  scf::InParallelOp terminator = producer.getTerminator();
  rewriter.inlineBlockBefore(producer.getBody(), slice, bbArgReplacements);

  rewriter.setInsertionPointAfter(terminator);
  auto parallelInsert =
      cast<tensor::ParallelInsertSliceOp>(*terminator.getYieldingOps().begin());

  replaceExtractSlice(rewriter, loc, parallelInsert, slice);

  rewriter.eraseOp(parallelInsert);
  rewriter.eraseOp(terminator);
  rewriter.eraseOp(producer);
  return success();
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

void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeStaticMultiMmaOpPattern>(patterns.getContext());
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
    rewriter.replaceOp(barrier, barrier.getInput());
    return success();
  }
};
} // namespace

void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerValueBarrierPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
