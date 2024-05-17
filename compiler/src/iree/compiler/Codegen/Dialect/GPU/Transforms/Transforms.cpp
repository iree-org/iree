// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  auto convertIteratorType = [](utils::IteratorType iteratorType) {
    if (iteratorType == utils::IteratorType::parallel) {
      return IREE::GPU::IteratorType::parallel;
    }
    return IREE::GPU::IteratorType::reduction;
  };

  llvm::SmallDenseSet<int64_t> droppedDims = {innerM, innerN, innerK};
  llvm::SmallDenseMap<int64_t, int64_t> oldDimsToNewDimsMap;
  int64_t currentDim = 0;
  int64_t numDims = lhsMap.getNumDims();
  SmallVector<IREE::GPU::IteratorType> iteratorTypes;
  for (int64_t dim = 0, e = numDims; dim < e; ++dim) {
    if (droppedDims.contains(dim)) {
      continue;
    }
    iteratorTypes.push_back(convertIteratorType(linalgIteratorTypes[dim]));
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

  auto newMmaOp = rewriter.replaceOpWithNewOp<IREE::GPU::MultiMmaOp>(
      linalgOp, inputs[0], inputs[1], inputs[2],
      ArrayRef<AffineMap>{outerLhsMap, outerRhsMap, outerAccMap}, iteratorTypes,
      mmaKind, lhsPerm, rhsPerm, accPerm);
  return newMmaOp;
}

//===----------------------------------------------------------------------===//
// MultiMmaOp Distribution
//===----------------------------------------------------------------------===//

FailureOr<Operation *> distributeMultiMmaOp(RewriterBase &rewriter,
                                            IREE::GPU::MultiMmaOp mmaOp) {
  if (!mmaOp.hasTensorSemantics() || mmaOp.hasThreadSemantics()) {
    return failure();
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
  if (failed(mmaOp.getKind().populateLhsOffsetsSizesStrides(
          rewriter, loc, *newForallOp.getSingleInductionVar(), lhsPermutation,
          lhsOffsets, lhsSizes, lhsStrides))) {
    return failure();
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
  if (failed(mmaOp.getKind().populateRhsOffsetsSizesStrides(
          rewriter, loc, *newForallOp.getSingleInductionVar(), rhsPermutation,
          rhsOffsets, rhsSizes, rhsStrides))) {
    return failure();
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
  if (failed(mmaOp.getKind().populateAccOffsetsSizesStrides(
          rewriter, loc, *newForallOp.getSingleInductionVar(), accPermutation,
          accOffsets, accSizes, accStrides))) {
    return failure();
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
// MultiMmaOp Vectorization
//===----------------------------------------------------------------------===//

LogicalResult vectorizeStaticMultiMmaOp(RewriterBase &rewriter,
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
  SmallVector<OpFoldResult> sizes;
  for (auto size : mmaOp.getResultType().getShape()) {
    sizes.push_back(rewriter.getIndexAttr(size));
  }
  Value dest = rewriter.create<tensor::EmptyOp>(loc, sizes, resultElementType);
  int64_t rank = mmaOp.getResultType().getRank();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
      mmaOp,
      /*vector=*/newMmaOp,
      /*source=*/dest,
      /*indices=*/SmallVector<Value>(rank, zero),
      /*inBounds=*/SmallVector<bool>(rank, true));
  return success();
}

namespace {
struct VectorizeStaticMultiMmaOpPattern
    : public OpRewritePattern<IREE::GPU::MultiMmaOp> {
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

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  return builder.create(loc, op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
}

struct UnrollMultiMmaPattern : public OpRewritePattern<GPU::MultiMmaOp> {
  UnrollMultiMmaPattern(MLIRContext *context,
                        const vector::UnrollVectorOptions &options,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<GPU::MultiMmaOp>(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(GPU::MultiMmaOp mmaOp,
                                PatternRewriter &rewriter) const override {

    if (options.filterConstraint && failed(options.filterConstraint(mmaOp))) {
      return failure();
    }
    assert(options.nativeShape &&
           "vector unrolling expects the native shape or native shape call "
           "back function to be set");
    std::optional<SmallVector<int64_t, 4>> maybeUnrollShape =
        mmaOp.getShapeForUnroll();
    if (!maybeUnrollShape) {
      return failure();
    }

    std::optional<SmallVector<int64_t>> targetShape =
        options.nativeShape(mmaOp);
    if (!targetShape) {
      return failure();
    }

    auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
    if (!maybeShapeRatio) {
      return failure();
    }

    // Early exit if unrolling has no effect.
    if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
      return failure();
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
      Operation *newOp = cloneOpWithOperandsAndTypes(
          rewriter, loc, mmaOp, slicesOperands, targetType);

      SmallVector<int64_t> dstOffets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      // Save the accumulated value untill all the loops are unrolled since
      // reduction loop keep updating the accumulator.
      accCache[dstOffets] = newOp->getResult(0);
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

    FailureOr<Value> concreteMmaOp = mmaOp.getKind().buildMmaOperation(
        rewriter, mmaOp.getLoc(), mmaOp.getResultType(), mmaOp.getLhs(),
        mmaOp.getRhs(), mmaOp.getAcc());
    if (failed(concreteMmaOp)) {
      return rewriter.notifyMatchFailure(mmaOp, "Failed to construct mma op");
    }
    rewriter.replaceOp(mmaOp, *concreteMmaOp);
    return success();
  }
};
} // namespace

void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerMultiMmaPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
