// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-decompose-map-store"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEMAPSTOREPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

/// Fold a subview op into the output of a map_store op. The offsets of the
/// subview op are folded into the body of the map_store, and each offset is
/// added to the corresponding yielded index. For this pattern to apply, the
/// map_store op must be vectorized and bufferized, and the subview must be
/// non-rank reducing with unit strides. The subview's result also should not
/// be collapsible, because the decomposition will work for collapsable memrefs,
/// and there is no need to fold the subview.
struct FoldSubViewIntoMapStore final : OpRewritePattern<MapStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(MapStoreOp mapStoreOp,
                                PatternRewriter &rewriter) const override {
    if (!mapStoreOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapStoreOp,
                                         "map_store op is not vectorized");
    }
    if (!mapStoreOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          mapStoreOp, "map_store op has non-buffer semantics");
    }
    auto subViewOp = mapStoreOp.getOutput().getDefiningOp<memref::SubViewOp>();
    if (!subViewOp) {
      return failure();
    }
    MemRefType subViewType = subViewOp.getType();
    bool isRankReducing =
        subViewOp.getSourceType().getRank() != subViewType.getRank();
    std::optional<llvm::SmallDenseSet<unsigned int>> rankReductionMask =
        computeRankReductionMask(subViewOp.getStaticSizes(),
                                 subViewType.getShape());
    if (isRankReducing && !rankReductionMask.has_value()) {
      return rewriter.notifyMatchFailure(
          subViewOp, "could not compute rank reduction mask");
    }
    SmallVector<OpFoldResult> strides = subViewOp.getMixedStrides();
    if (!areAllConstantIntValue(strides, 1)) {
      return rewriter.notifyMatchFailure(subViewOp,
                                         "subview op has non-unit strides");
    }
    SmallVector<ReassociationIndices> reassociations;
    reassociations.push_back(
        llvm::to_vector(llvm::seq<int64_t>(subViewType.getRank())));
    if (subViewType.getStridesAndOffset().first.back() == 1 &&
        memref::CollapseShapeOp::isGuaranteedCollapsible(subViewType,
                                                         reassociations)) {
      return rewriter.notifyMatchFailure(
          subViewOp, "subview op is non-strided and collapsible");
    }
    auto mapStoreBodyYield = cast<IREE::LinalgExt::YieldOp>(
        mapStoreOp.getTransformationRegion().front().getTerminator());
    rewriter.setInsertionPoint(mapStoreBodyYield);
    SmallVector<Value> newYieldedIndices;
    for (OpFoldResult subViewOffset : subViewOp.getMixedOffsets()) {
      newYieldedIndices.push_back(getValueOrCreateConstantIndexOp(
          rewriter, subViewOp.getLoc(), subViewOffset));
    }
    SmallVector<Value> yieldedIndices = mapStoreBodyYield.getOperands();
    Value yieldedMask = yieldedIndices.pop_back_val();
    size_t rankReducedIdx = 0;
    for (auto [idx, yieldedIdx] : llvm::enumerate(newYieldedIndices)) {
      if (isRankReducing && rankReductionMask->contains(idx)) {
        continue;
      }
      yieldedIdx =
          arith::AddIOp::create(rewriter, subViewOp.getLoc(), yieldedIdx,
                                yieldedIndices[rankReducedIdx++]);
    }
    SmallVector<Value> newYieldedValues(newYieldedIndices);
    newYieldedValues.push_back(yieldedMask);
    rewriter.modifyOpInPlace(mapStoreBodyYield, [&]() {
      mapStoreBodyYield.getOperandsMutable().assign(newYieldedValues);
    });
    Value subViewSource = subViewOp.getSource();
    rewriter.modifyOpInPlace(subViewOp, [&]() {
      mapStoreOp.getOutputMutable().assign(subViewSource);
    });
    return success();
  }
};

/// Simplify linearize→delinearize pairs where dimension products match.
///
/// When a delinearize directly consumes a linearize, we can group linearize
/// dimensions and match them to delinearize dimensions when their products
/// are equal. This breaks down the original operations into smaller chunks that
/// will avoid long multiply-add and divide-remainder chains when these
/// linearize→delinearize are further lowered.
///
/// This optimization currently handles one-to-one matches (a single linearize
/// dimension matches a single delinearize dimension) and many-to-one matches
/// (multiple linearize dimensions grouped together match a single delinearize
/// dimension).
///
/// Example:
///   %lin = affine.linearize_index [%a, %b, %c, %d, %e] by (%dyn0, 64, 4, 8, 8)
///   %delin:3 = affine.delinearize_index %lin into (%dyn1, 256, 64)
///
/// If %dyn0 == %dyn1, 64*4 == 256, and 8*8 == 64, then we can simplify to:
///   %delin#0 = %a  (direct passthrough)
///   %delin#1 = affine.linearize_index [%b, %c] by (64, 4)
///   %delin#2 = affine.linearize_index [%d, %e] by (8, 8)
struct SimplifyLinearizeDelinearizePairs final
    : OpRewritePattern<affine::AffineDelinearizeIndexOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(affine::AffineDelinearizeIndexOp delinearizeOp,
                                PatternRewriter &rewriter) const override {
    // Find the linearize op that produces the input to this delinearize.
    auto linearizeOp = delinearizeOp.getLinearIndex()
                           .getDefiningOp<affine::AffineLinearizeIndexOp>();
    if (!linearizeOp) {
      return rewriter.notifyMatchFailure(
          delinearizeOp, "delinearize op does not consume a linearize op");
    }
    // We only handle disjoint linearizations.
    if (!linearizeOp.getDisjoint()) {
      return rewriter.notifyMatchFailure(delinearizeOp,
                                         "linearize op is not disjoint");
    }
    SmallVector<OpFoldResult> linearizeBases = linearizeOp.getMixedBasis();
    SmallVector<OpFoldResult> delinearizeBases = delinearizeOp.getMixedBasis();
    ValueRange linearizeInputs = linearizeOp.getMultiIndex();

    Location loc = delinearizeOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Structure to store the inputs and bases for new linearize ops.
    struct LinearizeInfo {
      SmallVector<Value> inputs;
      SmallVector<OpFoldResult> bases;
    };
    SmallVector<LinearizeInfo> newLinearizeInfos;

    // For each delinearize output dimension, try to match it with a product of
    // one or more linearize dimensions. We build up the product incrementally
    // and check for equality using value bounds analysis, which works for both
    // static and dynamic dimensions.
    // TODO(#23032): Consider separating bounds checking from transformation
    // patterns into a shared utility for better organization of index analysis.
    size_t linIdx = 0;
    for (size_t delinIdx = 0;
         delinIdx < delinearizeBases.size() && linIdx < linearizeBases.size();
         ++delinIdx) {
      LinearizeInfo newLinearizeInfo;
      // Track operands for the AffineMap-based product expression.
      SmallVector<Value> productOperands;
      // Accumulate dimensions from the linearize op until we find a match.
      while (linIdx < linearizeBases.size()) {
        newLinearizeInfo.inputs.push_back(linearizeInputs[linIdx]);
        newLinearizeInfo.bases.push_back(linearizeBases[linIdx]);

        // Build up the product of basis dimension using affine expressions.
        AffineExpr productExpr = getAffineConstantExpr(1, ctx);
        productOperands.clear();
        for (OpFoldResult basis : newLinearizeInfo.bases) {
          if (auto attr = dyn_cast<Attribute>(basis)) {
            // Handle the static basis as a constant expression.
            int64_t val = cast<IntegerAttr>(attr).getInt();
            productExpr = productExpr * getAffineConstantExpr(val, ctx);
          } else {
            // Handle the dynamic basis as a symbol expression.
            Value val = cast<Value>(basis);
            productOperands.push_back(val);
            productExpr = productExpr *
                          getAffineSymbolExpr(productOperands.size() - 1, ctx);
          }
        }
        ++linIdx;
        // Create Variable from the product expression.
        AffineMap productMap =
            AffineMap::get(0, productOperands.size(), productExpr, ctx);
        ValueBoundsConstraintSet::Variable productVar(productMap,
                                                      productOperands);
        ValueBoundsConstraintSet::Variable delinearizeVar(
            delinearizeBases[delinIdx]);

        FailureOr<bool> areEqual =
            ValueBoundsConstraintSet::areEqual(productVar, delinearizeVar);
        if (succeeded(areEqual) && *areEqual) {
          newLinearizeInfos.push_back(newLinearizeInfo);
          break;
        }
      }
    }

    if (newLinearizeInfos.size() != delinearizeOp.getNumResults()) {
      return rewriter.notifyMatchFailure(
          delinearizeOp, "could not match all delinearize outputs");
    }

    SmallVector<Value> newResults;
    for (const auto &info : newLinearizeInfos) {
      if (info.inputs.size() == 1) {
        // If there is a one-to-one match between linearize and delinearize
        // dimensions, just pass through the linearize input.
        newResults.push_back(info.inputs[0]);
      } else {
        // Otherwise, for a many-to-one match, create a new linearize op.
        Value newLinearized = affine::AffineLinearizeIndexOp::create(
            rewriter, loc, info.inputs, info.bases,
            /*disjoint=*/true);
        newResults.push_back(newLinearized);
      }
    }
    rewriter.replaceOp(delinearizeOp, newResults);
    return success();
  }
};

/// Flatten the output buffer, and populate `strides` with the strides of the
/// flattened buffer if needed. If the buffer is collapsible, then the strides
/// will remain empty. Otherwise, the strides from the original buffer will be
/// added to the `strides` list.
static Value createFlatOutputBuffer(RewriterBase &rewriter, Location loc,
                                    Value outputBuffer,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &strides) {
  auto outputBufferType = cast<MemRefType>(outputBuffer.getType());
  SmallVector<ReassociationIndices> reassociations;
  reassociations.push_back(
      llvm::to_vector(llvm::seq<int64_t>(outputBufferType.getRank())));
  if (memref::CollapseShapeOp::isGuaranteedCollapsible(outputBufferType,
                                                       reassociations)) {
    return memref::CollapseShapeOp::create(rewriter, loc, outputBuffer,
                                           reassociations);
  }
  auto stridedMetadataOp =
      memref::ExtractStridedMetadataOp::create(rewriter, loc, outputBuffer);
  strides.append(stridedMetadataOp.getStrides().begin(),
                 stridedMetadataOp.getStrides().end());
  Value offset = stridedMetadataOp.getOffset();
  OpFoldResult collapsedSize =
      IREE::LinalgExt::computeProductUsingAffine(rewriter, loc, sizes);
  SmallVector<OpFoldResult> collapsedShape = {collapsedSize};
  SmallVector<OpFoldResult> collapsedStrides = {rewriter.getIndexAttr(1)};
  return memref::ReinterpretCastOp::create(rewriter, loc, outputBuffer, offset,
                                           collapsedShape, collapsedStrides);
}

/// Result of performIndexAndMaskVectorization containing the vectorized
/// index computation, mask, and the flattened output buffer.
struct VectorizationResult {
  /// The vectorized linear indices for scatter destinations.
  Value indexVector;
  /// The vectorized mask values for conditional stores.
  Value maskVector;
  /// The flattened 1D output buffer.
  Value flatOutputBuffer;
};

/// Vectorize the index computation and mask evaluation for a `map_store` op.
/// This creates a `linalg.generic` op that computes linearized output indices
/// and mask values for all elements of the input vector, then vectorizes it
/// to produce vector operations. Returns the computed index vector, mask
/// vector, and flattened output buffer. If `disregardInnerDimension` is true,
/// the innermost dimension is treated as size 1 for vectorization. This is used
/// by `decomposeToLoadStore` to decompose the `map_store` op into a sequence
/// of `vector.extract` and `vector.store` operations.
static FailureOr<VectorizationResult>
performIndexAndMaskVectorization(MapStoreOp mapStoreOp, RewriterBase &rewriter,
                                 bool disregardInnerDimension = false) {
  Location loc = mapStoreOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapStoreOp);
  SmallVector<OpFoldResult> outputSizes =
      memref::getMixedSizes(rewriter, loc, mapStoreOp.getOutput());
  SmallVector<Value> strides;
  Value flatOutputBuffer = createFlatOutputBuffer(
      rewriter, loc, mapStoreOp.getOutput(), outputSizes, strides);
  auto inputType = cast<VectorType>(mapStoreOp.getInputType());
  auto bodyBuilder = [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
    auto inlineBodyBuilder = [&](OpBuilder inlineBuilder, Location inlineLoc,
                                 ArrayRef<Value> yieldedValues) {
      SmallVector<Value> outputIndices(yieldedValues);
      Value mask = outputIndices.pop_back_val();
      Value linearIdx;
      // If strides are empty, this means that the memref layout was contiguous,
      // so we can simply linearize the indices based on the shape. Otherwise,
      // use the strides to compute the linear index.
      if (strides.empty()) {
        linearIdx = affine::AffineLinearizeIndexOp::create(
            inlineBuilder, inlineLoc, outputIndices, outputSizes,
            /*disjoint=*/true);
      } else {
        linearIdx = arith::ConstantIndexOp::create(inlineBuilder, inlineLoc, 0);
        for (auto [outputIdx, stride] :
             llvm::zip_equal(outputIndices, strides)) {
          Value stridedOutputIdx = arith::MulIOp::create(
              inlineBuilder, inlineLoc, outputIdx, stride);
          linearIdx = arith::AddIOp::create(inlineBuilder, inlineLoc, linearIdx,
                                            stridedOutputIdx);
        }
      }
      linalg::YieldOp::create(inlineBuilder, inlineLoc,
                              ValueRange{linearIdx, mask});
    };
    SmallVector<Value> indices = llvm::map_to_vector(
        llvm::seq<int64_t>(inputType.getRank()), [&](int64_t dim) -> Value {
          return linalg::IndexOp::create(b, nestedLoc, b.getIndexType(), dim);
        });
    mapStoreOp.inlineMapStoreBody(b, nestedLoc, indices, inlineBodyBuilder);
  };
  SmallVector<int64_t> shape(inputType.getShape());
  if (disregardInnerDimension) {
    shape[shape.size() - 1] = 1;
  }
  auto idxInit =
      tensor::EmptyOp::create(rewriter, loc, shape, rewriter.getIndexType());
  auto maskInit =
      tensor::EmptyOp::create(rewriter, loc, shape, rewriter.getIntegerType(1));
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iterTypes(inputType.getRank(),
                                             utils::IteratorType::parallel);
  SmallVector<Value> outs = {idxInit.getResult(), maskInit.getResult()};
  auto genericOp =
      linalg::GenericOp::create(rewriter, loc, TypeRange(outs), ValueRange(),
                                outs, maps, iterTypes, bodyBuilder);

  // Lower linearize and delinearize ops before vectorizing, because the
  // vectorizer can't handle them.
  SmallVector<affine::AffineLinearizeIndexOp> linearizeOps(
      genericOp.getBody()->getOps<affine::AffineLinearizeIndexOp>());
  for (auto linearizeOp : linearizeOps) {
    rewriter.setInsertionPoint(linearizeOp);
    if (failed(affine::lowerAffineLinearizeIndexOp(rewriter, linearizeOp))) {
      return rewriter.notifyMatchFailure(
          linearizeOp, "failed to lower affine.linearize_index op");
    }
  }
  SmallVector<affine::AffineDelinearizeIndexOp> delinearizeOps(
      genericOp.getBody()->getOps<affine::AffineDelinearizeIndexOp>());
  for (auto delinearizeOp : delinearizeOps) {
    rewriter.setInsertionPoint(delinearizeOp);
    if (failed(
            affine::lowerAffineDelinearizeIndexOp(rewriter, delinearizeOp))) {
      return rewriter.notifyMatchFailure(
          delinearizeOp, "failed to lower affine.delinearize_index op");
    }
  }

  FailureOr<linalg::VectorizationResult> result =
      linalg::vectorize(rewriter, genericOp);
  if (failed(result)) {
    return rewriter.notifyMatchFailure(mapStoreOp,
                                       "failed to generate index vector");
  }

  auto indexWriteOp =
      result->replacements[0].getDefiningOp<vector::TransferWriteOp>();
  auto maskWriteOp =
      result->replacements[1].getDefiningOp<vector::TransferWriteOp>();
  if (!indexWriteOp) {
    return failure();
  }
  Value indexVector = indexWriteOp.getVector();
  Value maskVector = maskWriteOp.getVector();
  // Erase unused tensor ops after vectorizing the linalg.generic.
  rewriter.eraseOp(indexWriteOp);
  rewriter.eraseOp(maskWriteOp);
  rewriter.eraseOp(genericOp);
  return VectorizationResult{indexVector, maskVector, flatOutputBuffer};
}

/// Decompose the `map_store` into a sequence of `vector.extract` and
/// `vector.store` operations.
static LogicalResult decomposeToLoadStore(MapStoreOp mapStoreOp,
                                          RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapStoreOp);
  FailureOr<VectorizationResult> vectorizationResult =
      performIndexAndMaskVectorization(mapStoreOp, rewriter,
                                       /*disregardInnerDimension=*/true);
  if (failed(vectorizationResult)) {
    return failure();
  }
  Value indexVector = vectorizationResult->indexVector;
  Value maskVector = vectorizationResult->maskVector;
  Value flatOutputBuffer = vectorizationResult->flatOutputBuffer;

  // Flatten all the index and mask vectors, since the scatter op lowering
  // expects 1D vectors.
  auto inputType = cast<VectorType>(mapStoreOp.getInputType());
  const int64_t flatIndexSize =
      llvm::product_of(inputType.getShape().drop_back());
  const int64_t flatVectorSize = llvm::product_of(inputType.getShape());
  Location loc = mapStoreOp.getLoc();
  auto flatIndexType =
      VectorType::get({flatIndexSize}, rewriter.getIndexType());
  indexVector =
      vector::ShapeCastOp::create(rewriter, loc, flatIndexType, indexVector);
  auto flatMaskType =
      VectorType::get({flatIndexSize}, rewriter.getIntegerType(1));
  maskVector =
      vector::ShapeCastOp::create(rewriter, loc, flatMaskType, maskVector);
  auto flatInputType =
      VectorType::get({flatIndexSize, flatVectorSize / flatIndexSize},
                      inputType.getElementType());
  Value inputVector = vector::ShapeCastOp::create(rewriter, loc, flatInputType,
                                                  mapStoreOp.getInput());
  for (int64_t i = 0; i < flatIndexSize; ++i) {
    Value index = arith::ConstantIndexOp::create(rewriter, loc, i);
    auto extractMask =
        vector::ExtractOp::create(rewriter, loc, maskVector, index);
    auto extractIndex =
        vector::ExtractOp::create(rewriter, loc, indexVector, index);
    auto extractValue =
        vector::ExtractOp::create(rewriter, loc, inputVector, index);

    // Only store if mask is true.
    auto ifOp = scf::IfOp::create(rewriter, loc, extractMask.getResult(),
                                  /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    vector::StoreOp::create(rewriter, loc, extractValue.getResult(),
                            flatOutputBuffer, {extractIndex});
  }
  rewriter.eraseOp(mapStoreOp);
  return success();
}

/// Decompose a map_store op into a single `vector.scatter` operation.
///
/// The function uses `performIndexAndMaskVectorization` to compute vectorized
/// indices and masks, then flattens all vectors to 1D and replaces the
/// `map_store` with a single `vector.scatter` operation.
static LogicalResult decomposeToScatter(MapStoreOp mapStoreOp,
                                        RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapStoreOp);
  FailureOr<VectorizationResult> vectorizationResult =
      performIndexAndMaskVectorization(mapStoreOp, rewriter,
                                       /*disregardInnerDimension=*/false);
  if (failed(vectorizationResult)) {
    return failure();
  }
  Value indexVector = vectorizationResult->indexVector;
  Value maskVector = vectorizationResult->maskVector;
  Value flatOutputBuffer = vectorizationResult->flatOutputBuffer;

  // Flatten all the vectors, since the scatter op lowering expects 1D vectors.
  auto inputType = cast<VectorType>(mapStoreOp.getInputType());
  const int64_t flatVectorSize = llvm::product_of(inputType.getShape());
  Location loc = mapStoreOp.getLoc();
  auto flatIndexType =
      VectorType::get({flatVectorSize}, rewriter.getIndexType());
  indexVector =
      vector::ShapeCastOp::create(rewriter, loc, flatIndexType, indexVector);
  auto flatMaskType =
      VectorType::get({flatVectorSize}, rewriter.getIntegerType(1));
  maskVector =
      vector::ShapeCastOp::create(rewriter, loc, flatMaskType, maskVector);
  auto flatInputType =
      VectorType::get({flatVectorSize}, inputType.getElementType());
  Value inputVector = vector::ShapeCastOp::create(rewriter, loc, flatInputType,
                                                  mapStoreOp.getInput());

  SmallVector<Value> offsets = {
      arith::ConstantIndexOp::create(rewriter, loc, 0)};
  SmallVector<Value> operands = {flatOutputBuffer, offsets[0], indexVector,
                                 maskVector, inputVector};
  rewriter.replaceOpWithNewOp<vector::ScatterOp>(
      mapStoreOp, /*resultTypes=*/TypeRange{}, operands);
  return success();
}

/// Decompose an `iree_linalg_ext.map_store` op with vector input and memref
/// output. This is the main dispatch function that analyzes the `map_store`
/// operation and chooses the most appropriate decomposition strategy.
///
/// Decomposition strategies (in order of preference):
/// 1. `decomposeToLoadStore`: Used when the innermost dimension mapping is unit
///    (identity) function from input to output and the mask doesn't depend on
///    the innermost input index. This vectorizes the index and mask
///    computations and generates a sequence of `vector.extract`
///    and `vector.store` operations.
/// 2. `decomposeToScatter`: The fallback. Used for regular scatter patterns
///    with types that are >= 8 bits. This vectorizes the index and mask
///    computations and generates a single `vector.scatter` operation.
static LogicalResult decomposeMapStore(MapStoreOp mapStoreOp,
                                       RewriterBase &rewriter) {
  Value innermostInputIdx =
      mapStoreOp.getInputIndex(mapStoreOp.getInputRank() - 1);
  Value innermostOutputIdx =
      mapStoreOp.getOutputIndex(mapStoreOp.getOutputRank() - 1);
  SetVector<Operation *> slice;
  getForwardSlice(innermostInputIdx, &slice);
  Operation *maskOp = mapStoreOp.getMask().getDefiningOp();
  const bool isMaskForwardSlice = maskOp && slice.contains(maskOp);
  const bool isUnitFunctionOfInnermostInputIdx =
      isUnitFunctionOf(innermostOutputIdx, innermostInputIdx);
  if (!isMaskForwardSlice && isUnitFunctionOfInnermostInputIdx) {
    return decomposeToLoadStore(mapStoreOp, rewriter);
  }
  // In case of a sub-byte map_store that hasn't been decomposed into a
  // sequence of extract/store ops above, there is a potential non-contiguous
  // copy on the inner dimension that is not a multiple of a byte size through a
  // stride or mask and the map_store can't be vectorized, so fail.
  const int64_t bitWidth = mapStoreOp.getInputType().getElementTypeBitWidth();
  if (bitWidth < 8) {
    if (!isUnitFunctionOfInnermostInputIdx) {
      return mapStoreOp.emitOpError() << "with an access on a sub-byte type "
                                         "that is not a multiple of the byte "
                                         "size can't be vectorized";
    }
    return mapStoreOp.emitOpError()
           << "map_store on sub-byte type with potentially non "
              "byte aligned transformation";
  }
  return decomposeToScatter(mapStoreOp, rewriter);
}

namespace {
struct DecomposeMapStorePass final
    : impl::DecomposeMapStorePassBase<DecomposeMapStorePass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldSubViewIntoMapStore, SimplifyLinearizeDelinearizePairs>(
        context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    if (testPreprocessingPatterns) {
      return;
    }

    // Decomposition is only supported for map_store ops that are both
    // vectorized and bufferized. Bufferization is a requirement because
    // vector.scatter only takes memref destinations.
    // TODO(#21135): Allow tensor outputs when vector.scatter supports tensor
    // destinations.
    SmallVector<MapStoreOp> candidates;
    funcOp->walk([&](MapStoreOp op) {
      if (isa<VectorType>(op.getInputType()) && op.hasPureBufferSemantics()) {
        candidates.push_back(op);
      }
    });
    IRRewriter rewriter(context);
    for (auto mapStoreOp : candidates) {
      if (failed(decomposeMapStore(mapStoreOp, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
