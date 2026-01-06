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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-decompose-map-scatter"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEMAPSCATTERPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

/// Fold a subview op into the output of a map_scatter op. The offsets of the
/// subview op are folded into the body of the map_scatter, and each offset is
/// added to the corresponding yielded index. For this pattern to apply, the
/// map_scatter op must be vectorized and bufferized, and the subview must be
/// non-rank reducing with unit strides. The subview's result also should not
/// be collapsible, because the decomposition will work for collapsable memrefs,
/// and there is no need to fold the subview.
struct FoldSubViewIntoMapScatter final : OpRewritePattern<MapScatterOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(MapScatterOp mapScatterOp,
                                PatternRewriter &rewriter) const override {
    if (!mapScatterOp.isVectorized()) {
      return rewriter.notifyMatchFailure(mapScatterOp,
                                         "map_scatter op is not vectorized");
    }
    if (!mapScatterOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          mapScatterOp, "map_scatter op has non-buffer semantics");
    }
    auto subViewOp =
        mapScatterOp.getOutput().getDefiningOp<memref::SubViewOp>();
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
    auto mapScatterBodyYield = cast<IREE::LinalgExt::YieldOp>(
        mapScatterOp.getTransformationRegion().front().getTerminator());
    rewriter.setInsertionPoint(mapScatterBodyYield);
    SmallVector<Value> newYieldedIndices;
    for (OpFoldResult subViewOffset : subViewOp.getMixedOffsets()) {
      newYieldedIndices.push_back(getValueOrCreateConstantIndexOp(
          rewriter, subViewOp.getLoc(), subViewOffset));
    }
    SmallVector<Value> yieldedIndices = mapScatterBodyYield.getOperands();
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
    rewriter.modifyOpInPlace(mapScatterBodyYield, [&]() {
      mapScatterBodyYield.getOperandsMutable().assign(newYieldedValues);
    });
    Value subViewSource = subViewOp.getSource();
    rewriter.modifyOpInPlace(subViewOp, [&]() {
      mapScatterOp.getOutputMutable().assign(subViewSource);
    });
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
  Value flatOutput;
};

/// Vectorize the index computation and mask evaluation for a `map_scatter` op.
/// This creates a `linalg.generic` op that computes linearized output indices
/// and mask values for all elements of the input vector, then vectorizes it
/// to produce vector operations. Returns the computed index vector, mask
/// vector, and flattened output buffer. If `disregardInnerDimension` is true,
/// the innermost dimension is treated as size 1 for vectorization. This is used
/// by `decomposeToLoadStore` to decompose the `map_scatter` op into a sequence
/// of `vector.extract` and `vector.store` operations.
static FailureOr<VectorizationResult>
performIndexAndMaskVectorization(MapScatterOp mapScatterOp,
                                 RewriterBase &rewriter,
                                 bool disregardInnerDimension = false) {
  Location loc = mapScatterOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapScatterOp);
  Value flatOutput;
  SmallVector<Value> strides;
  SmallVector<OpFoldResult> outputSizes =
      getDims(rewriter, loc, mapScatterOp.getOutput());
  if (mapScatterOp.hasPureBufferSemantics()) {
    flatOutput = createFlatOutputBuffer(rewriter, loc, mapScatterOp.getOutput(),
                                        outputSizes, strides);
  } else {
    // For tensor outputs, create a flat output buffer as an empty tensor.
    auto outputType = cast<TensorType>(mapScatterOp.getOutputType());
    SmallVector<ReassociationIndices> reassociations;
    reassociations.push_back(
        llvm::to_vector(llvm::seq<int64_t>(outputType.getRank())));
    flatOutput = tensor::CollapseShapeOp::create(
        rewriter, loc, mapScatterOp.getOutput(), reassociations);
  }
  auto inputType = cast<VectorType>(mapScatterOp.getInputType());
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
    mapScatterOp.inlineMapScatterBody(b, nestedLoc, indices, inlineBodyBuilder);
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
    return rewriter.notifyMatchFailure(mapScatterOp,
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
  return VectorizationResult{indexVector, maskVector, flatOutput};
}

/// Decompose the `map_scatter` into a sequence of `vector.extract` and
/// `vector.store` operations.
static LogicalResult decomposeToLoadStore(MapScatterOp mapScatterOp,
                                          RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapScatterOp);
  FailureOr<VectorizationResult> vectorizationResult =
      performIndexAndMaskVectorization(mapScatterOp, rewriter,
                                       /*disregardInnerDimension=*/true);
  if (failed(vectorizationResult)) {
    return failure();
  }
  Value indexVector = vectorizationResult->indexVector;
  Value maskVector = vectorizationResult->maskVector;
  Value flatOutputBuffer = vectorizationResult->flatOutput;

  // Flatten all the index and mask vectors, since the scatter op lowering
  // expects 1D vectors.
  auto inputType = cast<VectorType>(mapScatterOp.getInputType());
  const int64_t flatIndexSize =
      llvm::product_of(inputType.getShape().drop_back());
  const int64_t flatVectorSize = llvm::product_of(inputType.getShape());
  Location loc = mapScatterOp.getLoc();
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
                                                  mapScatterOp.getInput());
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
  rewriter.eraseOp(mapScatterOp);
  return success();
}

/// Decompose a map_scatter op into a single `vector.scatter` operation.
///
/// The function uses `performIndexAndMaskVectorization` to compute vectorized
/// indices and masks, then flattens all vectors to 1D and replaces the
/// `map_scatter` with a single `vector.scatter` operation.
static LogicalResult decomposeToScatter(MapScatterOp mapScatterOp,
                                        RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(mapScatterOp);
  FailureOr<VectorizationResult> vectorizationResult =
      performIndexAndMaskVectorization(mapScatterOp, rewriter,
                                       /*disregardInnerDimension=*/false);
  if (failed(vectorizationResult)) {
    return failure();
  }
  Value indexVector = vectorizationResult->indexVector;
  Value maskVector = vectorizationResult->maskVector;
  Value flatOutput = vectorizationResult->flatOutput;

  // Flatten all the vectors, since the scatter op lowering expects 1D vectors.
  auto inputType = cast<VectorType>(mapScatterOp.getInputType());
  const int64_t flatVectorSize = llvm::product_of(inputType.getShape());
  Location loc = mapScatterOp.getLoc();
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
                                                  mapScatterOp.getInput());

  SmallVector<Value> offsets = {
      arith::ConstantIndexOp::create(rewriter, loc, 0)};
  SmallVector<Value> operands = {flatOutput, offsets[0], indexVector,
                                 maskVector, inputVector};

  if (mapScatterOp.hasPureBufferSemantics()) {
    rewriter.replaceOpWithNewOp<vector::ScatterOp>(
        mapScatterOp, /*resultTypes=*/TypeRange{}, operands);
    return success();
  }

  // For tensor outputs, expand the result back to the original shape.
  auto scatterOp =
      vector::ScatterOp::create(rewriter, loc, flatOutput.getType(), operands);
  SmallVector<ReassociationIndices> reassociations;
  reassociations.push_back(llvm::to_vector(
      llvm::seq<int64_t>(mapScatterOp.getOutputType().getRank())));
  SmallVector<OpFoldResult> outputSizes =
      tensor::getMixedSizes(rewriter, loc, mapScatterOp.getOutput());
  auto expandOp = tensor::ExpandShapeOp::create(
      rewriter, loc, mapScatterOp.getOutputType(), scatterOp.getResult(),
      reassociations, outputSizes);
  rewriter.replaceOp(mapScatterOp, expandOp.getResult());
  return success();
}

/// Decompose an `iree_linalg_ext.map_scatter` op with vector input.
/// This is the main dispatch function that analyzes the `map_scatter`
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
static LogicalResult decomposeMapScatter(MapScatterOp mapScatterOp,
                                         RewriterBase &rewriter) {
  Value innermostInputIdx =
      mapScatterOp.getInputIndex(mapScatterOp.getInputRank() - 1);
  Value innermostOutputIdx =
      mapScatterOp.getOutputIndex(mapScatterOp.getOutputRank() - 1);
  SetVector<Operation *> slice;
  getForwardSlice(innermostInputIdx, &slice);
  Operation *maskOp = mapScatterOp.getMask().getDefiningOp();
  const bool isMaskForwardSlice = maskOp && slice.contains(maskOp);
  const bool isUnitFunctionOfInnermostInputIdx =
      isUnitFunctionOf(innermostOutputIdx, innermostInputIdx);
  if (!isMaskForwardSlice && isUnitFunctionOfInnermostInputIdx &&
      mapScatterOp.hasPureBufferSemantics()) {
    return decomposeToLoadStore(mapScatterOp, rewriter);
  }
  // In case of a sub-byte map_scatter that hasn't been decomposed into a
  // sequence of extract/store ops above, there is a potential non-contiguous
  // copy on the inner dimension that is not a multiple of a byte size through a
  // stride or mask and the map_scatter can't be vectorized, so fail.
  const int64_t bitWidth = mapScatterOp.getInputType().getElementTypeBitWidth();
  if (bitWidth < 8) {
    if (!isUnitFunctionOfInnermostInputIdx) {
      return mapScatterOp.emitOpError() << "with an access on a sub-byte type "
                                           "that is not a multiple of the byte "
                                           "size can't be vectorized";
    }
    return mapScatterOp.emitOpError()
           << "map_scatter on sub-byte type with potentially non "
              "byte aligned transformation";
  }
  return decomposeToScatter(mapScatterOp, rewriter);
}

namespace {
struct DecomposeMapScatterPass final
    : impl::DecomposeMapScatterPassBase<DecomposeMapScatterPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldSubViewIntoMapScatter>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    if (testPreprocessingPatterns) {
      return;
    }

    // Decomposition is only supported for map_scatter ops that are vectorized.
    SmallVector<MapScatterOp> candidates;
    funcOp->walk([&](MapScatterOp op) {
      if (isa<VectorType>(op.getInputType())) {
        candidates.push_back(op);
      }
    });
    IRRewriter rewriter(context);
    for (auto mapScatterOp : candidates) {
      if (failed(decomposeMapScatter(mapScatterOp, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::LinalgExt
