// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h"

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.cpp.inc"
// clang-format on

namespace mlir::iree_compiler {

namespace {

/// Extracts a boolean option from a DictionaryAttr.
static bool getBoolOption(DictionaryAttr options, StringRef name,
                          bool defaultValue = false) {
  if (!options) {
    return defaultValue;
  }
  if (auto attr = options.getAs<BoolAttr>(name)) {
    return attr.getValue();
  }
  return defaultValue;
}

struct GatherOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<GatherOpVectorizationModel,
                                                    IREE::LinalgExt::GatherOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto gatherOp = cast<IREE::LinalgExt::GatherOp>(op);
    // TODO: Support indexDepth > 1 by splitting the innermost dim of
    // `indices` into `indexDepth` vectors so that each independent index can
    // be passed to the transfer_gather op.
    if (gatherOp.getIndexDepth() != 1) {
      return false;
    }
    return true;
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto gatherOp = cast<IREE::LinalgExt::GatherOp>(op);
    int64_t batchRank = gatherOp.getBatchRank();
    Location loc = gatherOp.getLoc();
    RewriterBase::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(gatherOp);

    ShapedType indicesTy = gatherOp.getIndicesType();
    ShapedType gatherTy = gatherOp.getOutputType();
    ShapedType sourceTy = gatherOp.getSourceType();

    if (vectorSizes.empty()) {
      vectorSizes = gatherTy.getShape();
    }

    auto gatherVectorTy =
        VectorType::get(vectorSizes, gatherTy.getElementType());
    // Rank-reduced to remove the innermost unit dim.
    auto indicesVecTy =
        VectorType::get(vectorSizes.take_front(gatherOp.getBatchRank()),
                        rewriter.getIndexType());

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    VectorType indicesMaskType = indicesVecTy.clone(rewriter.getI1Type());
    SmallVector<OpFoldResult> gatherDims =
        tensor::getMixedSizes(rewriter, loc, gatherOp.getOutput());
    Value indicesMask = vector::CreateMaskOp::create(
        rewriter, loc, indicesMaskType,
        ArrayRef(gatherDims).take_front(gatherOp.getBatchRank()));
    auto indicesVecRead = vector::TransferReadOp::create(
        rewriter, loc, indicesVecTy.clone(indicesTy.getElementType()),
        gatherOp.getIndices(), SmallVector<Value>(indicesTy.getRank(), zero),
        std::nullopt);
    rewriter.modifyOpInPlace(indicesVecRead, [&] {
      indicesVecRead.getMaskMutable().assign(indicesMask);
    });
    Value indicesVec = indicesVecRead.getResult();
    indicesVec =
        arith::IndexCastOp::create(rewriter, loc, indicesVecTy, indicesVec);

    SmallVector<Value> baseOffsets(sourceTy.getRank(), zero);
    Value padding =
        ub::PoisonOp::create(rewriter, loc, gatherTy.getElementType());

    // Build indexing_maps for the transfer_gather.
    // Source map: (vector_dims)[s0] -> (s0, d_batch+1, ..., d_N)
    // First source dim is gathered (s0), rest are contiguous.
    MLIRContext *ctx = rewriter.getContext();
    int64_t vectorRank = vectorSizes.size();
    int64_t sourceRank = sourceTy.getRank();
    SmallVector<AffineExpr> sourceMapExprs;
    sourceMapExprs.push_back(getAffineSymbolExpr(0, ctx)); // gathered dim 0
    for (int64_t i = 1; i < sourceRank; ++i) {
      // Map remaining source dims to corresponding vector dims.
      // The batch dims come first, so source dim i maps to vector dim
      // (i - 1 + batchRank).
      sourceMapExprs.push_back(getAffineDimExpr(i - 1 + batchRank, ctx));
    }
    AffineMap sourceMap =
        AffineMap::get(vectorRank, /*symbolCount=*/1, sourceMapExprs, ctx);

    // Index vec map: (vector_dims)[s0] -> (d0, ..., d_{batchRank-1})
    SmallVector<AffineExpr> indexVecMapExprs;
    for (int64_t i = 0; i < batchRank; ++i) {
      indexVecMapExprs.push_back(getAffineDimExpr(i, ctx));
    }
    AffineMap indexVecMap =
        AffineMap::get(vectorRank, /*symbolCount=*/1, indexVecMapExprs, ctx);

    SmallVector<AffineMap> indexingMaps = {sourceMap, indexVecMap};

    VectorType gatherMaskType = gatherVectorTy.clone(rewriter.getI1Type());
    Value gatherMask =
        vector::CreateMaskOp::create(rewriter, loc, gatherMaskType, gatherDims);

    // Add a mask indexing map (identity) to the indexing_maps.
    // TODO: symbolCount is hardcoded to 1 because indexDepth != 1 bails out
    // above. All indexing maps must share the same symbol count (= number of
    // index vecs). Update this when indexDepth > 1 is supported.
    AffineMap maskMap =
        AffineMap::getMultiDimIdentityMap(vectorRank, rewriter.getContext());
    maskMap = AffineMap::get(vectorRank, /*symbolCount=*/1,
                             maskMap.getResults(), rewriter.getContext());
    indexingMaps.push_back(maskMap);

    auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
        rewriter, loc, gatherVectorTy, gatherOp.getSource(), baseOffsets,
        ValueRange{indicesVec}, rewriter.getAffineMapArrayAttr(indexingMaps),
        padding, /*mask=*/gatherMask);
    SmallVector<Value> writeIndices(gatherTy.getRank(), zero);
    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc, transferGatherOp.getResult(), gatherOp.getOutput(),
        writeIndices);
    rewriter.modifyOpInPlace(
        writeOp, [&] { writeOp.getMaskMutable().assign(gatherMask); });

    return SmallVector<Value>{writeOp.getResult()};
  }
};

struct ArgCompareOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<
          ArgCompareOpVectorizationModel, IREE::LinalgExt::ArgCompareOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto argCompareOp = cast<IREE::LinalgExt::ArgCompareOp>(op);
    // Only static shapes are supported. Dynamic shapes would require masking.
    // Check input shape (includes reduction dimension) to catch dynamic
    // reduction dimensions that wouldn't appear in the static output shape.
    auto inputValTy = cast<ShapedType>(argCompareOp.getInputValue().getType());
    if (!inputValTy.hasStaticShape()) {
      return false;
    }
    ShapedType outValTy = argCompareOp.getOutputValueType();
    SmallVector<int64_t> sizes(vectorSizes);
    if (sizes.empty()) {
      sizes = llvm::to_vector(outValTy.getShape());
    }
    if (sizes != llvm::to_vector(outValTy.getShape())) {
      return false;
    }
    return true;
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto argCompareOp = cast<IREE::LinalgExt::ArgCompareOp>(op);
    Location loc = argCompareOp.getLoc();
    RewriterBase::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(argCompareOp);

    auto inputValTy = cast<ShapedType>(argCompareOp.getInputValue().getType());
    // Only static shapes are supported. Dynamic shapes would require masking.
    if (!inputValTy.hasStaticShape()) {
      return failure();
    }

    ShapedType outValTy = argCompareOp.getOutputValueType();
    ShapedType outIdxTy = argCompareOp.getOutputIndexType();

    if (vectorSizes.empty()) {
      vectorSizes = outValTy.getShape();
    }

    // Ensure full tiles - partial tiles would require masking support.
    if (vectorSizes != outValTy.getShape()) {
      return failure();
    }

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Vectorize the DPS input operands (input_value and optional input_index).
    // We explicitly access these by name to avoid accidentally including
    // index_base.
    SmallVector<Value> inputVecs;
    auto vectorizeInput = [&](Value input) {
      auto inputTy = cast<ShapedType>(input.getType());
      SmallVector<Value> readIndices(inputTy.getRank(), zero);
      auto inputVecTy =
          VectorType::get(inputTy.getShape(), inputTy.getElementType());
      // TODO(Bangtian): Add masking/padding support for partial tiles.
      // Currently passes std::nullopt, assuming vector size matches tensor
      // shape.
      auto readOp = vector::TransferReadOp::create(
          rewriter, loc, inputVecTy, input, readIndices, std::nullopt);
      inputVecs.push_back(readOp);
    };

    vectorizeInput(argCompareOp.getInputValue());
    if (Value inputIndex = argCompareOp.getInputIndex()) {
      vectorizeInput(inputIndex);
    }

    SmallVector<Value> initVecs;
    for (Value init : argCompareOp.getDpsInits()) {
      auto initTy = cast<ShapedType>(init.getType());
      SmallVector<Value> readIndices(vectorSizes.size(), zero);
      auto initVecTy = VectorType::get(vectorSizes, initTy.getElementType());
      auto readOp = vector::TransferReadOp::create(
          rewriter, loc, initVecTy, init, readIndices, std::nullopt);
      initVecs.push_back(readOp);
    }

    auto outValVecTy = VectorType::get(vectorSizes, outValTy.getElementType());
    auto outIdxVecTy = VectorType::get(vectorSizes, outIdxTy.getElementType());

    Region &srcRegion = argCompareOp.getRegion();

    // Create the vector arg_compare operation using the builder that takes
    // individual operands (this properly handles AttrSizedOperandSegments).
    Value inputIndex = (inputVecs.size() > 1) ? inputVecs[1] : Value();
    Value indexBase = argCompareOp.getIndexBase();

    auto vectorArgCompareOp = IREE::VectorExt::ArgCompareOp::create(
        rewriter, loc, TypeRange{outValVecTy, outIdxVecTy},
        /*input_value=*/inputVecs[0],
        /*input_index=*/inputIndex,
        /*init_value=*/initVecs[0],
        /*init_index=*/initVecs[1],
        /*index_base=*/indexBase,
        /*dimension=*/argCompareOp.getDimension());

    Block *srcBlock = &srcRegion.front();
    Region &dstRegion = vectorArgCompareOp.getRegion();
    rewriter.modifyOpInPlace(vectorArgCompareOp,
                             [&]() { dstRegion.getBlocks().clear(); });
    SmallVector<Location> argLocs(srcBlock->getNumArguments(), loc);
    Block *dstBlock = rewriter.createBlock(
        &dstRegion, dstRegion.end(), srcBlock->getArgumentTypes(), argLocs);

    // Clone operations from source block to destination block using rewriter.
    IRMapping mapper;
    for (auto [srcArg, dstArg] :
         llvm::zip_equal(srcBlock->getArguments(), dstBlock->getArguments())) {
      mapper.map(srcArg, dstArg);
    }

    rewriter.setInsertionPointToStart(dstBlock);
    for (Operation &bodyOp : srcBlock->getOperations()) {
      auto yieldOp = dyn_cast<IREE::LinalgExt::YieldOp>(bodyOp);
      if (!yieldOp) {
        rewriter.clone(bodyOp, mapper);
        continue;
      }
      // Replace LinalgExt::YieldOp with VectorExt::YieldOp.
      SmallVector<Value> mappedOperands;
      for (Value operand : yieldOp.getOperands()) {
        mappedOperands.push_back(mapper.lookup(operand));
      }
      IREE::VectorExt::YieldOp::create(rewriter, yieldOp.getLoc(),
                                       mappedOperands);
    }

    // Set insertion point to after the vectorArgCompareOp for subsequent
    // operations.
    rewriter.setInsertionPointAfter(vectorArgCompareOp);

    SmallVector<Value> results;
    for (auto [result, output] : llvm::zip_equal(
             vectorArgCompareOp.getResults(), argCompareOp.getDpsInits())) {
      SmallVector<Value> writeIndices(vectorSizes.size(), zero);
      auto writeOp = vector::TransferWriteOp::create(rewriter, loc, result,
                                                     output, writeIndices);
      results.push_back(writeOp.getResult());
    }

    return results;
  }
};

struct ToLayoutOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<
          ToLayoutOpVectorizationModel, IREE::VectorExt::ToLayoutOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto toLayoutOp = cast<IREE::VectorExt::ToLayoutOp>(op);
    return toLayoutOp.hasTensorSemantics();
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto toLayoutOp = cast<IREE::VectorExt::ToLayoutOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(toLayoutOp);
    Location loc = toLayoutOp.getLoc();
    ShapedType inputTy = toLayoutOp.getType();
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto identityMap = rewriter.getMultiDimIdentityMap(inputTy.getRank());
    SmallVector<int64_t> readShape =
        toLayoutOp.getLayout().getUndistributedShape();
    Value mask = nullptr;
    bool needsMask = !toLayoutOp.getType().hasStaticShape() ||
                     (readShape != inputTy.getShape());
    if (needsMask) {
      SmallVector<OpFoldResult> mixedSourceDims =
          tensor::getMixedSizes(rewriter, loc, toLayoutOp.getInput());
      auto maskType = VectorType::get(readShape, rewriter.getI1Type());
      mask = vector::CreateMaskOp::create(rewriter, loc, maskType,
                                          mixedSourceDims);
    }
    VectorType vectorType =
        VectorType::get(readShape, inputTy.getElementType());
    auto inBounds = rewriter.getBoolArrayAttr(
        SmallVector<bool>(vectorType.getRank(), true));
    auto padValue =
        ub::PoisonOp::create(rewriter, loc, inputTy.getElementType());
    auto readOp = vector::TransferReadOp::create(
        rewriter, loc,
        /*type=*/vectorType,
        /*source=*/toLayoutOp.getInput(),
        /*indices=*/ValueRange{SmallVector<Value>(readShape.size(), zero)},
        /*permutation_map=*/identityMap,
        /*padding=*/padValue,
        /*mask=*/mask,
        /*in_bounds=*/inBounds);
    // Create the toLayout operation but with vector types instead.
    auto newLayoutOp = IREE::VectorExt::ToLayoutOp::create(
        rewriter, loc, readOp, toLayoutOp.getLayout(),
        toLayoutOp.getSharedMemoryConversion());
    // Create the write back to a tensor.
    ShapedType tensorTy = toLayoutOp.getType();
    auto resType =
        RankedTensorType::get(tensorTy.getShape(), tensorTy.getElementType());
    int64_t rank = tensorTy.getShape().size();
    auto writeInBounds =
        rewriter.getBoolArrayAttr(SmallVector<bool>(rank, true));
    auto writeIdentityMap = rewriter.getMultiDimIdentityMap(tensorTy.getRank());
    auto writeOp = vector::TransferWriteOp::create(
        rewriter, loc,
        /*result=*/resType,
        /*vector=*/newLayoutOp,
        /*source=*/toLayoutOp.getInput(),
        /*indices=*/ValueRange{SmallVector<Value>(rank, zero)},
        /*permutation_map=*/writeIdentityMap,
        /*mask=*/mask,
        /*inBounds=*/writeInBounds);
    return SmallVector<Value>{writeOp.getResult()};
  }
};

struct MapStoreOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<
          MapStoreOpVectorizationModel, IREE::LinalgExt::MapStoreOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto mapStoreOp = cast<IREE::LinalgExt::MapStoreOp>(op);
    if (mapStoreOp.isVectorized()) {
      return false;
    }
    ShapedType inputType = mapStoreOp.getInputType();
    if (!inputType.hasStaticShape()) {
      return false;
    }
    const int64_t innerSize = inputType.getShape()[inputType.getRank() - 1];
    const int64_t bitWidth = inputType.getElementTypeBitWidth();
    if ((innerSize * bitWidth % 8) != 0) {
      return false;
    }
    // In case of a sub-byte bitwidth, we check that there is a contiguous copy
    // on the inner dimension that is a multiple of a byte. Note that the mask
    // shouldn't depend on the inner index for this.
    if (bitWidth < 8) {
      // First check that the mask is not the forward slice of the inner index.
      Value innermostInputIdx =
          mapStoreOp.getInputIndex(mapStoreOp.getInputRank() - 1);
      SetVector<Operation *> slice;
      getForwardSlice(innermostInputIdx, &slice);
      Operation *maskOp = mapStoreOp.getMask().getDefiningOp();
      if (maskOp && slice.contains(maskOp)) {
        return false;
      }
      // Next check that the inner index of the yield is a unit function of
      // the inner input index.
      Value innermostOutputIdx =
          mapStoreOp.getOutputIndex(mapStoreOp.getOutputRank() - 1);
      if (!isUnitFunctionOf(innermostOutputIdx, innermostInputIdx)) {
        return false;
      }
    }
    return true;
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto mapStoreOp = cast<IREE::LinalgExt::MapStoreOp>(op);
    Location loc = mapStoreOp.getLoc();
    rewriter.setInsertionPoint(mapStoreOp);
    ShapedType inputType = mapStoreOp.getInputType();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> zeros(inputType.getRank(), zero);
    auto inputVectorType =
        VectorType::get(inputType.getShape(), inputType.getElementType());
    Value inputVector = vector::TransferReadOp::create(
        rewriter, loc, inputVectorType, mapStoreOp.getInput(),
        /*indices=*/zeros,
        /*padding=*/std::nullopt);
    auto vectorizedMapStoreOp =
        clone(rewriter, mapStoreOp, mapStoreOp.getResultTypes(),
              {inputVector, mapStoreOp.getOutput()});
    return SmallVector<Value>(vectorizedMapStoreOp->getResults());
  }
};

//===----------------------------------------------------------------------===//
// Im2col Vectorization
//===----------------------------------------------------------------------===//

/// Compute the padding mask and adjusted read indices for im2col
/// vectorization. Thin wrapper around computeIm2colPaddingBounds: gets
/// clamped read offsets and valid size, then converts to a vector mask.
static Value computeIm2colPaddingMask(
    OpBuilder &b, Location loc, IREE::LinalgExt::Im2colOp im2colOp,
    const IREE::LinalgExt::Im2colSourceIndices &srcIndices,
    ArrayRef<OpFoldResult> inputSizes, ArrayRef<OpFoldResult> padLow,
    int64_t vecWidth, ArrayRef<Value> outputIVs,
    ArrayRef<OpFoldResult> outputOffsets, std::optional<int64_t> vecOutputDim,
    SmallVector<Value> &readIndices) {
  int64_t inputRank = im2colOp.getInputRank();
  auto vecI1Type = VectorType::get({vecWidth}, b.getI1Type());

  OpFoldResult innerTileSize = b.getIndexAttr(vecWidth);
  IREE::LinalgExt::Im2colPaddingBounds bounds =
      IREE::LinalgExt::computeIm2colPaddingBounds(
          b, loc, im2colOp, srcIndices, inputSizes, padLow, innerTileSize,
          outputIVs, outputOffsets, vecOutputDim);

  for (int64_t d = 0; d < inputRank; ++d) {
    readIndices.push_back(
        getValueOrCreateConstantIndexOp(b, loc, bounds.readOffsets[d]));
  }

  return vector::CreateMaskOp::create(b, loc, vecI1Type, bounds.validSize);
}

struct Im2colOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<
          Im2colOpVectorizationModel, IREE::LinalgExt::Im2colOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto im2colOp = cast<IREE::LinalgExt::Im2colOp>(op);
    ShapedType outputType = im2colOp.getOutputType();
    if (!outputType.hasStaticShape()) {
      return false;
    }

    // Conservative unroll limit check: at worst, no dim is vectorized and
    // all output elements are unrolled.
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t totalElements = 1;
    for (int64_t s : outputShape)
      totalElements *= s;
    static constexpr int64_t kMaxVectorizeUnrollIters = 1024;
    if (totalElements > kMaxVectorizeUnrollIters) {
      return false;
    }

    return true;
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto im2colOp = cast<IREE::LinalgExt::Im2colOp>(op);
    RewriterBase::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(im2colOp);
    ShapedType outputType = im2colOp.getOutputType();
    Location loc = im2colOp.getLoc();
    bool hasPadding = im2colOp.hasPadding();

    SmallVector<OpFoldResult> mixedOffsets = im2colOp.getMixedOffsets();
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());

    int64_t inputRank = im2colOp.getInputRank();
    SmallVector<OpFoldResult> padLow(inputRank, rewriter.getIndexAttr(0));
    if (hasPadding) {
      SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
      if (!inputPadLow.empty()) {
        padLow = inputPadLow;
      }
    }

    SmallVector<Range> iterationDomain(im2colOp.getIterationDomain(rewriter));
    std::optional<int64_t> vecDim =
        IREE::LinalgExt::chooseDimToVectorize(rewriter, loc, im2colOp,
                                              iterationDomain, inputSizes,
                                              mixedOffsets);

    int64_t outputRank = im2colOp.getOutputRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = outputType.getElementType();

    int64_t vecWidth = vecDim ? outputShape[*vecDim] : 1;

    // When padding is present and we're vectorizing a dimension, reject if
    // the corresponding input dimension has non-zero low-side padding.
    // Low-side padding would require shifting all read indices which our
    // current mask logic doesn't handle correctly.
    if (hasPadding && vecWidth > 1) {
      int64_t vecInputDim = inputRank - 1;
      if (!isConstantIntValue(padLow[vecInputDim], 0)) {
        return failure();
      }
    }

    auto vecType = VectorType::get({vecWidth}, elemType);

    Value padValue;
    if (hasPadding) {
      padValue = im2colOp.getPadValue();
    } else {
      padValue = arith::ConstantOp::create(rewriter, loc, elemType,
                                           rewriter.getZeroAttr(elemType));
    }

    int64_t writeDim = vecDim ? *vecDim : (outputRank - 1);
    AffineMap writePermMap = AffineMap::get(
        outputRank, 0, rewriter.getAffineDimExpr(writeDim),
        rewriter.getContext());

    SmallVector<int64_t> loopDims;
    SmallVector<int64_t> loopBounds;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (vecDim && d == *vecDim)
        continue;
      loopDims.push_back(d);
      loopBounds.push_back(outputShape[d]);
    }

    int64_t totalIters = 1;
    for (int64_t bound : loopBounds)
      totalIters *= bound;

    Value result = im2colOp.getOutput();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    for (int64_t iter = 0; iter < totalIters; ++iter) {
      SmallVector<Value> ivs(outputRank, zeroIdx);
      int64_t remaining = iter;
      for (int64_t i = loopDims.size() - 1; i >= 0; --i) {
        int64_t idx = remaining % loopBounds[i];
        remaining /= loopBounds[i];
        ivs[loopDims[i]] =
            arith::ConstantIndexOp::create(rewriter, loc, idx);
      }

      IREE::LinalgExt::Im2colSourceIndices srcIndices =
          IREE::LinalgExt::computeIm2colSourceIndices(
              rewriter, loc, im2colOp, ivs, rewriter.getIndexAttr(vecWidth));

      SmallVector<Value> readIndices;
      Value mask;
      if (hasPadding) {
        ArrayRef<Value> outputIVs;
        ArrayRef<OpFoldResult> outputOffsets;
        std::optional<int64_t> vecOutDim;
        if (IREE::LinalgExt::hasOutputPadding(im2colOp)) {
          outputIVs = ivs;
          outputOffsets = mixedOffsets;
          if (vecDim) {
            vecOutDim = *vecDim;
          }
        }
        mask = computeIm2colPaddingMask(
            rewriter, loc, im2colOp, srcIndices, inputSizes, padLow,
            vecWidth, outputIVs, outputOffsets, vecOutDim, readIndices);
      } else {
        for (OpFoldResult ofr : srcIndices.sliceOffsets) {
          readIndices.push_back(
              getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
        }
      }

      Value readVec;
      if (mask) {
        AffineMap readPermMap = AffineMap::getMinorIdentityMap(
            inputRank, 1, rewriter.getContext());
        auto inBoundsAttr = rewriter.getBoolArrayAttr({true});
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            readPermMap, padValue, mask, inBoundsAttr);
      } else {
        readVec = vector::TransferReadOp::create(
            rewriter, loc, vecType, im2colOp.getInput(), readIndices,
            padValue);
      }

      SmallVector<Value> writeIndices(ivs);
      if (vecDim) {
        writeIndices[*vecDim] = zeroIdx;
      }
      result = vector::TransferWriteOp::create(rewriter, loc, readVec, result,
                                               writeIndices, writePermMap)
                   .getResult();
    }

    return SmallVector<Value>{result};
  }
};
} // namespace

void registerVectorizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    IREE::LinalgExt::GatherOp::attachInterface<GatherOpVectorizationModel>(
        *ctx);
    IREE::LinalgExt::ArgCompareOp::attachInterface<
        ArgCompareOpVectorizationModel>(*ctx);
    IREE::LinalgExt::MapStoreOp::attachInterface<MapStoreOpVectorizationModel>(
        *ctx);
    IREE::LinalgExt::Im2colOp::attachInterface<Im2colOpVectorizationModel>(
        *ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::VectorExt::IREEVectorExtDialect *dialect) {
    IREE::VectorExt::ToLayoutOp::attachInterface<ToLayoutOpVectorizationModel>(
        *ctx);
  });
}

} // namespace mlir::iree_compiler
