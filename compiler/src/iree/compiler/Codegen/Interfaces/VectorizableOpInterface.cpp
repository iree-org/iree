// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Utils/Indexing.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineMap.h"
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
    : VectorizableOpInterface::ExternalModel<GatherOpVectorizationModel,
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
    : VectorizableOpInterface::ExternalModel<ArgCompareOpVectorizationModel,
                                             IREE::LinalgExt::ArgCompareOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto argCompareOp = cast<IREE::LinalgExt::ArgCompareOp>(op);
    // Only static shapes are supported. Dynamic shapes would require masking.
    // Check input shape (includes reduction dimension) to catch dynamic
    // reduction dimensions that wouldn't appear in the static output shape.
    auto inputValTy = cast<ShapedType>(argCompareOp.getInputValue().getType());
    return inputValTy.hasStaticShape();
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
      vectorSizes = inputValTy.getShape();
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

    ArrayRef<int64_t> initShape = outValTy.getShape();
    SmallVector<Value> initVecs;
    for (Value init : argCompareOp.getDpsInits()) {
      auto initTy = cast<ShapedType>(init.getType());
      SmallVector<Value> readIndices(initShape.size(), zero);
      auto initVecTy = VectorType::get(initShape, initTy.getElementType());
      auto readOp = vector::TransferReadOp::create(
          rewriter, loc, initVecTy, init, readIndices, std::nullopt);
      initVecs.push_back(readOp);
    }

    auto outValVecTy = VectorType::get(initShape, outValTy.getElementType());
    auto outIdxVecTy = VectorType::get(initShape, outIdxTy.getElementType());

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
      SmallVector<Value> writeIndices(initShape.size(), zero);
      auto writeOp = vector::TransferWriteOp::create(rewriter, loc, result,
                                                     output, writeIndices);
      results.push_back(writeOp.getResult());
    }

    return results;
  }
};

struct ToLayoutOpVectorizationModel
    : VectorizableOpInterface::ExternalModel<ToLayoutOpVectorizationModel,
                                             IREE::VectorExt::ToLayoutOp> {

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
    : VectorizableOpInterface::ExternalModel<MapStoreOpVectorizationModel,
                                             IREE::LinalgExt::MapStoreOp> {

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

/// Builds a new linalg.generic from a subset of operations in `fullOp`'s body.
///
/// `tensorMap` maps each value inside the linalg body (block arguments and
/// intermediate results from earlier partials) to the (tensor, indexing map)
/// pair that backs it. It is read to wire up inputs and updated with new
/// entries for any results produced by the partial op.
static linalg::GenericOp
buildPartialGenericOp(RewriterBase &rewriter, linalg::GenericOp fullOp,
                      ArrayRef<int64_t> vectorSizes,
                      SmallVectorImpl<Operation *> &partial,
                      DenseMap<Value, std::pair<Value, AffineMap>> &tensorMap) {

  // Find all values used in partial that are defined inside the block.
  SetVector<Value> newInputs;
  SetVector<Value> newOutputs;
  for (Operation *op : partial) {
    for (Value operand : op->getOperands()) {
      if (operand.getParentBlock() != fullOp.getBody()) {
        continue;
      }

      if (tensorMap.contains(operand)) {
        newInputs.insert(operand);
      }
    }

    // If a user of the operation is not in partial, it needs to be a result.
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!llvm::is_contained(partial, user)) {
          newOutputs.insert(result);
        }
      }
    }
  }

  SmallVector<Value> ins, outs;
  SmallVector<AffineMap> indexingMaps;
  AffineMap ident = rewriter.getMultiDimIdentityMap(fullOp.getNumLoops());

  for (Value val : newInputs) {
    auto [in, map] = tensorMap[val];
    ins.push_back(in);
    indexingMaps.push_back(map);
  }

  for (Value val : newOutputs) {
    Value out = tensor::EmptyOp::create(rewriter, fullOp.getLoc(), vectorSizes,
                                        getElementTypeOrSelf(val));
    outs.push_back(out);
    indexingMaps.push_back(ident);
  }

  // If the last operation is a yield, add the out operands.
  bool hasYield = !partial.empty() && isa<linalg::YieldOp>(partial.back());
  if (hasYield) {
    for (OpOperand &operand : fullOp.getDpsInitsMutable()) {
      outs.push_back(operand.get());
      indexingMaps.push_back(fullOp.getMatchingIndexingMap(&operand));
    }
  }

  auto newOp = linalg::GenericOp::create(
      rewriter, fullOp.getLoc(), TypeRange(outs), ins, outs, indexingMaps,
      fullOp.getIteratorTypesArray(),
      [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
        IRMapping localMap;
        for (auto [oldVal, newVal] : llvm::zip_equal(
                 newInputs, blockArgs.take_front(newInputs.size()))) {
          localMap.map(oldVal, newVal);
        }

        if (!hasYield) {
          for (auto [oldVal, newVal] : llvm::zip_equal(
                   newOutputs, blockArgs.take_back(newOutputs.size()))) {
            localMap.map(oldVal, newVal);
          }
        }

        // Clone partial into this region.
        for (Operation *op : partial) {
          b.clone(*op, localMap);
        }

        if (!hasYield) {
          SmallVector<Value> yieldValues = llvm::map_to_vector(
              newOutputs, [&](Value val) { return localMap.lookup(val); });
          linalg::YieldOp::create(b, loc, yieldValues);
        }
      });

  // Add an entry in tensorMap for each value in newOutputs.
  for (auto [index, val] : llvm::enumerate(newOutputs)) {
    Value tensor = newOp.getResult(index);
    AffineMap map = indexingMaps[newInputs.size() + index];
    tensorMap[val] = {tensor, map};
  }

  return newOp;
}

static FailureOr<SmallVector<Value>> vectorizeGatherLikeGenericToTransferGather(
    RewriterBase &rewriter, linalg::GenericOp linalgOp,
    ArrayRef<int64_t> vectorSizes, ArrayRef<bool> scalableVecDims,
    bool vectorizeNDExtract) {

  // Since upstream vectorization does not support hooks to vectorize individual
  // operations inside a linalg.generic, we take an alternate approach here,
  // by splitting the generic into 3 operations, anchored around the first
  // tensor.extract operation:
  //
  // 1. pre-extract-generic
  // 2. extract-as-transfer-gather
  // 3. post-extract-generic

  // Find the first tensor.extract operation and use it as a cut-off point for
  // gather vectorization.
  SmallVector<Operation *> preExtract;
  tensor::ExtractOp extractOp;
  SmallVector<Operation *> postExtract;

  for (Operation &op : linalgOp.getBody()->getOperations()) {
    if (extractOp) {
      // Already found extract, add to postExtract.
      postExtract.push_back(&op);
      continue;
    }
    if (auto candidate = dyn_cast<tensor::ExtractOp>(op)) {
      extractOp = candidate;
      continue;
    }
    preExtract.push_back(&op);
  }

  // If no extract op was found, call generic vectorization.
  if (!extractOp) {
    FailureOr<linalg::VectorizationResult> result = linalg::vectorize(
        rewriter, linalgOp, vectorSizes, scalableVecDims, vectorizeNDExtract);
    if (failed(result)) {
      return failure();
    }
    return result->replacements;
  }

  Location loc = linalgOp->getLoc();
  SmallVector<int64_t> canonicalVectorSizes(vectorSizes);
  SmallVector<bool> canonicalScalableDims(scalableVecDims);

  // If vector sizes are not provided, assume static vector sizes and use loop
  // ranges.
  if (vectorSizes.empty()) {
    assert(canonicalScalableDims.empty() &&
           "vector sizes not provided but scalable vector sizes provided");
    canonicalVectorSizes = linalgOp.getStaticLoopRanges();
    canonicalScalableDims.append(linalgOp.getNumLoops(), false);

    // loop ranges must be static to infer vector sizes.
    if (ShapedType::isDynamicShape(canonicalVectorSizes)) {
      return failure();
    }
  }

  DenseMap<Value, std::pair<Value, AffineMap>> tensorMap;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    Value blockArg = linalgOp.getMatchingBlockArgument(&operand);
    tensorMap[blockArg] = {operand.get(), map};
  }

  rewriter.setInsertionPointAfter(linalgOp);

  // Build the preExtract linalg.generic and vectorize it.
  linalg::GenericOp preOp = buildPartialGenericOp(
      rewriter, linalgOp, canonicalVectorSizes, preExtract, tensorMap);

  // Build the iree_vector_ext.transfer_gather operation.
  SmallVector<Value> baseOffsets;
  SmallVector<Value> indexVecs;
  SmallVector<AffineMap> indexVecMaps;

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

  // Build the source indexing map. Every index is treated as gathered (symbol).
  // Canonicalization (foldTransferGatherFromStep, FoldSingleElementIndexVec,
  // etc.) will recover contiguous dims where possible.
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<AffineExpr> sourceMapExprs;
  int64_t numSymbols = 0;
  for (auto [i, index] : llvm::enumerate(extractOp.getIndices())) {
    if (!tensorMap.contains(index)) {
      // Value defined outside the block — loop-invariant (constant/broadcast).
      baseOffsets.push_back(index);
      sourceMapExprs.push_back(getAffineConstantExpr(0, ctx));
      continue;
    }

    auto [tensor, map] = tensorMap[index];

    Type elemType = getElementTypeOrSelf(index);
    AffineMap readMap = inverseAndBroadcastProjectedPermutation(map);
    VectorType readType = VectorType::get(canonicalVectorSizes, elemType);

    SmallVector<Value> operandIndices(map.getNumResults(), zero);

    // TODO: Mask the operation here. It's really hard to do that here though
    // because we don't have access to the vectorization infra, but maybe there
    // are easier ways to do it here.
    auto read = vector::TransferReadOp::create(
        rewriter, loc, readType, tensor, operandIndices,
        /*padding=*/std::nullopt, readMap);

    baseOffsets.push_back(zero);
    // This source dim is gathered: use a symbol.
    int64_t symIdx = numSymbols++;
    sourceMapExprs.push_back(getAffineSymbolExpr(symIdx, ctx));
    // The index vec map: the read result has shape canonicalVectorSizes, so
    // it's indexed by all vector dims (identity map).
    indexVecMaps.push_back(
        rewriter.getMultiDimIdentityMap(canonicalVectorSizes.size()));
    indexVecs.push_back(read.getResult());
  }

  auto sourceMap = AffineMap::get(
      /*dimCount=*/canonicalVectorSizes.size(), numSymbols, sourceMapExprs,
      ctx);

  // Build the full indexing_maps array: [sourceMap, indexVecMap0, ...]
  SmallVector<AffineMap> indexingMaps;
  indexingMaps.push_back(sourceMap);
  for (AffineMap &m : indexVecMaps) {
    // Add symbols to each index vec map to match the source map.
    m = AffineMap::get(m.getNumDims(), numSymbols, m.getResults(), ctx);
    indexingMaps.push_back(m);
  }

  auto gatherTy = VectorType::get(canonicalVectorSizes, extractOp.getType());
  Value padding = ub::PoisonOp::create(rewriter, loc, extractOp.getType());

  auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
      rewriter, loc, gatherTy, extractOp.getTensor(), baseOffsets, indexVecs,
      rewriter.getAffineMapArrayAttr(indexingMaps), padding,
      /*mask=*/Value());

  // Create a empty tensor to write to.
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, canonicalVectorSizes,
                                         gatherTy.getElementType());
  SmallVector<Value> writeIndices(canonicalVectorSizes.size(), zero);

  auto writeOp = vector::TransferWriteOp::create(
      rewriter, loc, transferGatherOp.getResult(), emptyOp, writeIndices);

  tensorMap[extractOp.getResult()] = {
      writeOp.getResult(),
      rewriter.getMultiDimIdentityMap(canonicalVectorSizes.size())};

  // Build the postExtract linalg.generic.
  linalg::GenericOp postOp = buildPartialGenericOp(
      rewriter, linalgOp, canonicalVectorSizes, postExtract, tensorMap);

  FailureOr<SmallVector<Value>> preResult =
      vectorizeGatherLikeGenericToTransferGather(
          rewriter, preOp, vectorSizes, scalableVecDims, vectorizeNDExtract);
  if (failed(preResult)) {
    return failure();
  }
  // Replace preOp so its users (e.g., postOp inputs) see the vectorized
  // results.
  rewriter.replaceOp(preOp, *preResult);

  auto postResult = vectorizeGatherLikeGenericToTransferGather(
      rewriter, postOp, vectorSizes, scalableVecDims, vectorizeNDExtract);
  if (failed(postResult)) {
    return failure();
  }

  return *postResult;
}

// Returns true if expr is a valid gather expression for the last input dim:
// either `d_lastLoop` (contiguous load with stride 1) or
// `d_lastLoop * stride + d_m` where d_m is a different AffineDimExpr.
static bool isValidLastDimGatherExpr(AffineExpr expr, unsigned lastLoopDim) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    return dim.getPosition() == lastLoopDim;
  }
  auto binop = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binop || binop.getKind() != AffineExprKind::Add) {
    return false;
  }
  // Accepts: (d_lastLoop | d_lastLoop * c) + d_m  or  d_m + (d_lastLoop |
  // d_lastLoop * c) where d_m is a plain AffineDimExpr != d_lastLoop.
  auto isLastLoopTerm = [lastLoopDim](AffineExpr e) -> bool {
    if (auto dim = dyn_cast<AffineDimExpr>(e)) {
      return dim.getPosition() == lastLoopDim;
    }
    auto mul = dyn_cast<AffineBinaryOpExpr>(e);
    if (!mul || mul.getKind() != AffineExprKind::Mul) {
      return false;
    }
    if (auto dim = dyn_cast<AffineDimExpr>(mul.getLHS())) {
      return dim.getPosition() == lastLoopDim &&
             isa<AffineConstantExpr>(mul.getRHS());
    }
    return false;
  };
  auto isOtherDim = [lastLoopDim](AffineExpr e) -> bool {
    auto dim = dyn_cast<AffineDimExpr>(e);
    return dim && dim.getPosition() != lastLoopDim;
  };
  return (isLastLoopTerm(binop.getLHS()) && isOtherDim(binop.getRHS())) ||
         (isOtherDim(binop.getLHS()) && isLastLoopTerm(binop.getRHS()));
}

static bool isImplicitGather(linalg::GenericOp genericOp) {
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    return false;
  }

  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }

  auto inType =
      cast<RankedTensorType>(genericOp.getDpsInputOperand(0)->get().getType());
  auto outType =
      cast<RankedTensorType>(genericOp.getDpsInitOperand(0)->get().getType());
  if (!inType.hasStaticShape() || !outType.hasStaticShape()) {
    return false;
  }

  AffineMap inputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap outputMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));

  if (inputMap.isProjectedPermutation()) {
    return false;
  }

  // Output map must be identity.
  if (!outputMap.isIdentity()) {
    return false;
  }

  // The last input dim expression must be d_{numLoops-1} or
  // d_{numLoops-1} * stride + d_m (d_m != d_{numLoops-1}). This ensures the
  // innermost loop dim drives the gather or contiguous load.
  unsigned lastLoopDim = genericOp.getNumLoops() - 1;
  AffineExpr lastInputExpr = inputMap.getResult(inputMap.getNumResults() - 1);
  if (!isValidLastDimGatherExpr(lastInputExpr, lastLoopDim)) {
    return false;
  }

  // All leading dims of both input and output tensors must have static size 1;
  // only the last dim can be >= 1.
  auto allLeadingDimsAreOne = [](RankedTensorType type) -> bool {
    ArrayRef<int64_t> shape = type.getShape();
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()) - 1; ++i) {
      if (shape[i] != 1) {
        return false;
      }
    }
    return true;
  };
  if (!allLeadingDimsAreOne(inType) || !allLeadingDimsAreOne(outType)) {
    return false;
  }

  Block *body = genericOp.getBlock();
  return hasSingleElement(*body);
}

// Extract the stride from the last input dim expression.
// Precondition: expr is valid per isValidLastDimGatherExpr — either
// AffineDimExpr(lastLoopDim) or d_lastLoop * stride + d_m (or reversed).
static int64_t extractLastDimStride(AffineExpr expr, unsigned lastLoopDim) {
  if (isa<AffineDimExpr>(expr)) {
    return 1;
  }
  // expr is Add. Check each side for the lastLoop term (plain or scaled).
  auto binop = cast<AffineBinaryOpExpr>(expr);
  for (AffineExpr side : {binop.getLHS(), binop.getRHS()}) {
    if (auto dim = dyn_cast<AffineDimExpr>(side)) {
      if (dim.getPosition() == lastLoopDim) {
        return 1;
      }
    }
    if (auto mul = dyn_cast<AffineBinaryOpExpr>(side)) {
      if (mul.getKind() != AffineExprKind::Mul) {
        continue;
      }
      if (auto dim = dyn_cast<AffineDimExpr>(mul.getLHS())) {
        if (dim.getPosition() == lastLoopDim) {
          return cast<AffineConstantExpr>(mul.getRHS()).getValue();
        }
      }
    }
  }
  llvm_unreachable("isImplicitGather should have rejected this expression");
}

FailureOr<SmallVector<Value>>
vectorizeImplicitGatherToTransferGather(RewriterBase &rewriter,
                                        linalg::GenericOp op,
                                        ArrayRef<int64_t> vectorSizes) {
  if (!isImplicitGather(op)) {
    return failure();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  rewriter.setInsertionPoint(op);

  OpOperand *inOperand = op.getDpsInputOperand(0);
  OpOperand *outOperand = op.getDpsInitOperand(0);
  auto inType = cast<RankedTensorType>(inOperand->get().getType());
  auto outType = cast<RankedTensorType>(outOperand->get().getType());
  Type elemType = outType.getElementType();

  int64_t inRank = inType.getRank();
  int64_t outRank = outType.getRank();
  int64_t numLoops = op.getNumLoops();

  if (vectorSizes.empty()) {
    vectorSizes = outType.getShape();
  }

  auto vectorType = VectorType::get(vectorSizes, elemType);

  AffineMap inputMap = op.getMatchingIndexingMap(inOperand);
  int64_t stride =
      extractLastDimStride(inputMap.getResult(inRank - 1), numLoops - 1);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

  // Build 1D index vector [0, stride, 2*stride, ...] over d_{numLoops-1}.
  // All leading loop dims are size 1 (from isImplicitGather invariant).
  int64_t gatherDimSize = vectorSizes.back();
  auto indexVecType = VectorType::get({gatherDimSize}, rewriter.getIndexType());
  Value step = vector::StepOp::create(rewriter, loc, indexVecType);
  Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
  Value strideVec =
      vector::BroadcastOp::create(rewriter, loc, indexVecType, strideVal);
  Value indices = arith::MulIOp::create(rewriter, loc, step, strideVec);

  // Source map: constant 0 for all leading dims. All non-last loop dims have
  // iteration size 1 (forced by the output identity map + leading output dims
  // == 1), so any input map expression for those dims evaluates to 0.
  SmallVector<AffineExpr> sourceExprs(inRank - 1,
                                      rewriter.getAffineConstantExpr(0));
  sourceExprs.push_back(rewriter.getAffineSymbolExpr(0));
  auto sourceMap =
      AffineMap::get(numLoops, /*symbolCount=*/1, sourceExprs, ctx);

  auto indexMap =
      AffineMap::get(numLoops, /*symbolCount=*/1,
                     {rewriter.getAffineDimExpr(numLoops - 1)}, ctx);

  Value f0 =
      arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(elemType));

  SmallVector<Value> baseOffsets(inRank, c0);
  auto transferGatherOp = IREE::VectorExt::TransferGatherOp::create(
      rewriter, loc, vectorType, inOperand->get(), baseOffsets,
      ValueRange{indices},
      rewriter.getAffineMapArrayAttr({sourceMap, indexMap}), f0, Value());

  SmallVector<Value> writeOffsets(outRank, c0);
  auto transferWriteOp = vector::TransferWriteOp::create(
      rewriter, loc, transferGatherOp.getResult(), outOperand->get(),
      writeOffsets);

  return SmallVector<Value>{transferWriteOp.getResult()};
}

/// External model for all linalg structured ops. Wraps upstream
/// linalg::vectorizeOpPrecondition and linalg::vectorize.
template <typename OpTy>
struct LinalgStructuredOpVectorizationModel
    : VectorizableOpInterface::ExternalModel<
          LinalgStructuredOpVectorizationModel<OpTy>, OpTy> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    bool vectorizeNDExtract = getBoolOption(options, "vectorizeNDExtract");
    bool flatten1DDepthwiseConv =
        getBoolOption(options, "flatten1DDepthwiseConv");
    bool vectorizeToTransferGather =
        getBoolOption(options, "vectorizeToTransferGather");

    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (vectorizeToTransferGather && isImplicitGather(genericOp)) {
        return true;
      }
    }
    return succeeded(linalg::vectorizeOpPrecondition(
        op, vectorSizes, scalableDims, vectorizeNDExtract,
        flatten1DDepthwiseConv));
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    bool vectorizeNDExtract = getBoolOption(options, "vectorizeNDExtract");
    bool flatten1DDepthwiseConv =
        getBoolOption(options, "flatten1DDepthwiseConv");
    bool createNamedContraction =
        getBoolOption(options, "createNamedContraction");

    // Handle gather-like generic vectorization via TransferGather.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      if (getBoolOption(options, "vectorizeToTransferGather")) {
        FailureOr<SmallVector<Value>> implicitGatherResult =
            vectorizeImplicitGatherToTransferGather(rewriter, genericOp,
                                                    vectorSizes);
        if (succeeded(implicitGatherResult)) {
          return *implicitGatherResult;
        }
        FailureOr<SmallVector<Value>> gatherResult =
            vectorizeGatherLikeGenericToTransferGather(
                rewriter, genericOp, vectorSizes, scalableDims,
                vectorizeNDExtract);
        if (succeeded(gatherResult)) {
          return *gatherResult;
        }
        // Fall through to normal vectorization if the gather path did not
        // apply.
      }
    }

    FailureOr<linalg::VectorizationResult> result = linalg::vectorize(
        rewriter, op, vectorSizes, scalableDims, vectorizeNDExtract,
        flatten1DDepthwiseConv, /*assumeDynamicDimsMatchVecSizes=*/false,
        createNamedContraction);

    if (failed(result)) {
      return failure();
    }
    return result->replacements;
  }
};

/// External model for linalg::PackOp, linalg::UnPackOp.
/// These go through linalg::vectorize but are not LinalgOp subclasses.
template <typename OpTy>
struct NonLinalgStructuredOpVectorizationModel
    : VectorizableOpInterface::ExternalModel<
          NonLinalgStructuredOpVectorizationModel<OpTy>, OpTy> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    return succeeded(
        linalg::vectorizeOpPrecondition(op, vectorSizes, scalableDims));
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op, vectorSizes, scalableDims);

    if (failed(result)) {
      return failure();
    }
    return result->replacements;
  }
};

/// External model for tensor::PadOp. Different dialect, goes through
/// linalg::vectorize.
struct PadOpVectorizationModel
    : VectorizableOpInterface::ExternalModel<PadOpVectorizationModel,
                                             tensor::PadOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    return succeeded(
        linalg::vectorizeOpPrecondition(op, vectorSizes, scalableDims));
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    FailureOr<linalg::VectorizationResult> result =
        linalg::vectorize(rewriter, op, vectorSizes, scalableDims);

    if (failed(result)) {
      return failure();
    }
    return result->replacements;
  }
};

/// External model for IREE::Codegen::InnerTiledOp. Reads tensor operands into
/// vectors, creates a vector-semantic InnerTiledOp, and writes results back.
struct InnerTiledOpVectorizationModel
    : VectorizableOpInterface::ExternalModel<InnerTiledOpVectorizationModel,
                                             IREE::Codegen::InnerTiledOp> {

  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto tiledOp = cast<IREE::Codegen::InnerTiledOp>(op);
    if (!tiledOp.hasTensorSemantics()) {
      return false;
    }
    SmallVector<int64_t> loopRanges;
    tiledOp.getIterationBounds(loopRanges);
    // If vector sizes are provided (from tile size analysis or config),
    // dynamic outer shapes are fine - they'll be masked during vectorization.
    // However, vector sizes must be >= the static outer dimension sizes.
    if (!vectorSizes.empty()) {
      return succeeded(
          vector::isValidMaskedInputVector(loopRanges, vectorSizes));
    }
    // Without vector sizes, require static outer shapes.
    return ShapedType::isStaticShape(loopRanges);
  }

  FailureOr<SmallVector<Value>> vectorize(Operation *op, RewriterBase &rewriter,
                                          ArrayRef<int64_t> vectorSizes,
                                          ArrayRef<bool> scalableDims,
                                          DictionaryAttr options) const {
    auto tiledOp = cast<IREE::Codegen::InnerTiledOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(tiledOp);
    Location loc = tiledOp.getLoc();

    SmallVector<ShapedType> argTypes = tiledOp.getOperandShapedTypes();
    SmallVector<AffineMap> indexingMaps = tiledOp.getIndexingMapsArray();

    // If no vector sizes are provided, use static loop ranges and the inBounds
    // attribute instead of masking.
    bool needsMasking = true;
    if (vectorSizes.empty()) {
      SmallVector<int64_t> loopRanges;
      tiledOp.getIterationBounds(loopRanges);
      assert(ShapedType::isStaticShape(loopRanges) &&
             "unable to infer vector sizes");
      vectorSizes = loopRanges;
      needsMasking = false;
    }

    // Construct the zero padding value for each operand. Ideally, we'd need the
    // InnerTile interface to return the padding value to use. If it is not
    // provided, ub::Poison is a better choice. Zero was chosen because the op
    // was designed for matmul, and zero padding is the most common case.
    SmallVector<Value> padValues =
        llvm::map_to_vector(argTypes, [&](ShapedType argType) -> Value {
          return arith::ConstantOp::create(
              rewriter, loc, rewriter.getZeroAttr(argType.getElementType()));
        });

    // Compute the read shape for each operand.
    SmallVector<SmallVector<int64_t>> readShapes;
    for (auto [i, argType] : llvm::enumerate(argTypes)) {
      if (!needsMasking) {
        readShapes.push_back(llvm::to_vector(argType.getShape()));
        continue;
      }
      // Outer dimensions come from vector sizes via the indexing map, inner
      // dimensions are static.
      SmallVector<int64_t> readShape;
      AffineMap map = indexingMaps[i];
      for (AffineExpr expr : map.getResults()) {
        auto dimExpr = cast<AffineDimExpr>(expr);
        readShape.push_back(vectorSizes[dimExpr.getPosition()]);
      }
      ArrayRef<int64_t> innerShape = tiledOp.getOperandInnerShape(i);
      readShape.append(innerShape.begin(), innerShape.end());
      readShapes.push_back(std::move(readShape));
    }

    // Read each operand into a vector, with masking if needed.
    SmallVector<Value> newOperands(tiledOp->getOperands());
    for (auto [operand, readShape, padValue] :
         llvm::zip_equal(newOperands, readShapes, padValues)) {
      operand = vector::createReadOrMaskedRead(
          rewriter, loc, operand, readShape, padValue,
          /*useInBoundsInsteadOfMasking=*/!needsMasking);
    }

    auto newTiledOp = IREE::Codegen::InnerTiledOp::create(
        rewriter, loc,
        ValueRange{newOperands}.take_front(tiledOp.getNumInputs()),
        ValueRange{newOperands}.take_back(tiledOp.getNumOutputs()),
        tiledOp.getIndexingMaps(), tiledOp.getIteratorTypes(),
        tiledOp.getKind(), tiledOp.getSemantics());

    // Write results back to tensor, with masking if needed.
    SmallVector<Value> results;
    for (auto [result, dest] :
         llvm::zip_equal(newTiledOp.getResults(), tiledOp.getOutputs())) {
      Operation *write = vector::createWriteOrMaskedWrite(
          rewriter, loc, result, dest, /*writeIndices=*/{},
          /*useInBoundsInsteadOfMasking=*/!needsMasking);
      results.push_back(write->getResult(0));
    }
    return results;
  }
};

/// Registers the LinalgStructuredOpVectorizationModel for a single op type.
template <typename OpTy>
static void registerInterfaceForLinalgOps(MLIRContext *ctx) {
  OpTy::template attachInterface<LinalgStructuredOpVectorizationModel<OpTy>>(
      *ctx);
}

/// Registers the LinalgStructuredOpVectorizationModel for multiple op types.
template <typename OpTy1, typename OpTy2, typename... More>
static void registerInterfaceForLinalgOps(MLIRContext *ctx) {
  registerInterfaceForLinalgOps<OpTy1>(ctx);
  registerInterfaceForLinalgOps<OpTy2, More...>(ctx);
}

//===----------------------------------------------------------------------===//
// Im2col Vectorization
//===----------------------------------------------------------------------===//

/// Compute the padding mask for im2col vectorization.
/// Gets the valid size from computeIm2colValidSize and converts it to a
/// vector mask. The mask guards against out-of-bounds accesses, so read
/// indices do not need clamping.
static Value computeIm2colPaddingMask(
    OpBuilder &b, Location loc, IREE::LinalgExt::Im2colOp im2colOp,
    const IREE::LinalgExt::Im2colSourceIndices &srcIndices, int64_t vecWidth,
    ArrayRef<Value> outputIVs, std::optional<int64_t> vecOutputDim) {
  auto vecI1Type = VectorType::get({vecWidth}, b.getI1Type());
  OpFoldResult innerTileSize = b.getIndexAttr(vecWidth);
  Value validSize = IREE::LinalgExt::computeIm2colValidSize(
      b, loc, im2colOp, srcIndices, innerTileSize, outputIVs, vecOutputDim);
  return vector::CreateMaskOp::create(b, loc, vecI1Type, validSize);
}

// Im2col vectorization uses the driver-provided vectorSizes to determine
// which output dimension to vectorize and the vector width. The vector
// sizes are computed by the MaterializeVectorTileSizes pass.
struct Im2colOpVectorizationModel
    : public VectorizableOpInterface::ExternalModel<Im2colOpVectorizationModel,
                                                    IREE::LinalgExt::Im2colOp> {
  bool isVectorizable(Operation *op, ArrayRef<int64_t> vectorSizes,
                      ArrayRef<bool> scalableDims,
                      DictionaryAttr options) const {
    auto im2colOp = cast<IREE::LinalgExt::Im2colOp>(op);
    return im2colOp.getOutputType().hasStaticShape();
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

    int64_t inputRank = im2colOp.getInputRank();

    int64_t outputRank = im2colOp.getOutputRank();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    Type elemType = outputType.getElementType();

    // Determine the vectorized dimension from the driver-provided vectorSizes.
    // The vectorized dim has size > 1; all others are 1.
    std::optional<int64_t> vecDim;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (d < static_cast<int64_t>(vectorSizes.size()) && vectorSizes[d] > 1) {
        vecDim = d;
        break;
      }
    }

    int64_t vecWidth = vecDim ? outputShape[*vecDim] : 1;

    auto vecType = VectorType::get({vecWidth}, elemType);

    // Pad value for transfer_read: use im2col's pad_value when padding is
    // present, otherwise use poison (transfer_read requires a padding operand).
    Value padValue = hasPadding ? im2colOp.getPadValue()
                                : ub::PoisonOp::create(rewriter, loc, elemType);

    int64_t writeDim = vecDim ? *vecDim : (outputRank - 1);
    AffineMap writePermMap =
        AffineMap::get(outputRank, 0, rewriter.getAffineDimExpr(writeDim),
                       rewriter.getContext());

    SmallVector<int64_t> loopDims;
    SmallVector<int64_t> loopBounds;
    int64_t totalIters = 1;
    for (int64_t d = 0; d < outputRank; ++d) {
      if (vecDim && d == *vecDim) {
        continue;
      }
      loopDims.push_back(d);
      loopBounds.push_back(outputShape[d]);
      totalIters *= outputShape[d];
    }

    Value result = im2colOp.getOutput();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Hoist loop-invariant padding and clamping state.
    SmallVector<OpFoldResult> padLow(inputRank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> inputPadLow = im2colOp.getMixedInputPadLow();
    if (!inputPadLow.empty()) {
      padLow = inputPadLow;
    }
    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, im2colOp.getInput());
    // AffineMap for clamping: max(d0, 0) and min(d0, d1 - 1).
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    AffineExpr d1 = getAffineDimExpr(1, ctx);
    AffineMap maxZeroMap =
        AffineMap::get(1, 0, {d0, getAffineConstantExpr(0, ctx)}, ctx);
    AffineMap clampHighMap = AffineMap::get(2, 0, {d0, d1 - 1}, ctx);

    for (int64_t iter = 0; iter < totalIters; ++iter) {
      SmallVector<Value> ivs(outputRank, zeroIdx);
      int64_t remaining = iter;
      for (int64_t i = loopDims.size() - 1; i >= 0; --i) {
        int64_t idx = remaining % loopBounds[i];
        remaining /= loopBounds[i];
        ivs[loopDims[i]] = arith::ConstantIndexOp::create(rewriter, loc, idx);
      }

      IREE::LinalgExt::Im2colSourceIndices srcIndices =
          IREE::LinalgExt::computeIm2colSourceIndices(
              rewriter, loc, im2colOp, ivs, rewriter.getIndexAttr(vecWidth));

      // Convert padded-space source offsets to actual input tensor coordinates
      // by subtracting padLow. When there is no padding, padLow is all zeros
      // and subOfrs folds to identity.
      SmallVector<Value> readIndices;
      for (int64_t d = 0; d < inputRank; ++d) {
        OpFoldResult adjusted = IREE::LinalgExt::subOfrs(
            rewriter, loc, srcIndices.sliceOffsets[d], padLow[d]);
        // Clamp to [0, dimSize - 1] so downstream optimizations can prove
        // buffer accesses are in-bounds. The mask already zeros out OOB reads,
        // so clamping doesn't affect correctness.
        if (hasPadding) {
          adjusted = affine::makeComposedFoldedAffineMax(
              rewriter, loc, maxZeroMap, {adjusted});
          adjusted = affine::makeComposedFoldedAffineMin(
              rewriter, loc, clampHighMap, {adjusted, inputSizes[d]});
        }
        readIndices.push_back(
            getValueOrCreateConstantIndexOp(rewriter, loc, adjusted));
      }
      Value mask;
      if (hasPadding) {
        mask = computeIm2colPaddingMask(rewriter, loc, im2colOp, srcIndices,
                                        vecWidth, ivs, vecDim);
      }

      AffineMap readPermMap =
          AffineMap::getMinorIdentityMap(inputRank, 1, rewriter.getContext());
      auto readOp = vector::TransferReadOp::create(
          rewriter, loc, vecType, im2colOp.getInput(), readIndices,
          AffineMapAttr::get(readPermMap), padValue, mask,
          rewriter.getBoolArrayAttr({true}));
      Value readVec = readOp.getResult();

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

  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::InnerTiledOp::attachInterface<
            InnerTiledOpVectorizationModel>(*ctx);
      });

  // Upstream linalg ops.
#define GET_OP_LIST
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::PackOp::attachInterface<
        NonLinalgStructuredOpVectorizationModel<linalg::PackOp>>(*ctx);
    linalg::UnPackOp::attachInterface<
        NonLinalgStructuredOpVectorizationModel<linalg::UnPackOp>>(*ctx);
    registerInterfaceForLinalgOps<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });

  // Upstream tensor ops.
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::PadOp::attachInterface<PadOpVectorizationModel>(*ctx);
  });
}

} // namespace mlir::iree_compiler
