// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

#define DEBUG_TYPE "iree-codegen-materialize-encoding"

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileSwizzle;

//===---------------------------------------------------------------------===//
// Methods to convert `set_encoding` and `unset_encoding` operations
// to `pack` and `unpack` operations respectively.
//===---------------------------------------------------------------------===//

FailureOr<Value> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, IREE::Encoding::SetEncodingOp encodingOp,
    Value source, const MaterializeEncodingTypeConverter &typeConverter) {
  RankedTensorType resultType = encodingOp.getResultType();
  MaterializeEncodingInfo encodingInfo =
      typeConverter.getEncodingInfo(resultType);

  // Shortcut to avoid creating new operations.
  if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
    return source;
  }

  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      typeConverter.getInnerTileSizesOfr(rewriter, loc, resultType,
                                         encodingInfo);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(resultType.getElementType()));
  SmallVector<OpFoldResult> sourceDims =
      tensor::getMixedSizes(rewriter, loc, source);
  SmallVector<OpFoldResult> resultDims = linalg::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr, encodingInfo.innerDimsPos,
      encodingInfo.outerDimsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  resultType.getElementType());
  return rewriter
      .create<linalg::PackOp>(loc, source, emptyOp, encodingInfo.innerDimsPos,
                              *innerTileSizesOfr, paddingValue,
                              encodingInfo.outerDimsPerm)
      .getResult();
}

FailureOr<Value> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, IREE::Encoding::UnsetEncodingOp encodingOp,
    Value packedValue, const MaterializeEncodingTypeConverter &typeConverter) {
  RankedTensorType sourceType = encodingOp.getSourceType();
  MaterializeEncodingInfo encodingInfo =
      typeConverter.getEncodingInfo(sourceType);

  // Shortcut to avoid creating new operations.
  if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
    return packedValue;
  }

  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      getMixedValues(encodingOp.getResultType().getShape(),
                     encodingOp.getResultDims(), rewriter);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  sourceType.getElementType());
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      typeConverter.getInnerTileSizesOfr(rewriter, loc, sourceType,
                                         encodingInfo);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  return rewriter
      .create<linalg::UnPackOp>(loc, packedValue, emptyOp,
                                encodingInfo.innerDimsPos, *innerTileSizesOfr,
                                encodingInfo.outerDimsPerm)
      .getResult();
}

/// Utility method to convert `tensor.empty` with encoding to a `tensor.empty`
/// of the materialized type.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, tensor::EmptyOp emptyOp,
                    ValueRange convertedOperands,
                    const MaterializeEncodingTypeConverter &typeConverter) {
  auto emptyType = cast<RankedTensorType>(emptyOp->getResultTypes()[0]);
  MaterializeEncodingInfo encodingInfo =
      typeConverter.getEncodingInfo(emptyType);
  Location loc = emptyOp.getLoc();
  if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
    return rewriter
        .create<tensor::EmptyOp>(loc, emptyOp.getMixedSizes(),
                                 emptyType.getElementType())
        .getOperation();
  }

  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      typeConverter.getInnerTileSizesOfr(rewriter, loc, emptyType,
                                         encodingInfo);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to generate runtime tile size query");
  }

  SmallVector<OpFoldResult> sourceDims = emptyOp.getMixedSizes();
  (void)foldDynamicIndexList(sourceDims);
  SmallVector<OpFoldResult> newShape = linalg::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr, encodingInfo.innerDimsPos,
      encodingInfo.outerDimsPerm);
  newShape = getSwizzledShape(newShape, encodingInfo);
  Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
      loc, newShape, emptyType.getElementType());
  return newEmptyOp;
}

/// Converts a linalg::GenericOp with encoded inputs into the packed domain,
/// with an optional swizzle expansion and permutation if applicable. The
/// `genericOp` must have all parallel iterator types and a single output with
/// an identity indexing map.
static FailureOr<Operation *> lowerGenericOpWithEncoding(
    RewriterBase &rewriter, linalg::GenericOp genericOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    const MaterializeEncodingTypeConverter &typeConverter) {
  OpOperand *outputOperand = genericOp.getDpsInitOperand(0);
  AffineMap outputMap = genericOp.getMatchingIndexingMap(outputOperand);
  if (!outputMap.isIdentity()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Output indexing map is not identity");
  }
  // Step 1: Retrieve the output encoding materialization information and
  // compute the new indexing maps for the packed and potentially swizzled
  // layout. This consists of an outer dimension and inner dimension permutation
  // vectors for the packing and an expanded result dimension permutation vector
  // for the optional swizzling. This assumes that the output map is identity,
  // and that all iterator types are parallel.
  //
  // Running example:
  //
  // Given following output layout:
  //
  // outputType:              tensor<2x128x64xf32>
  // outputPackInfo:          innerDimsPos = [1, 2],
  //                          innerTileSizes = [128, 16]
  //                          outerDimsPerm = [0, 1, 2]
  // outputSwizzle:           expandShape = [[4, 8, 4], [4, 4]]
  //                          permutation = [1, 4, 0, 2, 3]}
  //
  // Retrieve and compute the permutation vectors for the packing outer and
  // inner dimension permutation and for the expanded swizzle permutation. Then,
  // calculate the permutation that would transform the swizzled output
  // dimension map into the identity dimension map. This is the inverse swizzle
  // permutation.
  //
  // outInverseOuterDimsPerm: [0, 1, 2]
  // outInnerDimsPos:         [1, 2]
  // outSwizzlePerm:          [0, 1, 2, 4, 7, 3, 5, 6]
  // invOutSwizzlePerm:       [0, 1, 2, 5, 3, 6, 7, 4]
  MaterializeEncodingInfo outMaterializeEncodingInfo =
      typeConverter.getEncodingInfo(
          cast<RankedTensorType>(outputOperand->get().getType()));
  if (IREE::Codegen::isIdentityLayout(outMaterializeEncodingInfo)) {
    return dropEncodingAndCloneOp(rewriter, genericOp.getOperation(),
                                  convertedInputOperands,
                                  convertedOutputOperands);
  }

  auto convertedResultType =
      cast<RankedTensorType>(convertedOutputOperands[0].getType());
  SmallVector<utils::IteratorType> iteratorTypes(convertedResultType.getRank(),
                                                 utils::IteratorType::parallel);

  SmallVector<int64_t> outInverseOuterDimsPerm =
      invertPermutationVector(outMaterializeEncodingInfo.outerDimsPerm);
  ArrayRef<int64_t> outInnerDimsPos = outMaterializeEncodingInfo.innerDimsPos;
  SmallVector<int64_t> outSwizzlePerm =
      llvm::to_vector(llvm::seq<int64_t>(0, convertedResultType.getRank()));
  if (outMaterializeEncodingInfo.swizzle.has_value()) {
    const int outRank =
        cast<RankedTensorType>(outputOperand->get().getType()).getRank();
    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, outRank));
    for (auto perm : outMaterializeEncodingInfo.swizzle->permutation) {
      transposePerm.push_back(outRank + perm);
    }
    applyPermutationToVector(outSwizzlePerm, transposePerm);
  }
  SmallVector<int64_t> invOutSwizzlePerm =
      invertPermutationVector(outSwizzlePerm);

  // Calculate the running offset for every dimension position for easy lookup
  // when calculating the packed result dimensions for every operand.
  // Example:
  //   expandShape == [[4, 8, 4], [4, 4]]
  // In this case:
  //   outOffsetForDimsPos == [0, 3]
  // So that whenever we need the real dimension for an entry (`outerIndex`,
  // `innerIndex`) in the 2D expanded shape vector, we can calculate it as:
  //   dim(outerIndex, innerIndex) = outOffsetForDimsPos[outerIndex] +
  //   innerIndex
  SmallVector<int64_t> outOffsetForDimsPos(outInnerDimsPos.size(), 0);
  if (outMaterializeEncodingInfo.swizzle.has_value()) {
    int64_t runningSize = 0;
    for (size_t i = 0; i < outInnerDimsPos.size(); i++) {
      outOffsetForDimsPos[i] = runningSize;
      runningSize += outMaterializeEncodingInfo.swizzle->expandShape[i].size();
    }
  }

  SmallVector<AffineMap> packedIndexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    // Step 2: Retrieve the encoding for every input operand and perform the
    // outer dimension permutation, inner dimension expansion and permutation,
    // swizzle expansion and swizzle permutation.
    //
    // Running example:
    //
    // Given the input layout and indexing maps:
    //
    // inputType:       tensor<2x64xf32>
    // innerPackInfo:   innerDimsPos = [1]
    //                  innerTileSizes = [16]
    //                  outerDimsPerm = [0, 1]
    // innerSwizzle:    expandShape = [[4, 4]]
    //                  permutation = [1, 0]
    // inputMap:        [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
    //                   affine_map<(d0, d1, d2) -> (d0, d2)>]
    //
    // 1. Calculate the result dimensions from the indexing maps and perform the
    // outer dimension permutation:
    //
    // packedResultDims: [0, 2]
    //
    // 2. Perform inner dimension expansion, permutation and optional swizzle
    // expansion in one go. In this example, the inner dimension (64) would be
    // expanded into 4x16 based on `innerDimsPos` and `innerTileSizes` above,
    // and then expanded to 4x4x4 based on the swizzle.
    //
    // packedResultDims: [0, 2, 6, 7]
    //
    // 3. Perform the swizzle permutation:
    //
    // packedResultDims: [0, 2, 7, 6]
    MaterializeEncodingInfo materializeEncodingInfo =
        typeConverter.getEncodingInfo(
            cast<RankedTensorType>(inputOperand->get().getType()));
    if (IREE::Codegen::isIdentityLayout(materializeEncodingInfo)) {
      return rewriter.notifyMatchFailure(
          genericOp, "MaterializeEncodingInfo failed for input");
    }
    ArrayRef<int64_t> innerDimsPos = materializeEncodingInfo.innerDimsPos;
    ArrayRef<int64_t> outerDimsPerm = materializeEncodingInfo.outerDimsPerm;
    AffineMap inputMap = genericOp.getMatchingIndexingMap(inputOperand);
    // Permute result dims to the input packed domain, and map dims to the
    // output packed domain.
    SmallVector<int64_t> packedResultDims = llvm::map_to_vector(
        applyPermutation(inputMap.getResults(), outerDimsPerm),
        [&](AffineExpr expr) {
          auto dimExpr = cast<AffineDimExpr>(expr);
          return outInverseOuterDimsPerm[dimExpr.getPosition()];
        });
    // Add new dims for the inner tiles, taking the dim position from the
    // corresponding inner tile of the init operand.
    for (auto [idx, pos] : llvm::enumerate(innerDimsPos)) {
      auto dimPos = cast<AffineDimExpr>(inputMap.getResult(pos)).getPosition();
      for (auto [tileIdx, outDim] : llvm::enumerate(outInnerDimsPos)) {
        if (dimPos != outDim) {
          continue;
        }
        if (!materializeEncodingInfo.swizzle.has_value()) {
          packedResultDims.push_back(outputMap.getNumDims() + tileIdx);
          continue;
        }
        // In case of a layout with swizzle, an expanded set of dimensions
        // needs to be appended as specified by the swizzle's `expandedShape`
        // field. Note that the dimension index should be offset by the
        // calculated output starting offset as every dimension is now
        // transformed into an expanded sequence of indices and the correct
        // dimension index is:
        //   outOffsetForDimsPos[tileIdx] + innerIndex
        assert(idx < materializeEncodingInfo.swizzle->expandShape.size() &&
               "`innerDimsPos` index should not exceed the swizzle's "
               "`expandShape` size");
        const size_t dimSize =
            materializeEncodingInfo.swizzle->expandShape[idx].size();
        const int64_t outIdxOffset =
            outputMap.getNumDims() + outOffsetForDimsPos[tileIdx];
        for (size_t i = 0; i < dimSize; i++) {
          packedResultDims.push_back(outIdxOffset + i);
        }
      }
    }
    // In case of a layout with swizzle, the packed result dimensions need
    // to be transposed according to the swizzle's permutation vector.
    if (materializeEncodingInfo.swizzle.has_value()) {
      int inRank =
          cast<RankedTensorType>(inputOperand->get().getType()).getRank();
      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, inRank));
      for (auto perm : materializeEncodingInfo.swizzle->permutation) {
        transposePerm.push_back(inRank + perm);
      }
      applyPermutationToVector(packedResultDims, transposePerm);
    }

    // Step 3: Calculate the final packed result dimensions through the inverse
    // result dimensions permutation map. This effectively linearizes the packed
    // result dimensions with respect to the output dimensions. For example, if
    // the permuted output dimensions are [D0, D2, D1], this will transform all
    // packed operand result dimensions with the permutation map that would make
    // the output dimensions the identity map [D0, D1, D2], i.e. {D0 -> D0, D1
    // -> D2, D2 -> D1}. Suppose that the operand dimensions are [D0, D2], this
    // operation would transform it into [D0, D1] to align with the output
    // identity map.
    //
    // Running example:
    //
    // The packed and swizzled result dimensions for the input operand:
    //
    // packedResultDims:      [0, 2, 7, 6]
    //
    // Now we need to account for swizzled output result dimensions being
    // linearized to the identity map. This can be achieved by applying
    // `invOutSwizzlePerm` ([0, 1, 2, 5, 3, 6, 7, 4]):
    //
    // finalPackedResultDims: [0, 2, 4, 7]
    SmallVector<int64_t> finalPackedResultDims = llvm::map_to_vector(
        packedResultDims, [&](int64_t r) { return invOutSwizzlePerm[r]; });

    // Create the packed indexing map.
    SmallVector<AffineExpr> packedResultExprs =
        llvm::map_to_vector(finalPackedResultDims, [&](int64_t dim) {
          return rewriter.getAffineDimExpr(dim);
        });
    auto packedInputMap = AffineMap::get(
        /*dimCount=*/iteratorTypes.size(), /*symbolCount=*/0, packedResultExprs,
        rewriter.getContext());
    packedIndexingMaps.push_back(packedInputMap);
  }
  // Create the new packed identity map for the output.
  packedIndexingMaps.push_back(
      rewriter.getMultiDimIdentityMap(convertedResultType.getRank()));
  auto materializedGenericOp = rewriter.create<linalg::GenericOp>(
      genericOp.getLoc(), convertedResultType, convertedInputOperands,
      convertedOutputOperands, packedIndexingMaps, iteratorTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(),
                              materializedGenericOp.getRegion(),
                              materializedGenericOp.getRegion().begin());
  return materializedGenericOp.getOperation();
}

/// Utility method to convert from a linalg::LinalgOp on `tensor` types with
/// encodings to a linalg::LinalgOp on the materialized type. The current
/// supported op types are:
///  - linalg::FillOp
///  - linalg::GenericOp
//   - All the iterators are parallel iterators.
//   - The op has a single output.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands,
                    const MaterializeEncodingTypeConverter &typeConverter) {
  if (!linalgOp.hasPureTensorSemantics()) {
    return rewriter.notifyMatchFailure(linalgOp, "Not pure tensor semantics");
  }
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops()) {
    return rewriter.notifyMatchFailure(linalgOp, "Loops are not all parallel");
  }
  if (linalgOp.getNumDpsInits() != 1) {
    return rewriter.notifyMatchFailure(linalgOp, "Not only 1 init operand");
  }

  return TypeSwitch<Operation *, FailureOr<Operation *>>(linalgOp)
      .Case<linalg::FillOp>(
          [&](linalg::FillOp fillOp) -> FailureOr<Operation *> {
            Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
                fillOp.getLoc(), convertedOutputOperands[0].getType(),
                convertedInputOperands, convertedOutputOperands);
            return materializedFillOp;
          })
      .Case<linalg::GenericOp>(
          [&](linalg::GenericOp genericOp) -> FailureOr<Operation *> {
            return lowerGenericOpWithEncoding(
                rewriter, genericOp, convertedInputOperands,
                convertedOutputOperands, typeConverter);
          })
      .Default([](Operation *op) { return failure(); });
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the dynamic dimensions of the materialized shape of the
/// `dispatchTensorType`. The dynamic dimensions of the `dispatchTensorType` are
/// provided in `dynamicDims`.
static FailureOr<SmallVector<Value>> getPackedDynamicDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    const MaterializeEncodingTypeConverter &typeConverter,
    IREE::TensorExt::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims) {
  FailureOr<SmallVector<OpFoldResult>> convertedTargetShape =
      typeConverter.getPackedDimsForDispatchTensor(
          builder, loc, dispatchTensorType, dynamicDims);
  if (failed(convertedTargetShape)) {
    return failure();
  }
  SmallVector<int64_t> convertedStaticTargetShape;
  SmallVector<Value> convertedDynamicTargetShape;
  dispatchIndexOpFoldResults(convertedTargetShape.value(),
                             convertedDynamicTargetShape,
                             convertedStaticTargetShape);
  return convertedDynamicTargetShape;
}

namespace {
/// Pattern to materialize the encoding for `hal.interface.binding.subspan`
/// operations.
struct MaterializeInterfaceBindingEncoding
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp,
          "expected result type to be !iree_tensor_ext.dispatch.tensor");
    }
    auto boundTensorType =
        llvm::dyn_cast<RankedTensorType>(resultType.getBoundType());
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto convertedBoundType = getTypeConverter()->convertType(boundTensorType);
    if (convertedBoundType == boundTensorType) {
      return rewriter.notifyMatchFailure(subspanOp, "bound type already valid");
    }

    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    // Get the dynamic dims of the target.
    Location loc = subspanOp.getLoc();
    SmallVector<Value> newDynamicDims = subspanOp.getDynamicDims();
    FailureOr<SmallVector<Value>> convertedDynamicDims =
        getPackedDynamicDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                              resultType,
                                              subspanOp.getDynamicDims());
    // Drop the encoding if the target does not support it.
    if (succeeded(convertedDynamicDims)) {
      newDynamicDims = convertedDynamicDims.value();
    }

    auto newResultType = IREE::TensorExt::DispatchTensorType::get(
        resultType.getAccess(), convertedBoundType);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getLayout(), subspanOp.getBinding(),
        subspanOp.getByteOffset(), newDynamicDims, subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.load` operation when
/// materializing the encoding.
struct MaterializeTensorExtDispatchTensorLoadOp
    : public OpConversionPattern<IREE::TensorExt::DispatchTensorLoadOp> {
  using OpConversionPattern<
      IREE::TensorExt::DispatchTensorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::DispatchTensorLoadOp loadOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!iree_tensor_ext.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(loadOp, "bound type already valid");
    }

    SmallVector<OpFoldResult> newOffsets, newMixedSizes, newStrides;
    if (failed(typeConverter->getOffsetsSizesStrides(
            rewriter, loadOp.getLoc(), sourceType, loadOp.getSourceDims(),
            loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
            loadOp.getMixedStrides(), newOffsets, newMixedSizes, newStrides))) {
      return failure();
    }
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
        loadOp, adaptor.getSource(), newDynamicDims, newOffsets, newMixedSizes,
        newStrides);
    return success();
  }
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeTensorExtDispatchTensorStoreOp
    : public OpConversionPattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpConversionPattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!iree_tensor_ext.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(storeOp, "bound type already valid");
    }

    SmallVector<OpFoldResult> newOffsets, newMixedSizes, newStrides;
    if (failed(typeConverter->getOffsetsSizesStrides(
            rewriter, storeOp.getLoc(), targetType, storeOp.getTargetDims(),
            storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
            storeOp.getMixedStrides(), newOffsets, newMixedSizes,
            newStrides))) {
      return failure();
    }
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), newDynamicDims,
        newOffsets, newMixedSizes, newStrides);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Generic pattern to convert operation that is in Destination Passing Style.
template <typename OpTy>
struct MaterializeDPSOperation : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy dpsOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, dpsOp, adaptor.getInputs(), adaptor.getOutputs(), *converter);
    if (failed(convertedOp)) {
      return failure();
    }
    rewriter.replaceOp(dpsOp, convertedOp.value()->getResults());
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());
    FailureOr<Operation *> convertedOp =
        lowerOpWithEncoding(rewriter, op, adaptor.getOperands(), *converter);
    if (failed(convertedOp))
      return failure();

    SmallVector<Value> replacements;
    for (auto [type, res] : llvm::zip_equal(
             op->getResultTypes(), convertedOp.value()->getResults())) {
      Type targetType = this->getTypeConverter()->convertType(type);
      replacements.push_back(
          rewriter.createOrFold<tensor::CastOp>(op.getLoc(), targetType, res));
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct MaterializeOptimizationBarrierOp
    : public OpConversionPattern<IREE::Util::OptimizationBarrierOp> {
  using OpConversionPattern<
      IREE::Util::OptimizationBarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Util::OptimizationBarrierOp op,
                  IREE::Util::OptimizationBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperandTypes(), [](Type type) -> bool {
          auto tensorType = dyn_cast<RankedTensorType>(type);
          return tensorType && tensorType.getEncoding();
        })) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Util::OptimizationBarrierOp>(
        op, adaptor.getOperands());
    return success();
  }
};

static SmallVector<ReassociationIndices>
getReassociationIndices(int outerDims,
                        const TileSwizzle::ExpandShapeType &expandShape) {
  SmallVector<ReassociationIndices> result;
  int expandedIdx = 0;
  for (int i = 0; i < outerDims; ++i) {
    result.push_back({expandedIdx++});
  }
  for (auto expandShapeDim : expandShape) {
    result.push_back({});
    for (int i = 0, e = expandShapeDim.size(); i < e; ++i) {
      result.back().push_back(expandedIdx++);
    }
  }
  return result;
}

/// Convert iree_linalg_ext.set_encoding op to pack + tile swizzling ops. We use
/// expand_shape + linalg.transpose to represent a tile swizzling op.
struct SetEncodingOpLoweringConversion
    : public OpConversionPattern<IREE::Encoding::SetEncodingOp> {
  using OpConversionPattern<IREE::Encoding::SetEncodingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    auto packedValue = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), *converter);
    if (failed(packedValue)) {
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          encodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(encodingOp, result);
      return success();
    }

    MaterializeEncodingInfo encodingInfo =
        converter->getEncodingInfo(encodingOp.getResultType());
    if (!encodingInfo.swizzle) {
      rewriter.replaceOp(encodingOp, packedValue.value());
      return success();
    }

    Location loc = encodingOp.getLoc();

    // Create expand_shape op to tile the innermost two dimensions.
    int origRank = encodingOp.getSourceType().getRank();
    SmallVector<int64_t> expandShapeShape(
        cast<ShapedType>(packedValue->getType())
            .getShape()
            .take_front(origRank));
    expandShapeShape.append(
        getExpandedTileShape(encodingInfo.swizzle->expandShape));
    RankedTensorType expandShapeType =
        encodingOp.getSourceType().clone(expandShapeShape);

    SmallVector<ReassociationIndices> reassociation =
        getReassociationIndices(origRank, encodingInfo.swizzle->expandShape);
    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandShapeType, packedValue.value(), reassociation);

    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, origRank));
    for (auto perm : encodingInfo.swizzle->permutation) {
      transposePerm.push_back(origRank + perm);
    }
    SmallVector<OpFoldResult> transposeResultDims =
        tensor::getMixedSizes(rewriter, loc, expandShapeOp.getResult());
    applyPermutationToVector(transposeResultDims, transposePerm);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, transposeResultDims, encodingOp.getSourceType().getElementType());
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, expandShapeOp, emptyTensor, transposePerm);
    rewriter.replaceOp(encodingOp, transposeOp->getResult(0));

    return success();
  }
};

struct UnsetEncodingOpLoweringConversion
    : public OpConversionPattern<IREE::Encoding::UnsetEncodingOp> {
  using OpConversionPattern<
      IREE::Encoding::UnsetEncodingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Encoding::UnsetEncodingOp unsetEncodingOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    MaterializeEncodingInfo encodingInfo =
        converter->getEncodingInfo(unsetEncodingOp.getSource().getType());
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getSourceType());
      Value result = rewriter.createOrFold<tensor::CastOp>(
          unsetEncodingOp.getLoc(), targetType, adaptor.getSource());
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }

    Location loc = unsetEncodingOp.getLoc();
    Value unpackSrc = adaptor.getSource();
    if (encodingInfo.swizzle) {
      int targetRank = unsetEncodingOp.getResultType().getRank();
      auto srcConvertedType =
          cast<RankedTensorType>(adaptor.getSource().getType());
      SmallVector<OpFoldResult> emptyShape =
          tensor::getMixedSizes(rewriter, loc, adaptor.getSource());
      emptyShape.resize(targetRank);
      for (auto i : getExpandedTileShape(encodingInfo.swizzle->expandShape)) {
        emptyShape.push_back(rewriter.getIndexAttr(i));
      }
      auto emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, emptyShape, unsetEncodingOp.getSourceType().getElementType());

      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, targetRank));
      for (auto perm : encodingInfo.swizzle->permutation) {
        transposePerm.push_back(targetRank + perm);
      }
      auto invertedTransposePerm = invertPermutationVector(transposePerm);
      auto transposeOp = rewriter.create<linalg::TransposeOp>(
          loc, adaptor.getSource(), emptyTensor, invertedTransposePerm);

      SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
          targetRank, encodingInfo.swizzle->expandShape);
      SmallVector<int64_t> unpackSrcShape(
          srcConvertedType.getShape().take_front(targetRank));
      unpackSrcShape.append(encodingInfo.innerTileSizes.begin(),
                            encodingInfo.innerTileSizes.end());
      RankedTensorType unpackSrcType =
          unsetEncodingOp.getResultType().clone(unpackSrcShape);
      unpackSrc = rewriter.create<tensor::CollapseShapeOp>(
          loc, unpackSrcType, transposeOp->getResult(0), reassociation);
    }

    auto unpackedValue = lowerUnsetEncodingToUnpackOp(rewriter, unsetEncodingOp,
                                                      unpackSrc, *converter);
    if (failed(unpackedValue)) {
      Type targetType =
          getTypeConverter()->convertType(unsetEncodingOp.getResultType());
      Value result = rewriter.createOrFold<tensor::CastOp>(loc, targetType,
                                                           adaptor.getSource());
      rewriter.replaceOp(unsetEncodingOp, result);
      return success();
    }
    rewriter.replaceOp(unsetEncodingOp, unpackedValue.value());
    return success();
  }
};

/// Pattern to convert contraction operations.
class MaterializeContractionOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  MaterializeContractionOp(
      const MaterializeEncodingTypeConverter &typeConverter,
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<linalg::LinalgOp>(typeConverter, context,
                                                       benefit) {}

  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(op)) {
      return rewriter.notifyMatchFailure(
          op, "does not implement ContractionOpInterface");
    }

    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        this->getTypeConverter());

    IREE::Encoding::LayoutMaterializerAttr layoutAttr =
        converter->getLayoutAttr();
    SmallVector<Type> convertedResTypes;
    for (auto init : op.getDpsInits()) {
      convertedResTypes.push_back(converter->convertType(init.getType()));
    }
    Operation *newOp =
        layoutAttr.lowerOp(rewriter, op, convertedResTypes, operands);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
        auto resultType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
            subspanOp.getResult().getType());
        // For types that are not `TensorExt::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });
  target.addIllegalOp<IREE::Encoding::SetEncodingOp,
                      IREE::Encoding::UnsetEncodingOp>();
  target.addDynamicallyLegalOp<IREE::TensorExt::DispatchTensorStoreOp>(
      [&typeConverter](IREE::TensorExt::DispatchTensorStoreOp storeOp) {
        auto resultType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
            storeOp.getTargetType());
        // For types that are not `TensorExt::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });
  target.addDynamicallyLegalOp<IREE::TensorExt::DispatchTensorLoadOp>(
      [&typeConverter](IREE::TensorExt::DispatchTensorLoadOp loadOp) {
        auto resultType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
            loadOp.getSourceType());
        // For types that are not `TensorExt::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });

  patterns.insert<MaterializeContractionOp, SetEncodingOpLoweringConversion,
                  UnsetEncodingOpLoweringConversion,
                  MaterializeDPSOperation<linalg::FillOp>,
                  MaterializeDPSOperation<linalg::GenericOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  MaterializeOptimizationBarrierOp,
                  MaterializeTensorExtDispatchTensorLoadOp,
                  MaterializeTensorExtDispatchTensorStoreOp,
                  MaterializeInterfaceBindingEncoding>(typeConverter, context);
};

} // namespace mlir::iree_compiler
