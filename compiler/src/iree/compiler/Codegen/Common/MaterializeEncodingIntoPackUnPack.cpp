// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

//===---------------------------------------------------------------------===//
// Utility methods
//===---------------------------------------------------------------------===//

static Operation *dropEncodingAndCloneOp(OpBuilder &builder, Operation *op,
                                         ValueRange convertedInputOperands,
                                         ValueRange convertedOutputOperands) {
  SmallVector<Value> operands;
  operands.append(convertedInputOperands.begin(), convertedInputOperands.end());
  operands.append(convertedOutputOperands.begin(),
                  convertedOutputOperands.end());
  return mlir::clone(
      builder, op,
      {dropEncoding(
          convertedOutputOperands[0].getType().cast<RankedTensorType>())},
      operands);
}

static FailureOr<SmallVector<OpFoldResult>>
getInnerTileSizesOfr(OpBuilder &rewriter, Location loc,
                     RankedTensorType tensorType,
                     const MaterializeEncodingInfo &materializeEncodingInfo,
                     MaterializeEncodingValueFn materializeEncodingValueFn) {
  ArrayRef<int64_t> staticTileSizes = materializeEncodingInfo.innerTileSizes;
  if (llvm::all_of(staticTileSizes,
                   [](int64_t i) { return !ShapedType::isDynamic(i); })) {
    return getAsOpFoldResult(rewriter.getI64ArrayAttr(staticTileSizes));
  }
  assert(materializeEncodingValueFn &&
         "When dynamic tile sizes are generated, a MaterializeEncodingValueFn "
         "should be provided.");

  FailureOr<MaterializeEncodingValueInfo> materializeEncodingValueInfo =
      materializeEncodingValueFn(tensorType, rewriter, loc);
  if (failed(materializeEncodingValueInfo)) {
    return failure();
  }
  ArrayRef<Value> innerTileSizeValues =
      materializeEncodingValueInfo->innerTileSizes;

  SmallVector<OpFoldResult> result(staticTileSizes.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (staticTileSizes[i] == ShapedType::kDynamic) {
      result[i] = innerTileSizeValues[i];
    } else if (tensorType.isDynamicDim(i)) {
      result[i] =
          rewriter.create<arith::ConstantIndexOp>(loc, staticTileSizes[i])
              .getResult();
    } else {
      result[i] = rewriter.getI64IntegerAttr(staticTileSizes[i]);
    }
  }
  return result;
}

//===---------------------------------------------------------------------===//
// Methods to convert `set_encoding` and `unset_encoding` operations
// to `pack` and `unpack` operations respectively.
//===---------------------------------------------------------------------===//

/// Utility method to get the optional padding value to use with pack operation
/// if source is defined using a `tensor.pad` operation. Note `source` is
/// passed by reference. It is updated to use the source of the pad operation.
static std::optional<Value> getPaddingValue(Value &source) {
  auto padOp = source.getDefiningOp<tensor::PadOp>();
  if (!padOp || padOp.getNofold() || !padOp.hasZeroLowPad()) {
    return std::nullopt;
  }

  Value constantPaddingValue = padOp.getConstantPaddingValue();
  if (!constantPaddingValue) {
    return std::nullopt;
  }

  source = padOp.getSource();
  return constantPaddingValue;
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// For now this takes a `paddingValue` as input. The source is also taken
/// as input so that these could be used with `OpConversionPatterns`.
static FailureOr<tensor::PackOp> lowerSetEncodingOpToPackOp(
    RewriterBase &rewriter, SetEncodingOp encodingOp, Value source,
    MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType resultType = encodingOp.getResultType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(resultType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled result encoding");
  }
  // Create `tensor.empty` operation for the result of the pack operation.
  Location loc = encodingOp.getLoc();
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  auto encoding = getEncodingAttr(resultType);
  if (!encoding) {
    return failure();
  }
  std::optional<Value> paddingValue = getPaddingValue(source);
  SmallVector<OpFoldResult> sourceDims =
      tensor::getMixedSizes(rewriter, loc, source);
  SmallVector<OpFoldResult> resultDims = tensor::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr,
      materializeEncodingInfo->innerDimsPos,
      materializeEncodingInfo->outerDimsPerm);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  resultType.getElementType());
  return rewriter.create<tensor::PackOp>(
      loc, source, emptyOp, materializeEncodingInfo->innerDimsPos,
      *innerTileSizesOfr, paddingValue, materializeEncodingInfo->outerDimsPerm);
}

/// Utility method to convert from `set_encoding` op to `pack` operation.
/// The source is taken as input so that these could be used with
/// `OpConversionPatterns`.
static FailureOr<tensor::UnPackOp> lowerUnsetEncodingToUnpackOp(
    RewriterBase &rewriter, UnsetEncodingOp encodingOp, Value packedValue,
    MaterializeEncodingFn materializeEncodingFn,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  RankedTensorType sourceType = encodingOp.getSourceType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(sourceType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(encodingOp, "unhandled source encoding");
  }
  // Create an `tensor.empty` for the result of the unpack operation.
  Location loc = encodingOp.getLoc();
  SmallVector<OpFoldResult> resultDims =
      tensor::getMixedSizes(rewriter, loc, encodingOp.getSource());
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resultDims,
                                                  sourceType.getElementType());
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, sourceType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  return rewriter.create<tensor::UnPackOp>(
      loc, packedValue, emptyOp, materializeEncodingInfo->innerDimsPos,
      *innerTileSizesOfr, materializeEncodingInfo->outerDimsPerm);
}

static FailureOr<SmallVector<Value>> lowerUpperBoundTileSizeOpToConstants(
    RewriterBase &rewriter, UpperBoundTileSizeOp upperBoundTileSizeOp,
    MaterializeEncodingFn materializeEncodingFn) {
  Location loc = upperBoundTileSizeOp.getLoc();
  RankedTensorType tensorType = upperBoundTileSizeOp.getTensorType();
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(tensorType);
  if (failed(materializeEncodingInfo)) {
    return rewriter.notifyMatchFailure(upperBoundTileSizeOp,
                                       "unhandled source encoding");
  }
  ArrayRef<int64_t> innerTileSizes = materializeEncodingInfo->innerTileSizes;
  ArrayRef<int64_t> innerDimsPos = materializeEncodingInfo->innerDimsPos;
  SmallVector<Value> results(tensorType.getRank());
  for (unsigned i = 0; i < innerTileSizes.size(); ++i) {
    int64_t tileSize = innerTileSizes[i];
    if (ShapedType::isDynamic(tileSize)) {
      tileSize = 16;
    }
    results[innerDimsPos[i]] =
        rewriter.create<arith::ConstantIndexOp>(loc, tileSize);
  }
  // For the dims that have no inner tiles, use 1 as tile size to avoid padding.
  for (unsigned i = 0; i < results.size(); ++i) {
    if (!results[i]) {
      results[i] = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    }
  }
  return results;
}

/// Utility method to convert from `linalg.matmul` with
/// - lhs encoding with role=LHS
/// - rhs encoding with role=RHS
/// - result encoding with role=RESULT
/// to linalg.mmt4d op.
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::MatmulOp matmulOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    MaterializeEncodingFn materializeEncodingFn, MaterializeEncodingValueFn) {
  if (!matmulOp.hasTensorSemantics()) {
    return failure();
  }
  auto inputs = matmulOp.getDpsInputOperands();
  auto outputs = matmulOp.getDpsInits();
  auto lhsEncoding =
      getEncodingAttr(inputs[0]->get().getType().cast<RankedTensorType>());
  auto rhsEncoding =
      getEncodingAttr(inputs[1]->get().getType().cast<RankedTensorType>());
  auto resultEncoding =
      getEncodingAttr(outputs[0].getType().cast<RankedTensorType>());
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }
  if (!isMatmulEncodingUser(lhsEncoding.getUser().getValue()) ||
      !isMatmulEncodingUser(rhsEncoding.getUser().getValue()) ||
      !isMatmulEncodingUser(resultEncoding.getUser().getValue()) ||
      lhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::LHS ||
      rhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RHS ||
      resultEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RESULT) {
    return failure();
  }

  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(getOriginalTypeWithEncoding(
          matmulOp.getResultTypes()[0].cast<RankedTensorType>()));
  Operation *result;
  if (failed(materializeEncodingInfo)) {
    result = dropEncodingAndCloneOp(rewriter, matmulOp, convertedInputOperands,
                                    convertedOutputOperands);
  } else {
    result = rewriter.create<linalg::Mmt4DOp>(
        matmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  }
  return result;
}

/// Utility method to convert from `linalg.batch_matmul` with
/// - lhs encoding with user=BATCH_MATMUL_*, role=LHS
/// - rhs encoding with user=BATCH_MATMUL_*, role=RHS
/// - result encoding with user=BATCH_MATMUL_*, role=RESULT
/// to linalg.batch_mmt4d op.
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::BatchMatmulOp batchMatmulOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    MaterializeEncodingFn materializeEncodingFn, MaterializeEncodingValueFn) {
  if (!batchMatmulOp.hasTensorSemantics())
    return failure();
  auto inputs = batchMatmulOp.getDpsInputOperands();
  auto outputs = batchMatmulOp.getDpsInits();
  auto lhsEncoding =
      getEncodingAttr(inputs[0]->get().getType().cast<RankedTensorType>());
  auto rhsEncoding =
      getEncodingAttr(inputs[1]->get().getType().cast<RankedTensorType>());
  auto resultEncoding =
      getEncodingAttr(outputs[0].getType().cast<RankedTensorType>());
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }

  if (!isBatchMatmulEncodingUser(lhsEncoding.getUser().getValue()) ||
      !isBatchMatmulEncodingUser(rhsEncoding.getUser().getValue()) ||
      !isBatchMatmulEncodingUser(resultEncoding.getUser().getValue()) ||
      lhsEncoding.getRole().getValue() != EncodingRole::LHS ||
      rhsEncoding.getRole().getValue() != EncodingRole::RHS ||
      resultEncoding.getRole().getValue() != EncodingRole::RESULT) {
    return failure();
  }
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(getOriginalTypeWithEncoding(
          batchMatmulOp.getResultTypes()[0].cast<RankedTensorType>()));
  Operation *result;
  if (failed(materializeEncodingInfo)) {
    result =
        dropEncodingAndCloneOp(rewriter, batchMatmulOp, convertedInputOperands,
                               convertedOutputOperands);
  } else {
    result = rewriter.create<linalg::BatchMmt4DOp>(
        batchMatmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  }
  return result;
}

/// Utility method to convert from `linalg.fill` on `tensor` type with
/// encoding to fill of the materialized type
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::FillOp fillOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands, MaterializeEncodingFn,
                    MaterializeEncodingValueFn) {
  if (!fillOp.hasTensorSemantics())
    return failure();
  Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
      fillOp.getLoc(), convertedOutputOperands[0].getType(),
      convertedInputOperands, convertedOutputOperands);
  return materializedFillOp;
}

/// Utility method to convert `tensor.empty` with encoding to a `tensor.empty`
/// of the materialized type.
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, tensor::EmptyOp emptyOp,
                    ValueRange convertedOperands,
                    MaterializeEncodingFn materializeEncodingFn,
                    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto emptyType = emptyOp->getResultTypes()[0].cast<RankedTensorType>();
  auto resultType =
      getOriginalTypeWithEncoding(emptyType).clone(emptyType.getElementType());
  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(resultType);
  Location loc = emptyOp.getLoc();
  if (failed(materializeEncodingInfo)) {
    Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, emptyOp.getMixedSizes(), resultType.getElementType());
    return newEmptyOp;
  }

  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      getInnerTileSizesOfr(rewriter, loc, resultType, *materializeEncodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        emptyOp, "failed to generate runtime tile size query");
  }
  SmallVector<OpFoldResult> sourceDims = emptyOp.getMixedSizes();
  (void)foldDynamicIndexList(sourceDims);
  SmallVector<OpFoldResult> newShape =
      PackOp::getResultShape(rewriter, loc, sourceDims, *innerTileSizesOfr,
                             materializeEncodingInfo->innerDimsPos,
                             materializeEncodingInfo->outerDimsPerm);
  Operation *newEmptyOp = rewriter.create<tensor::EmptyOp>(
      loc, newShape, resultType.getElementType());

  return newEmptyOp;
}

/// Utility method to convert from `linalg.generic` on `tensor` type with
/// encoding to `linalg.generic` on the materialized type
static FailureOr<Operation *>
lowerOpWithEncoding(RewriterBase &rewriter, linalg::GenericOp genericOp,
                    ValueRange convertedInputOperands,
                    ValueRange convertedOutputOperands, MaterializeEncodingFn,
                    MaterializeEncodingValueFn) {
  if (!genericOp.hasTensorSemantics() || !isElementwise(genericOp) ||
      genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "linalg.generic op is not elementwise "
                                       "with single input and single output");
  }
  if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); })) {
    return rewriter.notifyMatchFailure(
        genericOp, "indexing maps are not all identity maps");
  }
  auto convertedResultType =
      convertedOutputOperands[0].getType().cast<RankedTensorType>();
  SmallVector<AffineMap> maps(
      2, AffineMap::getMultiDimIdentityMap(convertedResultType.getRank(),
                                           rewriter.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(convertedResultType.getRank(),
                                                 utils::IteratorType::parallel);
  auto materializedGenericOp = rewriter.create<linalg::GenericOp>(
      genericOp.getLoc(), convertedResultType, convertedInputOperands,
      convertedOutputOperands, maps, iteratorTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(),
                              materializedGenericOp.getRegion(),
                              materializedGenericOp.getRegion().begin());
  return materializedGenericOp.getOperation();
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the materialized shape of the `dispatchTensorType`. The
/// dynamic dimensions of the `dispatchTensorType` are provided in
/// `dynamicDims`.
static FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    const MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto boundTensorType =
      llvm::dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
  if (!boundTensorType) {
    return failure();
  }

  RankedTensorType originalTensorType =
      getOriginalTypeWithEncoding(boundTensorType);

  MaterializeEncodingFn materializeEncodingFn =
      typeConverter.getMaterializeEncodingFn();
  FailureOr<MaterializeEncodingInfo> encodingInfo =
      materializeEncodingFn(boundTensorType);
  if (failed(encodingInfo)) {
    return failure();
  }

  SmallVector<OpFoldResult> targetShape =
      getMixedValues(originalTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes =
      getInnerTileSizesOfr(builder, loc, originalTensorType, *encodingInfo,
                           materializeEncodingValueFn);
  if (failed(innerTileSizes)) {
    return failure();
  }
  SmallVector<OpFoldResult> convertedTargetShape =
      tensor::PackOp::getResultShape(builder, loc, targetShape, *innerTileSizes,
                                     encodingInfo->innerDimsPos,
                                     encodingInfo->outerDimsPerm);
  return convertedTargetShape;
}

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the dynamic dimensions of the materialized shape of the
/// `dispatchTensorType`. The dynamic dimensions of the `dispatchTensorType` are
/// provided in `dynamicDims`.
static FailureOr<SmallVector<Value>> getPackedDynamicDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    const MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  FailureOr<SmallVector<OpFoldResult>> convertedTargetShape =
      getPackedDimsForDispatchTensor(builder, loc, typeConverter,
                                     dispatchTensorType, dynamicDims,
                                     materializeEncodingValueFn);
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
    : public OpMaterializeEncodingPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using OpMaterializeEncodingPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
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
    FailureOr<SmallVector<Value>> convertedDynamicDims =
        getPackedDynamicDimsForDispatchTensor(
            rewriter, loc, *typeConverter, resultType,
            subspanOp.getDynamicDims(), this->materializeEncodingValueFn);
    if (failed(convertedDynamicDims)) {
      return rewriter.notifyMatchFailure(
          subspanOp, "failed to get converted dynamic dims");
    }

    auto newResultType = IREE::Flow::DispatchTensorType::get(
        resultType.getAccess(), convertedBoundType);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getSet(), subspanOp.getBinding(),
        subspanOp.getDescriptorType(), subspanOp.getByteOffset(),
        convertedDynamicDims.value(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorLoadOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorLoadOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = sourceType.getBoundType();
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(loadOp, "bound type already valid");
    }

    Location loc = loadOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       sourceType, loadOp.getSourceDims(),
                                       this->materializeEncodingValueFn);
    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          loadOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        loadOp, adaptor.getSource(), convertedDynamicDims, convertedOffsets,
        convertedMixedSizes.value(), convertedStrides);

    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp
    : public OpMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorStoreOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = targetType.getBoundType();
    auto *typeConverter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

    if (typeConverter->convertType(boundTensorType) == boundTensorType) {
      return rewriter.notifyMatchFailure(storeOp, "bound type already valid");
    }

    Location loc = storeOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensor(rewriter, loc, *typeConverter,
                                       targetType, storeOp.getTargetDims(),
                                       this->materializeEncodingValueFn);
    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          storeOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims);
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), convertedDynamicDims,
        convertedOffsets, convertedMixedSizes.value(), convertedStrides);
    return success();
  }
};

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Convert `set_encoding` op to `pack` op.
struct SetEncodingOpToPackOpConversion
    : public OpMaterializeEncodingPattern<SetEncodingOp> {
  using OpMaterializeEncodingPattern<
      SetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            getTypeConverter())
            ->getMaterializeEncodingFn();
    auto packOp = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(packOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(encodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, packOp->getResult());
    return success();
  }
};

/// Convert `unset_encoding` op to `unpack` op.
struct UnsetEncodingOpToUnPackOpConversion
    : public OpMaterializeEncodingPattern<UnsetEncodingOp> {
  using OpMaterializeEncodingPattern<
      UnsetEncodingOp>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(UnsetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    auto unpackOp = lowerUnsetEncodingToUnpackOp(
        rewriter, encodingOp, adaptor.getSource(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(unpackOp)) {
      Value result = adaptor.getSource();
      Type targetType =
          getTypeConverter()->convertType(encodingOp.getResultType());
      if (targetType != result.getType()) {
        result = rewriter.create<tensor::CastOp>(encodingOp.getLoc(),
                                                 targetType, result);
      }
      rewriter.replaceOp(encodingOp, result);
      return success();
    }
    rewriter.replaceOp(encodingOp, unpackOp->getResult());
    return success();
  }
};

/// Convert `upper_bound_tile_size` op to `constant` op. If the
/// `materializeEncodingFn` returns a failure, the pattern will materialize it
/// to the same shape.
struct UpperBoundTileSizeToConstantOpConversion
    : public OpRewritePattern<UpperBoundTileSizeOp> {
  UpperBoundTileSizeToConstantOpConversion(
      MLIRContext *context, MaterializeEncodingFn materializeEncodingFn)
      : OpRewritePattern<UpperBoundTileSizeOp>(context),
        materializeEncodingFn(materializeEncodingFn) {}

  LogicalResult matchAndRewrite(UpperBoundTileSizeOp upperBoundTileSizeOp,
                                PatternRewriter &rewriter) const override {

    auto constants = lowerUpperBoundTileSizeOpToConstants(
        rewriter, upperBoundTileSizeOp, materializeEncodingFn);
    if (failed(constants)) {
      SmallVector<Value> results(upperBoundTileSizeOp.getNumResults(),
                                 rewriter.create<arith::ConstantIndexOp>(
                                     upperBoundTileSizeOp.getLoc(), 1));
      rewriter.replaceOp(upperBoundTileSizeOp, results);
      return success();
    }
    rewriter.replaceOp(upperBoundTileSizeOp, *constants);
    return success();
  }

  MaterializeEncodingFn materializeEncodingFn;
};

/// Generic pattern to convert operation that is in Destination Passing Style.
template <typename OpTy>
struct MaterializeDPSOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy dpsOp, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, dpsOp, adaptor.getInputs(), adaptor.getOutputs(),
        materializeEncodingFn, this->materializeEncodingValueFn);
    if (failed(convertedOp)) {
      return failure();
    }
    rewriter.replaceOp(dpsOp, convertedOp.value()->getResults());
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public OpMaterializeEncodingPattern<OpTy> {
  using OpMaterializeEncodingPattern<OpTy>::OpMaterializeEncodingPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, op, adaptor.getOperands(), materializeEncodingFn,
        this->materializeEncodingValueFn);
    if (failed(convertedOp))
      return failure();

    SmallVector<Value> replacements;
    for (auto [type, res] : llvm::zip_equal(
             op->getResultTypes(), convertedOp.value()->getResults())) {
      Type targetType = this->getTypeConverter()->convertType(type);
      if (targetType == res.getType()) {
        replacements.push_back(res);
      } else {
        replacements.push_back(
            rewriter.create<tensor::CastOp>(op.getLoc(), targetType, res));
      }
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

static FailureOr<MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc) {
  SmallVector<Type> resultTypes(tensorType.getRank(), builder.getIndexType());
  auto op = builder.create<IREE::Codegen::QueryTileSizesOp>(
      loc, resultTypes, TypeAttr::get(tensorType));
  MaterializeEncodingValueInfo result;
  result.innerTileSizes = op.getResults();
  return result;
}

MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isVMVXBackend(targetAttr) && hasUkernel(targetAttr)) {
    return chooseDynamicEncodingInfoVMVXMicrokernels;
  }
  return {};
}

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  MLIRContext *context = patterns.getContext();

  typeConverter.addConversion(
      [&typeConverter](IREE::Flow::DispatchTensorType dispatchTensorType) {
        Type boundType = dispatchTensorType.getBoundType();
        Type convertedBoundType = typeConverter.convertType(boundType);
        if (convertedBoundType == boundType) {
          return dispatchTensorType;
        }
        return IREE::Flow::DispatchTensorType::get(
            dispatchTensorType.getAccess(), convertedBoundType);
      });

  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
        auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
            subspanOp.getResult().getType());
        // For types that are not `Flow::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });

  // Add all patterns for converting from encoded type to the materialized
  // type.
  patterns.insert<MaterializeDPSOperation<linalg::FillOp>,
                  MaterializeDPSOperation<linalg::MatmulOp>,
                  MaterializeDPSOperation<linalg::BatchMatmulOp>,
                  MaterializeDPSOperation<linalg::GenericOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  SetEncodingOpToPackOpConversion,
                  UnsetEncodingOpToUnPackOpConversion>(
      patterns.getContext(), typeConverter, materializeEncodingValueFn);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);

  patterns.insert<MaterializeFlowDispatchTensorLoadOp,
                  MaterializeFlowDispatchTensorStoreOp,
                  MaterializeInterfaceBindingEncoding>(
      context, typeConverter, materializeEncodingValueFn);
}

void populateMaterializeUpperBoundTileSizePatterns(
    RewritePatternSet &patterns, MaterializeEncodingFn materializeEncodingFn) {
  patterns.insert<UpperBoundTileSizeToConstantOpConversion>(
      patterns.getContext(), materializeEncodingFn);
}

} // namespace iree_compiler
} // namespace mlir
