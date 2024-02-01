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
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

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
    if (ShapedType::isDynamic(staticTileSizes[i])) {
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

RankedTensorType getExpandedType(RankedTensorType type, bool isBatched,
                                 bool isTransposed,
                                 SmallVectorImpl<ReassociationIndices> &ri) {
  if (!isBatched) {
    ri.assign({{0, 1}, {2, 3}});
    if (!isTransposed) {
      return RankedTensorType::get(
          {1, type.getDimSize(0), 1, type.getDimSize(1)},
          type.getElementType());
    }
    return RankedTensorType::get({type.getDimSize(0), 1, type.getDimSize(1), 1},
                                 type.getElementType());
  }

  ri.assign({{0}, {1, 2}, {3, 4}});
  if (!isTransposed) {
    return RankedTensorType::get(
        {type.getDimSize(0), 1, type.getDimSize(1), 1, type.getDimSize(2)},
        type.getElementType());
  }
  return RankedTensorType::get(
      {type.getDimSize(0), type.getDimSize(1), 1, type.getDimSize(2), 1},
      type.getElementType());
}

/// Given an input Value and a desired output element type, create and return
/// an element-wise linalg::GenericOp that extends the input Value to the
/// output element type.
static Value createElementWiseExtUIOp(RewriterBase &rewriter, Value input,
                                      Location loc, Type outElemType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto castedType = inputType.clone(outElemType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(rewriter, loc, input);
  Value init =
      rewriter.create<tensor::EmptyOp>(loc, inputMixedSizes, outElemType);
  return rewriter
      .create<linalg::GenericOp>(
          loc, castedType, input, init, maps, iteratorTypes,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value castRes =
                b.create<arith::ExtUIOp>(nestedLoc, outElemType, args[0])
                    ->getResult(0);
            b.create<linalg::YieldOp>(nestedLoc, castRes);
          })
      .getResult(0);
}

/// If needed, expand and the input Value, and return the resulting input with
/// the canonical mmt4d input shape. If the input element type is unsigned,
/// create a producer Linalg::GenericOp on the input that unsigned extends the
/// input to the output element type. This extension is required to keep the
/// unsignedness information on the input for ukernels.
Value getMmt4dOperand(Value value, linalg::LinalgOp linalgOp,
                      RewriterBase &rewriter,
                      SmallVectorImpl<ReassociationIndices> &ri,
                      ArrayRef<Type> elemTypes, int operandIdx) {
  assert(linalgOp.getNumDpsInputs() == 2);
  assert(linalgOp.getNumDpsInits() == 1);
  auto cDims = linalg::inferContractionDims(linalgOp);
  Location loc = linalgOp->getLoc();
  Value expandedValue = value;
  // If vecmat with non-rhs operandIdx or matvec with non-lhs operandIdx, the
  // operand is a vector and must be extended
  if ((cDims->m.empty() && operandIdx != 1) ||
      (cDims->n.empty() && operandIdx != 0)) {
    auto type = value.getType().cast<RankedTensorType>();
    RankedTensorType newType = getExpandedType(
        type, /*isBatched=*/!cDims->batch.empty(),
        /*isTransposed=*/operandIdx == 2 && cDims->n.empty(), ri);
    expandedValue =
        rewriter.create<tensor::ExpandShapeOp>(loc, newType, value, ri);
  }
  if (elemTypes[operandIdx].isUnsignedInteger()) {
    return createElementWiseExtUIOp(rewriter, expandedValue, loc,
                                    elemTypes.back());
  }
  return expandedValue;
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

static FailureOr<Operation *>
lowerContractionOpWithEncoding(RewriterBase &rewriter,
                               linalg::LinalgOp linalgOp, ValueRange operands,
                               MaterializeEncodingFn materializeEncodingFn) {
  if (!linalgOp.hasPureTensorSemantics())
    return failure();

  auto inputs = linalgOp.getDpsInputOperands();
  auto outputs = linalgOp.getDpsInits();

  auto lhsType = inputs[0]->get().getType().cast<RankedTensorType>();
  auto rhsType = inputs[1]->get().getType().cast<RankedTensorType>();
  auto resultType = outputs[0].getType().cast<RankedTensorType>();
  auto lhsEncoding = getEncodingAttr(lhsType);
  auto rhsEncoding = getEncodingAttr(rhsType);
  auto resultEncoding = getEncodingAttr(resultType);
  if (!lhsEncoding || !rhsEncoding || !resultEncoding) {
    return failure();
  }

  if (lhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::LHS ||
      rhsEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RHS ||
      resultEncoding.getRole().getValue() !=
          mlir::iree_compiler::IREE::LinalgExt::EncodingRole::RESULT) {
    return failure();
  }

  FailureOr<MaterializeEncodingInfo> materializeEncodingInfo =
      materializeEncodingFn(getOriginalTypeWithEncoding(
          linalgOp->getResultTypes()[0].cast<RankedTensorType>()));

  Operation *result;
  if (failed(materializeEncodingInfo)) {
    result = dropEncodingAndCloneOp(rewriter, linalgOp,
                                    operands.take_front(inputs.size()),
                                    operands.drop_front(inputs.size()));
  } else {
    auto elemTypes = llvm::map_to_vector(
        lhsEncoding.getElementTypes().getValue(),
        [](Attribute a) { return a.cast<TypeAttr>().getValue(); });
    SmallVector<ReassociationIndices> ri;
    Value newLhs =
        getMmt4dOperand(operands[0], linalgOp, rewriter, ri, elemTypes,
                        /*operandIdx=*/0);
    Value newRhs =
        getMmt4dOperand(operands[1], linalgOp, rewriter, ri, elemTypes,
                        /*operandIdx=*/1);
    Value newResult =
        getMmt4dOperand(operands[2], linalgOp, rewriter, ri, elemTypes,
                        /*operandIdx=*/2);

    Type newResultType = newResult.getType();

    auto cDims = getEncodingContractionDims(lhsEncoding);
    if (cDims->batch.empty()) {
      result = rewriter.create<linalg::Mmt4DOp>(
          linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
          ValueRange{newResult});
    } else {
      result = rewriter.create<linalg::BatchMmt4DOp>(
          linalgOp.getLoc(), newResultType, ValueRange{newLhs, newRhs},
          ValueRange{newResult});
    }
    if (!ri.empty()) {
      result = rewriter.create<tensor::CollapseShapeOp>(
          linalgOp->getLoc(), operands[2].getType(), result->getResult(0), ri);
    }
  }
  return result;
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

/// Utility method to convert from a linalg::LinalgOp on `tensor` types with
/// encodings to a linalg::LinalgOp on the materialized type. The current
/// supported op types are:
///  - linalg::LinalgOp that `isaContractionOpInterface`
///  - linalg::FillOp
///  - element-wise linalg::GenericOp with single input and output
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands,
    MaterializeEncodingFn materializeEncodingFn, MaterializeEncodingValueFn) {
  if (linalg::isaContractionOpInterface(linalgOp)) {
    SmallVector<Value> operands;
    operands.append(convertedInputOperands.begin(),
                    convertedInputOperands.end());
    operands.append(convertedOutputOperands.begin(),
                    convertedOutputOperands.end());
    return lowerContractionOpWithEncoding(rewriter, linalgOp, operands,
                                          materializeEncodingFn);
  }

  return TypeSwitch<Operation *, FailureOr<Operation *>>(linalgOp)
      .Case<linalg::FillOp>(
          [&](linalg::FillOp fillOp) -> FailureOr<Operation *> {
            if (!fillOp.hasPureTensorSemantics())
              return failure();
            Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
                fillOp.getLoc(), convertedOutputOperands[0].getType(),
                convertedInputOperands, convertedOutputOperands);
            return materializedFillOp;
          })
      .Case<linalg::GenericOp>([&](linalg::GenericOp genericOp)
                                   -> FailureOr<Operation *> {
        if (!genericOp.hasPureTensorSemantics() || !isElementwise(genericOp) ||
            genericOp.getNumDpsInputs() != 1 ||
            genericOp.getNumDpsInits() != 1) {
          return rewriter.notifyMatchFailure(
              genericOp, "linalg.generic op is not elementwise "
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
        SmallVector<utils::IteratorType> iteratorTypes(
            convertedResultType.getRank(), utils::IteratorType::parallel);
        auto materializedGenericOp = rewriter.create<linalg::GenericOp>(
            genericOp.getLoc(), convertedResultType, convertedInputOperands,
            convertedOutputOperands, maps, iteratorTypes,
            /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
        rewriter.inlineRegionBefore(genericOp.getRegion(),
                                    materializedGenericOp.getRegion(),
                                    materializedGenericOp.getRegion().begin());
        return materializedGenericOp.getOperation();
      })
      .Default([](Operation *op) { return failure(); });
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

/// Pattern to convert contraction operations.
class MaterializeContractionOp : public OpInterfaceConversionPattern<
                                     mlir::linalg::ContractionOpInterface> {
public:
  MaterializeContractionOp(
      MLIRContext *context,
      const MaterializeEncodingTypeConverter &typeConverter,
      MaterializeEncodingValueFn materializeEncodingValueFn = {},
      PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<mlir::linalg::ContractionOpInterface>(
            typeConverter, context, benefit),
        materializeEncodingValueFn(materializeEncodingValueFn) {}

  LogicalResult
  matchAndRewrite(mlir::linalg::ContractionOpInterface op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MaterializeEncodingFn materializeEncodingFn =
        static_cast<const MaterializeEncodingTypeConverter *>(
            this->getTypeConverter())
            ->getMaterializeEncodingFn();
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp || operands.size() != 3) {
      return failure();
    }
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, linalgOp, operands.take_front(2), operands.take_back(1),
        materializeEncodingFn, this->materializeEncodingValueFn);
    if (failed(convertedOp)) {
      return failure();
    }
    rewriter.replaceOp(op.getOperation(), convertedOp.value()->getResult(0));
    return success();
  }

protected:
  const MaterializeEncodingValueFn materializeEncodingValueFn;
};

} // namespace

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
                  MaterializeDPSOperation<linalg::GenericOp>,
                  MaterializeOperation<tensor::EmptyOp>,
                  MaterializeContractionOp, SetEncodingOpToPackOpConversion,
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

} // namespace mlir::iree_compiler
