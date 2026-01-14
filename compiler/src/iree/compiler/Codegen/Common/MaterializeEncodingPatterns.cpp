// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
  Value paddingValue = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(resultType.getElementType()));
  SmallVector<OpFoldResult> sourceDims =
      tensor::getMixedSizes(rewriter, loc, source);
  SmallVector<OpFoldResult> resultDims = linalg::PackOp::getResultShape(
      rewriter, loc, sourceDims, *innerTileSizesOfr, encodingInfo.innerDimsPos,
      encodingInfo.outerDimsPerm);
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, resultDims,
                                         resultType.getElementType());
  return linalg::PackOp::create(rewriter, loc, source, emptyOp,
                                encodingInfo.innerDimsPos, *innerTileSizesOfr,
                                paddingValue, encodingInfo.outerDimsPerm)
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
  auto emptyOp = tensor::EmptyOp::create(rewriter, loc, resultDims,
                                         sourceType.getElementType());
  FailureOr<SmallVector<OpFoldResult>> innerTileSizesOfr =
      typeConverter.getInnerTileSizesOfr(rewriter, loc, sourceType,
                                         encodingInfo);
  if (failed(innerTileSizesOfr)) {
    return rewriter.notifyMatchFailure(
        encodingOp, "failed to generate runtime tile size query");
  }
  return linalg::UnPackOp::create(rewriter, loc, packedValue, emptyOp,
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
    return tensor::EmptyOp::create(rewriter, loc, emptyOp.getMixedSizes(),
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
  Operation *newEmptyOp = tensor::EmptyOp::create(rewriter, loc, newShape,
                                                  emptyType.getElementType());
  return newEmptyOp;
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
    auto origResultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!origResultType) {
      return rewriter.notifyMatchFailure(
          subspanOp,
          "expected result type to be !iree_tensor_ext.dispatch.tensor");
    }
    auto origBoundTensorType =
        dyn_cast<RankedTensorType>(origResultType.getBoundType());
    if (!origBoundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto typeConverter = getTypeConverter<MaterializeEncodingTypeConverter>();
    auto convertedResultType =
        typeConverter->convertType<IREE::TensorExt::DispatchTensorType>(
            origResultType);
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(subspanOp,
                                         "expected converted result type to be "
                                         "!iree_tensor_ext.dispatch.tensor");
    }
    if (origResultType == convertedResultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "DispatchTensorType type already valid");
    }

    // Get the dynamic dims of the target.
    // TODO(hanchung): We only have getOffsetsSizesStrides interface method that
    // handles all three together. It would be cleaner to have a separate method
    // to get dynamic sizes only.
    Location loc = subspanOp.getLoc();
    ValueRange origDynamicDims = subspanOp.getDynamicDims();
    SmallVector<OpFoldResult> origSizes = getMixedValues(
        origBoundTensorType.getShape(), origDynamicDims, rewriter);
    SmallVector<OpFoldResult> origOffsets(origDynamicDims.size(),
                                          rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> origStrides(origDynamicDims.size(),
                                          rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(typeConverter->getOffsetsSizesStrides(
            rewriter, loc, origResultType, origDynamicDims, origOffsets,
            origSizes, origStrides, newOffsets, newSizes, newStrides))) {
      return failure();
    }

    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, convertedResultType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(), newDynamicDims,
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());
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
    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = cast<RankedTensorType>(sourceType.getBoundType());
    auto typeConverter = getTypeConverter<MaterializeEncodingTypeConverter>();
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
    auto targetType = storeOp.getTargetType();
    auto boundTensorType = cast<RankedTensorType>(targetType.getBoundType());
    auto typeConverter = getTypeConverter<MaterializeEncodingTypeConverter>();
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
// Patterns for layout transfers. They decompse load/store ops into
// set_encoding/unset_encoding + load/store, if the converted types mismatch.
//===---------------------------------------------------------------------===//

/// Returns the value that brings `src` to `destType` by inserting the necessary
/// encoding ops.
static Value generateEncodingTransferOps(RewriterBase &rewriter, Value src,
                                         ArrayRef<Value> dynamicDims,
                                         RankedTensorType destType) {
  auto srcType = cast<RankedTensorType>(src.getType());
  if (srcType == destType) {
    return src;
  }
  Value value = src;
  if (srcType.getEncoding()) {
    value = IREE::Encoding::UnsetEncodingOp::create(
        rewriter, src.getLoc(), srcType.dropEncoding(), value, dynamicDims,
        /*encodingDims=*/ValueRange{});
  }
  if (destType.getEncoding()) {
    value = IREE::Encoding::SetEncodingOp::create(
        rewriter, src.getLoc(), destType, value, /*encodingDims=*/ValueRange{});
  }
  return value;
}

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.load` operation when
/// materializing the encoding.
struct DecomposeMismatchEncodingTensorLoadOp
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorLoadOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorLoadOp>::OpRewritePattern;

  DecomposeMismatchEncodingTensorLoadOp(
      MaterializeEncodingTypeConverter &converter, MLIRContext *ctx,
      PatternBenefit benefit = 0)
      : OpRewritePattern<IREE::TensorExt::DispatchTensorLoadOp>(ctx, benefit),
        typeConverter(converter) {}

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    IREE::TensorExt::DispatchTensorType srcType = loadOp.getSourceType();
    auto boundTensorType = dyn_cast<RankedTensorType>(srcType.getBoundType());
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          loadOp, "source bound type is not a RankedTensorType");
    }

    // Only decompose if there's an encoding involved. If neither the source
    // nor the destination has an encoding, this pattern should not match.
    // This can happen when isLoadOfWholeSource() returns true but the load
    // reshapes the tensor (e.g., loading a 4D tensor from a 5D source).
    RankedTensorType destType = loadOp.getResult().getType();
    if (!boundTensorType.getEncoding() && !destType.getEncoding()) {
      return rewriter.notifyMatchFailure(
          loadOp, "no encoding involved in source or destination");
    }

    // We have to check the bound type from converted DispatchTensorType because
    // it is what we'll see in encoding materialization. E.g.,
    // GPUPaddingResolver converts RankedTensorType into the same type, but it
    // creates different IREE::TensorExt::DispatchTensorType that may have
    // larger tensor shape for bound type.
    auto convertedSrcType =
        typeConverter.convertType<IREE::TensorExt::DispatchTensorType>(srcType);
    if (typeConverter.convertType(convertedSrcType.getBoundType()) ==
        typeConverter.convertType(destType)) {
      return rewriter.notifyMatchFailure(
          loadOp, "the source type and the result type match after conversion");
    }

    LDBG() << "Performance warning: decomposing mismatched encoding load op: "
           << loadOp;
    Location loc = loadOp.getLoc();
    Value result = IREE::TensorExt::DispatchTensorLoadOp::create(
        rewriter, loc, boundTensorType, loadOp.getSource(),
        loadOp.getSourceDims(), loadOp.getMixedOffsets(),
        loadOp.getMixedSizes(), loadOp.getMixedStrides());
    SmallVector<Value> dynamicDims = llvm::to_vector(loadOp.getSizes());
    result =
        generateEncodingTransferOps(rewriter, result, dynamicDims, destType);
    rewriter.replaceOp(loadOp, result);
    return success();
  }

private:
  MaterializeEncodingTypeConverter &typeConverter;
};

/// Pattern to convert `iree_tensor_ext.dispatch.tensor.store` operation when
/// materializing the encoding.
struct DecomposeMismatchEncodingTensorStoreOp
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  DecomposeMismatchEncodingTensorStoreOp(
      MaterializeEncodingTypeConverter &converter, MLIRContext *ctx,
      PatternBenefit benefit = 0)
      : OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp>(ctx, benefit),
        typeConverter(converter) {}

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    IREE::TensorExt::DispatchTensorType targetType = storeOp.getTargetType();
    auto boundTensorType =
        dyn_cast<RankedTensorType>(targetType.getBoundType());
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          storeOp, "target bound type is not a RankedTensorType");
    }

    // Only decompose if there's an encoding involved. If neither the value
    // nor the target has an encoding, this pattern should not match.
    // This can happen when isStoreToWholeTarget() returns true but the store
    // reshapes the tensor (e.g., storing a 4D tensor to a 5D target).
    RankedTensorType valueType = storeOp.getValue().getType();
    if (!boundTensorType.getEncoding() && !valueType.getEncoding()) {
      return rewriter.notifyMatchFailure(
          storeOp, "no encoding involved in value or target");
    }

    // Similar to DecomposeMismatchEncodingTensorLoadOp, we have to check with
    // the bound type from converted DispatchTensorType.
    auto convertedTargetType =
        typeConverter.convertType<IREE::TensorExt::DispatchTensorType>(
            targetType);
    if (typeConverter.convertType(convertedTargetType.getBoundType()) ==
        typeConverter.convertType(valueType)) {
      return rewriter.notifyMatchFailure(
          storeOp, "the value type and the target type match");
    }

    LDBG() << "Performance warning: decomposing mismatched encoding store op: "
           << storeOp;
    Location loc = storeOp.getLoc();
    Value valueToStore = storeOp.getValue();
    SmallVector<Value> dynamicDims = llvm::to_vector(storeOp.getSizes());
    valueToStore = generateEncodingTransferOps(rewriter, valueToStore,
                                               dynamicDims, boundTensorType);
    IREE::TensorExt::DispatchTensorStoreOp::create(
        rewriter, loc, valueToStore, storeOp.getTarget(),
        storeOp.getTargetDims(), storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());
    rewriter.eraseOp(storeOp);
    return success();
  }

private:
  MaterializeEncodingTypeConverter &typeConverter;
};

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter =
        this->template getTypeConverter<MaterializeEncodingTypeConverter>();
    FailureOr<Operation *> convertedOp =
        lowerOpWithEncoding(rewriter, op, adaptor.getOperands(), *converter);
    if (failed(convertedOp))
      return failure();

    rewriter.replaceOp(op, convertedOp.value());
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
  using Base::Base;

  LogicalResult
  matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (encodingOp.getSource().getType().getRank() == 0) {
      rewriter.replaceOp(encodingOp, adaptor.getSource());
      return success();
    }
    auto converter = getTypeConverter<MaterializeEncodingTypeConverter>();
    auto packedValue = lowerSetEncodingOpToPackOp(
        rewriter, encodingOp, adaptor.getSource(), *converter);
    if (failed(packedValue)) {
      rewriter.replaceOp(encodingOp, adaptor.getSource());
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
    auto expandShapeOp = tensor::ExpandShapeOp::create(
        rewriter, loc, expandShapeType, packedValue.value(), reassociation);

    SmallVector<int64_t> transposePerm =
        llvm::to_vector(llvm::seq<int64_t>(0, origRank));
    for (auto perm : encodingInfo.swizzle->permutation) {
      transposePerm.push_back(origRank + perm);
    }
    SmallVector<OpFoldResult> transposeResultDims =
        tensor::getMixedSizes(rewriter, loc, expandShapeOp.getResult());
    applyPermutationToVector(transposeResultDims, transposePerm);

    auto emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, transposeResultDims,
                                encodingOp.getSourceType().getElementType());
    auto transposeOp = linalg::TransposeOp::create(rewriter, loc, expandShapeOp,
                                                   emptyTensor, transposePerm);
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
    auto converter = getTypeConverter<MaterializeEncodingTypeConverter>();
    MaterializeEncodingInfo encodingInfo =
        converter->getEncodingInfo(unsetEncodingOp.getSource().getType());
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      rewriter.replaceOp(unsetEncodingOp, adaptor.getSource());
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
      auto emptyTensor = tensor::EmptyOp::create(
          rewriter, loc, emptyShape,
          unsetEncodingOp.getSourceType().getElementType());

      SmallVector<int64_t> transposePerm =
          llvm::to_vector(llvm::seq<int64_t>(0, targetRank));
      for (auto perm : encodingInfo.swizzle->permutation) {
        transposePerm.push_back(targetRank + perm);
      }
      auto invertedTransposePerm = invertPermutationVector(transposePerm);
      auto transposeOp =
          linalg::TransposeOp::create(rewriter, loc, adaptor.getSource(),
                                      emptyTensor, invertedTransposePerm);

      SmallVector<ReassociationIndices> reassociation = getReassociationIndices(
          targetRank, encodingInfo.swizzle->expandShape);
      SmallVector<int64_t> unpackSrcShape(
          srcConvertedType.getShape().take_front(targetRank));
      unpackSrcShape.append(encodingInfo.innerTileSizes.begin(),
                            encodingInfo.innerTileSizes.end());
      RankedTensorType unpackSrcType =
          unsetEncodingOp.getResultType().clone(unpackSrcShape);
      unpackSrc = tensor::CollapseShapeOp::create(rewriter, loc, unpackSrcType,
                                                  transposeOp->getResult(0),
                                                  reassociation);
    }

    auto unpackedValue = lowerUnsetEncodingToUnpackOp(rewriter, unsetEncodingOp,
                                                      unpackSrc, *converter);
    if (failed(unpackedValue)) {
      rewriter.replaceOp(unsetEncodingOp, adaptor.getSource());
      return success();
    }
    rewriter.replaceOp(unsetEncodingOp, unpackedValue.value());
    return success();
  }
};

/// Pattern to rewrite linalg::LinalgOp by materializing its encoding using the
/// provided LayoutMaterializerAttr.
class MaterializeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  MaterializeLinalgOp(const MaterializeEncodingTypeConverter &typeConverter,
                      MLIRContext *context, PatternBenefit benefit = 1)
      : OpInterfaceConversionPattern<linalg::LinalgOp>(typeConverter, context,
                                                       benefit) {}

  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter<MaterializeEncodingTypeConverter>();
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

static bool isRankedTensorTypeWithEncoding(Type type) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (!rankedTensorType) {
    return false;
  }
  return rankedTensorType.getEncoding() ? true : false;
}

struct MaterializeFuncReturnOp final
    : public OpConversionPattern<func::ReturnOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateDecomposeMismatchedLayoutLoadStoreOpsPatterns(
    RewritePatternSet &patterns,
    MaterializeEncodingTypeConverter &typeConverter) {
  patterns.insert<DecomposeMismatchEncodingTensorLoadOp,
                  DecomposeMismatchEncodingTensorStoreOp>(
      typeConverter, patterns.getContext());
}

void populateMaterializeEncodingPatterns(
    RewritePatternSet &patterns, MaterializeEncodingConversionTarget &target,
    MaterializeEncodingTypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
      [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
        auto resultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
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
        auto resultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
            storeOp.getTargetType());
        // For types that are not `TensorExt::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });
  target.addDynamicallyLegalOp<IREE::TensorExt::DispatchTensorLoadOp>(
      [&typeConverter](IREE::TensorExt::DispatchTensorLoadOp loadOp) {
        auto resultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
            loadOp.getSourceType());
        // For types that are not `TensorExt::DispatchTensorType` mark as legal.
        if (!resultType)
          return true;
        return resultType == typeConverter.convertType(resultType);
      });
  target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp returnOp) {
    return !llvm::any_of(returnOp.getOperandTypes(),
                         isRankedTensorTypeWithEncoding);
  });

  patterns.insert<MaterializeLinalgOp, SetEncodingOpLoweringConversion,
                  UnsetEncodingOpLoweringConversion,
                  MaterializeOperation<tensor::EmptyOp>,
                  MaterializeOptimizationBarrierOp,
                  MaterializeTensorExtDispatchTensorLoadOp,
                  MaterializeTensorExtDispatchTensorStoreOp,
                  MaterializeInterfaceBindingEncoding, MaterializeFuncReturnOp>(
      typeConverter, context);
};

} // namespace mlir::iree_compiler
