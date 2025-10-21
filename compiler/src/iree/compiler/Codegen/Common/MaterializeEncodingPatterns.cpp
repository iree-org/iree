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
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());
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
    auto converter = static_cast<const MaterializeEncodingTypeConverter *>(
        getTypeConverter());

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
