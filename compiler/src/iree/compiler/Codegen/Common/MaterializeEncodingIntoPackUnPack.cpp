// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Common/EncodingInfo.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "runtime/src/iree/builtins/ukernel/exported_bits.h"

namespace mlir {
namespace iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

/// For `dispatchTensorType` that bind a `RankedTensorType` with encoding,
/// returns the materialized shape of the `dispatchTensorType`. The
/// dynamic dimensions of the `dispatchTensorType` are provided in
/// `dynamicDims`.
static FailureOr<SmallVector<OpFoldResult>> getPackedDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    MaterializeEncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims,
    MaterializeEncodingValueFn materializeEncodingValueFn) {
  auto boundTensorType =
      llvm::dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
  if (!boundTensorType) {
    return failure();
  }

  MaterializeEncodingFn materializeEncodingFn =
      typeConverter.getMaterializeEncodingFn();
  FailureOr<MaterializeEncodingInfo> encodingInfo =
      materializeEncodingFn(boundTensorType);
  if (failed(encodingInfo)) {
    return failure();
  }

  SmallVector<OpFoldResult, 4> targetShape =
      getMixedValues(dispatchTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes = getInnerTileSizesOfr(
      builder, loc, boundTensorType, *encodingInfo, materializeEncodingValueFn);
  if (failed(innerTileSizes)) return failure();
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
    MaterializeEncodingTypeConverter &typeConverter,
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

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
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

    auto *typeConverter =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter());
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

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the load covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial loads.
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    auto boundTensorType = sourceType.getBoundType();
    auto *typeConverter =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter());
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

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle operations where the store covers the entire
    // `!flow.dispatch.tensor` type.
    // TODO(ravishankarm): Relax this for partial stores.
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }

    auto targetType = storeOp.getTargetType();
    auto boundTensorType = targetType.getBoundType();
    auto *typeConverter =
        static_cast<MaterializeEncodingTypeConverter *>(getTypeConverter());
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

}  // namespace

IREE::LinalgExt::MaterializeEncodingInfo chooseEncodingInfoForMatmul(
    MatmulType type, MatmulOperandRole operandRole,
    MatmulTileParams tileParams) {
  MaterializeEncodingInfo encodingInfo;
  encodingInfo.innerDimsPos = {0, 1};
  switch (operandRole) {
    case (MatmulOperandRole::LHS): {
      encodingInfo.innerTileSizes = {tileParams.M, tileParams.K};
      break;
    }
    case (MatmulOperandRole::RHS): {
      encodingInfo.innerTileSizes = {tileParams.N, tileParams.K};
      encodingInfo.innerDimsPos = {1, 0};
      encodingInfo.outerDimsPerm = {1, 0};
      break;
    }
    case (MatmulOperandRole::RESULT): {
      encodingInfo.innerTileSizes = {tileParams.M, tileParams.N};
      break;
    }
    default: {
      assert(false);
      return {};
    }
  }
  return encodingInfo;
}

std::optional<TensorEncoding> getEncoding(RankedTensorType tensorType) {
  auto encodingAttr =
      llvm::dyn_cast_if_present<EncodingAttr>(tensorType.getEncoding());
  if (!encodingAttr) return std::nullopt;
  return encodingAttr.getEncoding().getValue();
}

std::optional<MatmulType> getMatmulType(TensorEncoding encoding) {
  switch (encoding) {
    case TensorEncoding::MATMUL_F32F32F32_LHS:
    case TensorEncoding::MATMUL_F32F32F32_RHS:
    case TensorEncoding::MATMUL_F32F32F32_RESULT:
      return MatmulType::F32F32F32;
    case TensorEncoding::MATMUL_I8I8I32_LHS:
    case TensorEncoding::MATMUL_I8I8I32_RHS:
    case TensorEncoding::MATMUL_I8I8I32_RESULT:
      return MatmulType::I8I8I32;
    default:
      return std::nullopt;
  }
}

std::optional<MatmulOperandRole> getMatmulOperandRole(TensorEncoding encoding) {
  switch (encoding) {
    case TensorEncoding::MATMUL_F32F32F32_LHS:
    case TensorEncoding::MATMUL_I8I8I32_LHS:
      return MatmulOperandRole::LHS;
    case TensorEncoding::MATMUL_F32F32F32_RHS:
    case TensorEncoding::MATMUL_I8I8I32_RHS:
      return MatmulOperandRole::RHS;
    case TensorEncoding::MATMUL_F32F32F32_RESULT:
    case TensorEncoding::MATMUL_I8I8I32_RESULT:
      return MatmulOperandRole::RESULT;
    default:
      return std::nullopt;
  }
}

void adjustTileSizesToNarrowStaticShape(MaterializeEncodingInfo &encodingInfo,
                                        ArrayRef<int64_t> shape) {
  for (size_t i = 0; i < shape.size(); i++) {
    int64_t size = shape[encodingInfo.innerDimsPos[i]];
    // Dynamic sizes are assumed to be large enough, not to be candidates for
    // narrow kernels.
    if (ShapedType::isDynamic(size)) continue;
    int64_t &tileSize = encodingInfo.innerTileSizes[i];
    // Let's not try to handle any dynamic tile sizes here. We could handle the
    // case where size==1 (as whatever is the runtime value of tileSize, it
    // can't be less than that, so it should be OK to replace it with 1) but
    // in general, adjusting dynamic tile sizes has to be done by the
    // materializeEncodingValueFn which we obtain those tileSizes from.
    if (ShapedType::isDynamic(tileSize)) continue;
    auto generateNarrowTileSize = [&](int64_t n) {
      if (size <= n && tileSize >= n) tileSize = n;
    };
    generateNarrowTileSize(1);
    generateNarrowTileSize(2);
    generateNarrowTileSize(4);
  }
}

FailureOr<MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc) {
  std::optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding) return failure();
  auto matmulType = getMatmulType(*encoding);
  auto matmulOperandRole = getMatmulOperandRole(*encoding);
  if (!matmulType || !matmulOperandRole) return failure();
  uint32_t flags = 0;
  if (*matmulType == MatmulType::F32F32F32) {
    flags |= IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32;
  } else if (*matmulType == MatmulType::I8I8I32) {
    flags |= IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
  } else {
    return failure();
  }
  if (*matmulOperandRole == MatmulOperandRole::LHS) {
    flags |= IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS;
  } else if (*matmulOperandRole == MatmulOperandRole::RHS) {
    flags |= IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS;
  } else if (*matmulOperandRole == MatmulOperandRole::RESULT) {
    flags |= IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT;
  } else {
    return failure();
  }
  SmallVector<Type> tileSizesTypes(tensorType.getRank(),
                                   builder.getIndexType());
  SmallVector<Value> shapeValues;
  for (int64_t i : tensorType.getShape()) {
    shapeValues.push_back(builder.create<arith::ConstantIndexOp>(loc, i));
  }
  auto op = builder.create<IREE::VMVX::QueryTileSizesOp>(
      loc, tileSizesTypes, shapeValues, builder.getI32IntegerAttr(flags));
  MaterializeEncodingValueInfo result;
  result.innerTileSizes = op.getTileSizes();
  return result;
}

MaterializeEncodingValueFn getMaterializeEncodingValueFn(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (isVMVXBackend(targetAttr) && hasMicrokernels(targetAttr)) {
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
        if (!resultType) return true;
        return resultType == typeConverter.convertType(resultType);
      });

  populateMaterializeEncodingPatterns(patterns, target, typeConverter,
                                      materializeEncodingValueFn);
  patterns.insert<MaterializeFlowDispatchTensorLoadOp,
                  MaterializeFlowDispatchTensorStoreOp,
                  MaterializeInterfaceBindingEncoding>(
      context, typeConverter, materializeEncodingValueFn);
}

}  // namespace iree_compiler
}  // namespace mlir
