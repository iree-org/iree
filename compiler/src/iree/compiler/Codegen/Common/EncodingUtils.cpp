// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-encoding-utils"

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Encoding::PaddingAttr;

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    IREE::Encoding::LayoutMaterializerAttr layoutAttr)
    : layoutAttr(layoutAttr) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) {
    return cast<RankedTensorType>(getLayoutAttr().convertType(type));
  });
  addConversion([&](IREE::TensorExt::DispatchTensorType dispatchTensorType) {
    return cast<IREE::TensorExt::DispatchTensorType>(
        getLayoutAttr().convertType(dispatchTensorType));
  });
}

MaterializeEncodingConversionTarget::MaterializeEncodingConversionTarget(
    MLIRContext &context)
    : ConversionTarget(context) {
  // Mark any operation that has operands/results with encoding as
  // illegal.
  markUnknownOpDynamicallyLegal([](Operation *op) {
    auto typeHasDataTilingEncoding = [](Type t) -> bool {
      auto tensorType = dyn_cast<RankedTensorType>(t);
      if (!tensorType || !tensorType.getEncoding()) {
        return false;
      }
      return isa<IREE::Encoding::ContractionEncodingAttrInterface,
                 IREE::Encoding::LayoutAttr>(tensorType.getEncoding());
    };
    auto valueHasDataTilingEncoding = [=](Value v) -> bool {
      return typeHasDataTilingEncoding(v.getType());
    };
    bool hasOperandOrResultsWithEncoding =
        llvm::any_of(op->getOperands(), valueHasDataTilingEncoding) ||
        llvm::any_of(op->getResultTypes(), typeHasDataTilingEncoding);
    return !hasOperandOrResultsWithEncoding;
  });
}

IREE::Codegen::MaterializeEncodingInfo
MaterializeEncodingTypeConverter::getEncodingInfo(RankedTensorType type) const {
  return getEncodingInfoFromLayout(type, layoutAttr);
}

FailureOr<SmallVector<OpFoldResult>>
MaterializeEncodingTypeConverter::getInnerTileSizesOfr(
    OpBuilder &rewriter, Location loc, RankedTensorType tensorType,
    const IREE::Codegen::MaterializeEncodingInfo &materializeEncodingInfo)
    const {
  return getInnerTileSizesOfrImpl(rewriter, loc, tensorType, layoutAttr,
                                  materializeEncodingInfo);
}

FailureOr<SmallVector<OpFoldResult>>
MaterializeEncodingTypeConverter::getPackedDimsForDispatchTensor(
    OpBuilder &builder, Location loc,
    IREE::TensorExt::DispatchTensorType dispatchTensorType,
    ValueRange dynamicDims) const {

  auto boundTensorType =
      dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
  if (!boundTensorType) {
    return failure();
  }
  MaterializeEncodingInfo encodingInfo =
      getEncodingInfoFromLayout(boundTensorType, layoutAttr);
  return getPackedDimsForDispatchTensorImpl(
      builder, loc, dispatchTensorType, dynamicDims, layoutAttr, encodingInfo);
}

LogicalResult MaterializeEncodingTypeConverter::getOffsetsSizesStrides(
    OpBuilder &builder, Location loc, IREE::TensorExt::DispatchTensorType type,
    ValueRange dynamicDims, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
    SmallVectorImpl<OpFoldResult> &newOffsets,
    SmallVectorImpl<OpFoldResult> &newSizes,
    SmallVectorImpl<OpFoldResult> &newStrides) const {
  return getLayoutAttr().getOffsetsSizesStrides(
      builder, loc, type, dynamicDims, offsets, sizes, strides, newOffsets,
      newSizes, newStrides);
}

} // namespace mlir::iree_compiler
