// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    IREE::Codegen::LayoutAttrInterface layoutAttr)
    : layoutAttr(layoutAttr) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) -> RankedTensorType {
    // For a given tensor type with an encoding, return the materialized
    // type to use for it. If no encoding is set, then return the tensor type
    // itself.
    MaterializeEncodingInfo encodingInfo = getEncodingInfo(type);
    if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
      return type.dropEncoding();
    }
    auto packedType = cast<RankedTensorType>(linalg::PackOp::inferPackedType(
        type, encodingInfo.innerTileSizes, encodingInfo.innerDimsPos,
        encodingInfo.outerDimsPerm));

    // There is no swizzle, we are already done. Typically the case on CPU.
    if (!encodingInfo.swizzle) {
      return packedType;
    }

    // There is a swizzle, we need to handle it. Typically the case on GPU.
    auto swizzle = *encodingInfo.swizzle;
    SmallVector<int64_t> newShape(
        packedType.getShape().drop_back(encodingInfo.innerTileSizes.size()));
    SmallVector<int64_t> swizzledTileShape =
        IREE::Codegen::getExpandedTileShape(swizzle.expandShape);
    applyPermutationToVector(swizzledTileShape, swizzle.permutation);
    newShape.append(swizzledTileShape);
    return RankedTensorType::get(newShape, packedType.getElementType());
  });
  addConversion([&](IREE::TensorExt::DispatchTensorType dispatchTensorType)
                    -> IREE::TensorExt::DispatchTensorType {
    Type boundType = dispatchTensorType.getBoundType();
    Type convertedBoundType = convertType(boundType);
    if (convertedBoundType == boundType) {
      return dispatchTensorType;
    }
    return IREE::TensorExt::DispatchTensorType::get(
        dispatchTensorType.getAccess(), convertedBoundType);
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
  // If the layout is present in the encoding, use it directly. It means that
  // the layout is already resolved and some information could be dropped during
  // the lowering. Thus, we prioritize the resolved layout.
  if (auto maybeEncodingInfo = getEncodingInfoFromLayouts(type)) {
    return maybeEncodingInfo.value();
  }
  return layoutAttr.getEncodingInfo(type);
}

std::optional<IREE::Codegen::MaterializeEncodingInfo>
getEncodingInfoFromLayouts(RankedTensorType type) {
  auto layoutAttr =
      dyn_cast_or_null<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (!layoutAttr) {
    return std::nullopt;
  }
  ArrayRef<Attribute> layouts = layoutAttr.getLayouts().getValue();
  assert(layouts.size() == 1 && "only single layout is supported");
  if (auto layout = dyn_cast<IREE::Codegen::LayoutAttrInterface>(layouts[0])) {
    return layout.getEncodingInfo(type);
  }
  return std::nullopt;
}

} // namespace mlir::iree_compiler
