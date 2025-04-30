// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EncodingUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <optional>

#define DEBUG_TYPE "iree-codegen-encoding-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Encoding::EncodingAttr;
using IREE::Encoding::getEncodingAttr;
using IREE::Encoding::getEncodingContractionDims;
using IREE::Encoding::PadEncodingLayoutAttr;
using IREE::Encoding::SerializableEncodingAttrInterface;

// Returns the layout as a SerializableEncodingAttrInterface, or nullptr if this
// is not the only layout or if there's no encoding at all.
static SerializableEncodingAttrInterface
getSerializableEncodingAttr(IREE::Codegen::LayoutAttrInterface layoutAttr,
                            RankedTensorType type) {
  if (!type.getEncoding()) {
    return nullptr;
  }
  auto encoding = dyn_cast<IREE::Encoding::LayoutAttr>(type.getEncoding());
  if (encoding) {
    ArrayAttr layouts = encoding.getLayouts();
    if (layouts.size() != 1) {
      return nullptr;
    }
    return dyn_cast<SerializableEncodingAttrInterface>(*layouts.begin());
  }
  auto encodingResolver =
      dyn_cast<IREE::Encoding::EncodingLayoutResolverAttrInterface>(layoutAttr);
  if (!encodingResolver) {
    return nullptr;
  }
  Attribute resolvedEncoding = encodingResolver.getLayout(type);
  LDBG("Unresolved type: " << type);
  LDBG("layoutAttr: " << layoutAttr);
  LDBG("Resolved into: " << resolvedEncoding);
  return dyn_cast<SerializableEncodingAttrInterface>(resolvedEncoding);
}

MaterializeEncodingTypeConverter::MaterializeEncodingTypeConverter(
    IREE::Codegen::LayoutAttrInterface layoutAttr)
    : layoutAttr(layoutAttr) {
  addConversion([](IntegerType intType) { return intType; });
  addConversion([](IndexType indexType) { return indexType; });
  addConversion([](FloatType floatType) { return floatType; });
  addConversion([](MemRefType memrefType) { return memrefType; });
  addConversion([=](RankedTensorType type) {
    SerializableEncodingAttrInterface serializableEncodingAttr =
        getSerializableEncodingAttr(getLayoutAttr(), type);
    // TODO(jornt): The isa<IREE::Encoding::PadEncodingLayoutAttr> check is
    // needed because PadEncodingLayoutAttr is a serializable attribute, but it
    // relies on its own type conversion for now. Once PadEncodingLayoutAttr
    // implements `convertType`, this can be removed.
    if (serializableEncodingAttr &&
        !isa<IREE::Encoding::PadEncodingLayoutAttr>(serializableEncodingAttr)) {
      return cast<RankedTensorType>(serializableEncodingAttr.convertType(type));
    }
    return type.dropEncoding();
  });
  addConversion([&](IREE::TensorExt::DispatchTensorType dispatchTensorType) {
    auto boundType =
        dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
    if (!boundType || !boundType.getEncoding()) {
      return dispatchTensorType;
    }
    SerializableEncodingAttrInterface serializableEncodingAttr =
        getSerializableEncodingAttr(getLayoutAttr(), boundType);
    // TODO(jornt): The isa<IREE::Encoding::PadEncodingLayoutAttr> check is
    // needed because PadEncodingLayoutAttr is a serializable attribute, but it
    // relies on its own type conversion for now. Once PadEncodingLayoutAttr
    // implements `convertType`, this can be removed.
    if (serializableEncodingAttr &&
        !isa<IREE::Encoding::PadEncodingLayoutAttr>(serializableEncodingAttr)) {
      return cast<IREE::TensorExt::DispatchTensorType>(
          serializableEncodingAttr.convertType(dispatchTensorType));
    }
    Type convertedBoundType = convertType(boundType);
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
