// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/CodegenExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"

namespace mlir::iree_compiler::IREE::Codegen {

using IREE::TensorExt::DispatchTensorType;

struct EncodingNopLayoutMaterializerAttr final
    : IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          EncodingNopLayoutMaterializerAttr, EncodingNopLayoutAttr> {
  Type convertType(Attribute attr, Type type) const {
    return TypeSwitch<Type, Type>(type)
        .Case<RankedTensorType>([&](auto rankedTensorType) {
          return rankedTensorType.dropEncoding();
        })
        .Case<DispatchTensorType>([&](auto dispatchTensorType) {
          auto boundType =
              dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
          if (!boundType || !boundType.getEncoding()) {
            return dispatchTensorType;
          }
          Type convertedBoundType = convertType(attr, boundType);
          return DispatchTensorType::get(dispatchTensorType.getAccess(),
                                         convertedBoundType);
        })
        .Default([&](auto concreteType) { return concreteType; });
  }

  LogicalResult getOffsetsSizesStrides(
      Attribute attr, OpBuilder &builder, Location loc,
      IREE::TensorExt::DispatchTensorType type, ValueRange dynamicDims,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      ArrayRef<OpFoldResult> strides, SmallVectorImpl<OpFoldResult> &newOffsets,
      SmallVectorImpl<OpFoldResult> &newSizes,
      SmallVectorImpl<OpFoldResult> &newStrides) const {
    // Only handle cases where the slice spans the whole
    // `!iree_tensor_ext.dispatch.tensor` type.
    // TODO(jornt): Enable partial slices.
    if (!type.doesSliceSpanWholeTensor(dynamicDims, offsets, sizes, strides)) {
      return failure();
    }
    auto boundTensorType = cast<RankedTensorType>(type.getBoundType());
    newSizes = getMixedValues(boundTensorType.getShape(), dynamicDims, builder);
    newOffsets.resize(newSizes.size(), builder.getIndexAttr(0));
    newStrides.resize(newSizes.size(), builder.getIndexAttr(1));
    return success();
  }

  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    return clone(b, op, convertedResTypes, convertedOperands);
  }
};

struct EncodingNopLayoutResolverAttr final
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<
          EncodingNopLayoutResolverAttr, EncodingNopLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    return attr;
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    return attr;
  }
};

void registerCodegenExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::Codegen::IREECodegenDialect *dialect) {
    EncodingNopLayoutAttr::attachInterface<EncodingNopLayoutResolverAttr,
                                           EncodingNopLayoutMaterializerAttr>(
        *ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::Codegen
