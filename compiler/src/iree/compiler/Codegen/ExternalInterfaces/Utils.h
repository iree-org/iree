// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/Utils/EncodingUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler::IREE {

static const char kEncodingInfoAttrName[] = "encoding_info";

// This class is the base class for the external model of different packed
// encoding layout attributes. It provides a public method, `getEncodingInfo` to
// reduce the duplicated implementations before. To inherit it, it requires the
// derived class to implement the `getConfiguration` method and the
// `getEncodingInfoImpl` method.
template <typename EncodingPackedLayoutMaterializerAttr,
          typename EncodingLayoutAttr>
struct PackedLayoutMaterializerAttrExternalModelBase
    : public IREE::Codegen::PackedLayoutMaterializerAttr::ExternalModel<
          EncodingPackedLayoutMaterializerAttr, EncodingLayoutAttr> {
public:
  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(Attribute attr, RankedTensorType type) const {
    const EncodingPackedLayoutMaterializerAttr *impl =
        static_cast<const EncodingPackedLayoutMaterializerAttr *>(this);
    // If the layout is already resolved, use it directly.
    if (auto config = impl->getConfiguration(attr)) {
      if (auto namedAttr = config.getNamed(kEncodingInfoAttrName)) {
        std::optional<IREE::Codegen::MaterializeEncodingInfo> info =
            IREE::Codegen::deserializeEncodingInfo(
                cast<DictionaryAttr>(namedAttr->getValue()));
        assert(info && "encoding_info is invalid");
        return info.value();
      }
    }
    return impl->getEncodingInfoImpl(attr, type);
  }
};

template <typename EncodingLayoutMaterializerAttr, typename EncodingLayoutAttr>
struct EncodingLayoutMaterializerAttrExternalModelBase
    : public IREE::Encoding::LayoutMaterializerAttr::ExternalModel<
          EncodingLayoutMaterializerAttr, EncodingLayoutAttr> {
public:
  IREE::Codegen::MaterializeEncodingInfo
  getEncodingInfo(EncodingLayoutAttr layoutAttr, RankedTensorType type) const {
    return getEncodingInfoFromLayout(
        type, cast<IREE::Encoding::LayoutMaterializerAttr>(layoutAttr));
  }

  Type convertType(Attribute attr, Type type) const {
    EncodingLayoutAttr layoutAttr = cast<EncodingLayoutAttr>(attr);
    return TypeSwitch<Type, Type>(type)
        .template Case<RankedTensorType>([&](auto type) {
          // For a given tensor type with an encoding, return the materialized
          // type to use for it. If no encoding is set, then return the tensor
          // type itself.
          IREE::Codegen::MaterializeEncodingInfo encodingInfo =
              getEncodingInfo(layoutAttr, type);
          if (IREE::Codegen::isIdentityLayout(encodingInfo)) {
            return type.dropEncoding();
          }
          auto packedType =
              cast<RankedTensorType>(linalg::PackOp::inferPackedType(
                  type, encodingInfo.innerTileSizes, encodingInfo.innerDimsPos,
                  encodingInfo.outerDimsPerm));

          // There is no swizzle, we are already done. Typically the case on
          // CPU.
          if (!encodingInfo.swizzle) {
            return packedType;
          }

          // There is a swizzle, we need to handle it. Typically the case on
          // GPU.
          auto swizzle = *encodingInfo.swizzle;
          SmallVector<int64_t> newShape(packedType.getShape().drop_back(
              encodingInfo.innerTileSizes.size()));
          SmallVector<int64_t> swizzledTileShape =
              IREE::Codegen::getExpandedTileShape(swizzle.expandShape);
          applyPermutationToVector(swizzledTileShape, swizzle.permutation);
          newShape.append(swizzledTileShape);
          return RankedTensorType::get(newShape, packedType.getElementType());
        })
        .template Case<IREE::TensorExt::DispatchTensorType>(
            [&](auto dispatchTensorType) {
              Type boundType = dispatchTensorType.getBoundType();
              Type convertedBoundType = convertType(attr, boundType);
              if (convertedBoundType == boundType) {
                return dispatchTensorType;
              }
              return IREE::TensorExt::DispatchTensorType::get(
                  dispatchTensorType.getAccess(), convertedBoundType);
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
    auto layoutAttr = cast<IREE::Encoding::LayoutMaterializerAttr>(attr);
    // Only handle cases where the slice spans the whole
    // `!iree_tensor_ext.dispatch.tensor` type.
    // TODO(jornt): Enable partial slices.
    if (!type.doesSliceSpanWholeTensor(dynamicDims, offsets, sizes, strides)) {
      return failure();
    }
    auto boundTensorType = cast<RankedTensorType>(type.getBoundType());
    IREE::Codegen::MaterializeEncodingInfo encodingInfo =
        getEncodingInfoFromLayout(boundTensorType, layoutAttr);
    newSizes = getMixedValues(boundTensorType.getShape(), dynamicDims, builder);
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes =
        getPackedDimsForDispatchTensorImpl(builder, loc, type, dynamicDims,
                                           layoutAttr, encodingInfo);
    if (succeeded(convertedMixedSizes)) {
      newSizes = convertedMixedSizes.value();
    }
    newOffsets.resize(newSizes.size(), builder.getIndexAttr(0));
    newStrides.resize(newSizes.size(), builder.getIndexAttr(1));
    return success();
  }
};

/// Calculates the storage size in bytes for the given `type` with a packed
/// layout encoding `attr`. Requirement: `attr` must implement
/// IREE::Codegen::PackedLayoutMaterializerAttr.
Value calculatePackedStorageSizeInBytesImpl(Attribute attr, Location loc,
                                            OpBuilder &builder,
                                            RankedTensorType type,
                                            ValueRange dynamicDims);

/// Returns a dictionary attribute that contains the materialized encoding info,
/// i.e., serialized MaterializeEncodingInfo struct. The EncodingAttr attribute
/// is attached to the dictionary, if it is present in `type` and
/// `addEncodingAttr` is true.
/// TODO(hanchung): only attach needed information to the configuration. The
/// `addEncodingAttr` is mainly for VMVX ukernel path because the ukernel ops
/// lowering requires all the information. There are no direct mappings from
/// layouts to ukernels.
/// Requirement: `attr` must implement
/// IREE::Codegen::PackedLayoutMaterializerAttr.
DictionaryAttr getPackedLayoutImpl(Attribute attr, RankedTensorType type,
                                   bool addEncodingAttr = false);

/// Appends the NamedAttribute into `config` if there is a `name` NamedAttribute
/// in the `dictAttr`.
void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name);

} // namespace mlir::iree_compiler::IREE

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILSS_H_
