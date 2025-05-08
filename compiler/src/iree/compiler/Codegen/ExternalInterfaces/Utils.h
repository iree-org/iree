// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler::IREE {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::TensorExt::DispatchTensorType;

static const char kEncodingInfoAttrName[] = "encoding_info";

// This class is the base class for the external model of different encoding
// resolver attributes. It provides a public method, `getEncodingInfo` to reduce
// the duplicated implementations before. To inherit it, it requires the derived
// class to implement the `getConfiguration` method and the
// `getEncodingInfoImpl` method.
template <typename DeviceEncodingLayoutResolverAttrInterface,
          typename EncodingLayoutAttr>
struct DeviceEncodingLayoutResolverExternalModelBase
    : public Codegen::LayoutAttrInterface::ExternalModel<
          DeviceEncodingLayoutResolverAttrInterface, EncodingLayoutAttr> {
public:
  Codegen::MaterializeEncodingInfo
  getEncodingInfo(Attribute attr, RankedTensorType type) const {
    const DeviceEncodingLayoutResolverAttrInterface *impl =
        static_cast<const DeviceEncodingLayoutResolverAttrInterface *>(this);
    // If the layout is already resolved, use it directly.
    if (auto config = impl->getConfiguration(attr)) {
      if (auto namedAttr = config.getNamed(kEncodingInfoAttrName)) {
        std::optional<Codegen::MaterializeEncodingInfo> info =
            Codegen::deserializeEncodingInfo(
                cast<DictionaryAttr>(namedAttr->getValue()));
        assert(info && "encoding_info is invalid");
        return info.value();
      }
    }
    return impl->getEncodingInfoImpl(attr, type);
  }
};

static std::optional<MaterializeEncodingInfo>
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

template <typename DeviceSerializableEncodingAttrInterface,
          typename EncodingLayoutAttr>
struct HostSerializableEncodingAttrInterfaceExternalModelBase
    : public IREE::Encoding::SerializableEncodingAttrInterface::ExternalModel<
          DeviceSerializableEncodingAttrInterface, EncodingLayoutAttr> {
public:
  MaterializeEncodingInfo getEncodingInfo(EncodingLayoutAttr layoutAttr,
                                          RankedTensorType type) const {
    // If the layout is present in the encoding, use it directly. It means that
    // the layout is already resolved and some information could be dropped
    // during the lowering. Thus, we prioritize the resolved layout.
    if (std::optional<MaterializeEncodingInfo> maybeEncodingInfo =
            getEncodingInfoFromLayouts(type)) {
      return maybeEncodingInfo.value();
    }
    return cast<IREE::Codegen::LayoutAttrInterface>(layoutAttr)
        .getEncodingInfo(type);
  }

  Type convertType(Attribute attr, Type type) const {
    EncodingLayoutAttr layoutAttr = cast<EncodingLayoutAttr>(attr);
    return TypeSwitch<Type, Type>(type)
        .template Case<RankedTensorType>([&](auto type) {
          // For a given tensor type with an encoding, return the materialized
          // type to use for it. If no encoding is set, then return the tensor
          // type itself.
          MaterializeEncodingInfo encodingInfo =
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
        .template Case<DispatchTensorType>([&](auto dispatchTensorType) {
          Type boundType = dispatchTensorType.getBoundType();
          Type convertedBoundType = convertType(attr, boundType);
          if (convertedBoundType == boundType) {
            return dispatchTensorType;
          }
          return DispatchTensorType::get(dispatchTensorType.getAccess(),
                                         convertedBoundType);
        })
        .Default([&](auto concreteType) { return concreteType; });
  }
};

/// Calculates the storage size in bytes for the given `type` with a layout
/// encoding `attr`.
/// Requirement: `attr` must implement IREE::Codegen::LayoutAttrInterface.
Value calculateStorageSizeInBytesImpl(Attribute attr, Location loc,
                                      OpBuilder &builder, RankedTensorType type,
                                      ValueRange dynamicDims);

/// Returns a dictionary attribute that contains the materialized encoding info,
/// i.e., serialized MaterializeEncodingInfo struct. The EncodingAttr attribute
/// is attached to the dictionary, if it is present in `type` and
/// `addEncodingAttr` is true.
/// TODO(hanchung): only attach needed information to the configuration. The
/// `addEncodingAttr` is mainly for VMVX ukernel path because the ukernel ops
/// lowering requires all the information. There are no direct mappings from
/// layouts to ukernels.
/// Requirement: `attr` must implement IREE::Codegen::LayoutAttrInterface.
DictionaryAttr getLayoutImpl(Attribute attr, RankedTensorType type,
                             bool addEncodingAttr = false);

/// Appends the NamedAttribute into `config` if there is a `name` NamedAttribute
/// in the `dictAttr`.
void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name);

} // namespace mlir::iree_compiler::IREE

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILSS_H_
