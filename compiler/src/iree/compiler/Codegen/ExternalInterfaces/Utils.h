// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_
#define IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::iree_compiler::IREE {

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

/// Calculates the storage size in bytes for the given `type` with a layout
/// encoding `attr`.
/// Requirement: `attr` must implement IREE::Codegen::LayoutAttrInterface.
Value calculateStorageSizeInBytesImpl(Attribute attr, Location loc,
                                      OpBuilder &builder, RankedTensorType type,
                                      ValueRange dynamicDims);

/// Returns a dictionary attribute that contains the materialized encoding info,
/// i.e., serialized MaterializeEncodingInfo struct.
/// Requirement: `attr` must implement IREE::Codegen::LayoutAttrInterface.
DictionaryAttr getLayoutImpl(Attribute attr, RankedTensorType type);

/// Appends the NamedAttribute into `config` if there is a `name` NamedAttribute
/// in the `dictAttr`.
void storeNamedAttrIfPresent(SmallVectorImpl<NamedAttribute> &config,
                             DictionaryAttr dictAttr, StringRef name);

} // namespace mlir::iree_compiler::IREE

#endif // IREE_COMPILER_CODEGEN_EXTERNALINTERFACES_UTILSS_H_
