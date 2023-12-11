// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_
#define IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_

#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h.inc" // IWYU pragma: export

namespace mlir::iree_compiler::IREE::Vulkan {

namespace detail {
struct TargetEnvAttributeStorage;
} // namespace detail

/// An attribute that specifies the target version, supported extensions, and
/// resource limits. These information describles a Vulkan target environment.
class TargetEnvAttr
    : public Attribute::AttrBase<TargetEnvAttr, Attribute,
                                 detail::TargetEnvAttributeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "vk.target_env";

  /// Gets a TargetEnvAttr instance.
  // TODO(antiagainst): support other physical device core properties, physical
  // device core features and per-extension features.
  static TargetEnvAttr get(Version version, uint32_t revision,
                           ArrayRef<Extension> extensions,
                           spirv::Vendor vendorID, spirv::DeviceType deviceType,
                           uint32_t deviceID, CapabilitiesAttr capabilities);
  static TargetEnvAttr get(IntegerAttr version, IntegerAttr revision,
                           ArrayAttr extensions, spirv::Vendor vendorID,
                           spirv::DeviceType deviceType, uint32_t deviceID,
                           CapabilitiesAttr capabilities);

  /// Returns the attribute kind's name (without the 'vk.' prefix).
  static StringRef getKindName();

  /// Returns the target Vulkan version; e.g., for 1.1.120, it should be V_1_1.
  Version getVersion();

  /// Returns the target Vulkan revision; e.g., for 1.1.120, it should be 120.
  unsigned getRevision();

  struct ext_iterator final
      : public llvm::mapped_iterator<ArrayAttr::iterator,
                                     Extension (*)(Attribute)> {
    explicit ext_iterator(ArrayAttr::iterator it);
  };
  using ext_range = llvm::iterator_range<ext_iterator>;

  /// Returns the target Vulkan instance and device extensions.
  ext_range getExtensions();
  /// Returns the target Vulkan instance and device extensions as an string
  /// array attribute.
  ArrayAttr getExtensionsAttr();

  /// Returns the vendor ID.
  spirv::Vendor getVendorID();

  /// Returns the device type.
  spirv::DeviceType getDeviceType();

  /// Returns the device ID.
  uint32_t getDeviceID();

  /// Returns the dictionary attribute containing various Vulkan capabilities
  /// bits.
  CapabilitiesAttr getCapabilitiesAttr();

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              IntegerAttr version, IntegerAttr revision,
                              ArrayAttr extensions, spirv::Vendor vendorID,
                              spirv::DeviceType deviceType, uint32_t deviceID,
                              CapabilitiesAttr capabilities);
};

} // namespace mlir::iree_compiler::IREE::Vulkan

#endif // IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_
