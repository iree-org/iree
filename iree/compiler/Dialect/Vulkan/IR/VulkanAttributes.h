// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_
#define IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_

#include "iree/compiler/Dialect/IREE/IR/IREEAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h.inc"

namespace Vulkan {

namespace detail {
struct TargetEnvAttributeStorage;
}  // namespace detail

/// An attribute that specifies the target version, supported extensions, and
/// resource limits. These information describles a Vulkan target environment.
class TargetEnvAttr
    : public Attribute::AttrBase<TargetEnvAttr, Attribute,
                                 detail::TargetEnvAttributeStorage> {
 public:
  using Base::Base;

  /// Gets a TargetEnvAttr instance.
  // TODO(antiagainst): support other physical device core properties, physical
  // device core features and per-extension features.
  static TargetEnvAttr get(IntegerAttr version, IntegerAttr revision,
                           ArrayAttr extensions, DictionaryAttr capabilities);

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

  /// Returns the dictionary attribute containing various Vulkan capabilities
  /// bits.
  CapabilitiesAttr getCapabilitiesAttr();

  static LogicalResult verifyConstructionInvariants(
      Location loc, IntegerAttr version, IntegerAttr revision,
      ArrayAttr extensions, DictionaryAttr capabilities);
};

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VULKAN_IR_VULKANATTRIBUTES_H_
