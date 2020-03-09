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

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.cpp.inc"

namespace Vulkan {

namespace detail {
struct TargetEnvAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Attribute, Attribute, Attribute, Attribute>;

  TargetEnvAttributeStorage(Attribute version, Attribute revision,
                            Attribute extensions, Attribute capabilities)
      : version(version),
        revision(revision),
        extensions(extensions),
        capabilities(capabilities) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == version && std::get<1>(key) == revision &&
           std::get<2>(key) == extensions && std::get<3>(key) == capabilities;
  }

  static TargetEnvAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetEnvAttributeStorage>())
        TargetEnvAttributeStorage(std::get<0>(key), std::get<1>(key),
                                  std::get<2>(key), std::get<3>(key));
  }

  Attribute version;
  Attribute revision;
  Attribute extensions;
  Attribute capabilities;
};
}  // namespace detail

TargetEnvAttr TargetEnvAttr::get(IntegerAttr version, IntegerAttr revision,
                                 ArrayAttr extensions,
                                 DictionaryAttr capabilities) {
  assert(version && revision && extensions && capabilities);
  MLIRContext *context = version.getContext();
  return Base::get(context, AttrKind::TargetEnv, version, revision, extensions,
                   capabilities);
}

StringRef TargetEnvAttr::getKindName() { return "target_env"; }

Version TargetEnvAttr::getVersion() {
  return static_cast<Version>(
      getImpl()->version.cast<IntegerAttr>().getValue().getZExtValue());
}

unsigned TargetEnvAttr::getRevision() {
  return getImpl()->revision.cast<IntegerAttr>().getValue().getZExtValue();
}

TargetEnvAttr::ext_iterator::ext_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator, Extension (*)(Attribute)>(
          it, [](Attribute attr) {
            return *symbolizeExtension(attr.cast<StringAttr>().getValue());
          }) {}

TargetEnvAttr::ext_range TargetEnvAttr::getExtensions() {
  auto range = getExtensionsAttr().getValue();
  return {ext_iterator(range.begin()), ext_iterator(range.end())};
}

ArrayAttr TargetEnvAttr::getExtensionsAttr() {
  return getImpl()->extensions.cast<ArrayAttr>();
}

CapabilitiesAttr TargetEnvAttr::getCapabilitiesAttr() {
  return getImpl()->capabilities.cast<CapabilitiesAttr>();
}

LogicalResult TargetEnvAttr::verifyConstructionInvariants(
    Location loc, IntegerAttr version, IntegerAttr revision,
    ArrayAttr extensions, DictionaryAttr capabilities) {
  if (!version.getType().isInteger(32))
    return emitError(loc) << "expected 32-bit integer for version";

  if (!revision.getType().isInteger(32))
    return emitError(loc) << "expected 32-bit integer for revision";

  if (!llvm::all_of(extensions.getValue(), [](Attribute attr) {
        if (auto strAttr = attr.dyn_cast<StringAttr>())
          if (symbolizeExtension(strAttr.getValue())) return true;
        return false;
      }))
    return emitError(loc) << "unknown extension in extension list";

  if (!capabilities.isa<CapabilitiesAttr>())
    return emitError(loc)
           << "expected vulkan::CapabilitiesAttr for capabilities";

  return success();
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
