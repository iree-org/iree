// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"

#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.cpp.inc"  // IWYU pragma: keep

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

//===----------------------------------------------------------------------===//
// TargetEnv
//===----------------------------------------------------------------------===//

namespace detail {
struct TargetEnvAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Attribute, Attribute, Attribute, spirv::Vendor,
                           spirv::DeviceType, uint32_t, Attribute>;

  TargetEnvAttributeStorage(Attribute version, Attribute revision,
                            Attribute extensions, spirv::Vendor vendorID,
                            spirv::DeviceType deviceType, uint32_t deviceID,
                            Attribute capabilities)
      : version(version),
        revision(revision),
        extensions(extensions),
        capabilities(capabilities),
        vendorID(vendorID),
        deviceType(deviceType),
        deviceID(deviceID) {}

  bool operator==(const KeyTy &key) const {
    return key == std::make_tuple(version, revision, extensions, vendorID,
                                  deviceType, deviceID, capabilities);
  }

  static TargetEnvAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetEnvAttributeStorage>())
        TargetEnvAttributeStorage(std::get<0>(key), std::get<1>(key),
                                  std::get<2>(key), std::get<3>(key),
                                  std::get<4>(key), std::get<5>(key),
                                  std::get<6>(key));
  }

  Attribute version;
  Attribute revision;
  Attribute extensions;
  Attribute capabilities;
  spirv::Vendor vendorID;
  spirv::DeviceType deviceType;
  uint32_t deviceID;
};
}  // namespace detail

TargetEnvAttr TargetEnvAttr::get(Vulkan::Version version, uint32_t revision,
                                 ArrayRef<Extension> extensions,
                                 spirv::Vendor vendorID,
                                 spirv::DeviceType deviceType,
                                 uint32_t deviceID,
                                 CapabilitiesAttr capabilities) {
  mlir::Builder builder(capabilities.getContext());
  llvm::SmallVector<Attribute, 0> extAttrs;
  extAttrs.reserve(extensions.size());
  for (auto ext : extensions) {
    extAttrs.push_back(ExtensionAttr::get(builder.getContext(), ext));
  }
  return get(builder.getI32IntegerAttr(static_cast<uint32_t>(version)),
             builder.getI32IntegerAttr(revision),
             builder.getArrayAttr(extAttrs), vendorID, deviceType, deviceID,
             capabilities);
}

TargetEnvAttr TargetEnvAttr::get(IntegerAttr version, IntegerAttr revision,
                                 ArrayAttr extensions, spirv::Vendor vendorID,
                                 spirv::DeviceType deviceType,
                                 uint32_t deviceID,
                                 CapabilitiesAttr capabilities) {
  assert(version && revision && extensions && capabilities);
  MLIRContext *context = version.getContext();
  return Base::get(context, version, revision, extensions, vendorID, deviceType,
                   deviceID, capabilities);
}

StringRef TargetEnvAttr::getKindName() { return "target_env"; }

Version TargetEnvAttr::getVersion() {
  return static_cast<Version>(
      llvm::cast<IntegerAttr>(getImpl()->version).getValue().getZExtValue());
}

unsigned TargetEnvAttr::getRevision() {
  return llvm::cast<IntegerAttr>(getImpl()->revision).getValue().getZExtValue();
}

TargetEnvAttr::ext_iterator::ext_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator, Extension (*)(Attribute)>(
          it, [](Attribute attr) {
            return llvm::cast<ExtensionAttr>(attr).getValue();
          }) {}

TargetEnvAttr::ext_range TargetEnvAttr::getExtensions() {
  auto range = getExtensionsAttr().getValue();
  return {ext_iterator(range.begin()), ext_iterator(range.end())};
}

ArrayAttr TargetEnvAttr::getExtensionsAttr() {
  return llvm::cast<ArrayAttr>(getImpl()->extensions);
}

spirv::Vendor TargetEnvAttr::getVendorID() { return getImpl()->vendorID; }

spirv::DeviceType TargetEnvAttr::getDeviceType() {
  return getImpl()->deviceType;
}

uint32_t TargetEnvAttr::getDeviceID() { return getImpl()->deviceID; }

CapabilitiesAttr TargetEnvAttr::getCapabilitiesAttr() {
  return llvm::cast<CapabilitiesAttr>(getImpl()->capabilities);
}

LogicalResult TargetEnvAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, IntegerAttr version,
    IntegerAttr revision, ArrayAttr extensions, spirv::Vendor /*vendorID*/,
    spirv::DeviceType /*deviceType*/, uint32_t /*deviceID*/,
    CapabilitiesAttr capabilities) {
  if (!version.getType().isInteger(32))
    return emitError() << "expected 32-bit integer for version";

  if (!revision.getType().isInteger(32))
    return emitError() << "expected 32-bit integer for revision";

  return success();
}

//===----------------------------------------------------------------------===//
// Attribute Parsing
//===----------------------------------------------------------------------===//

namespace {

/// Parses a comma-separated list of keywords, invokes `processKeyword` on each
/// of the parsed keyword, and returns failure if any error occurs.
ParseResult parseKeywordList(
    DialectAsmParser &parser,
    function_ref<LogicalResult(llvm::SMLoc, StringRef)> processKeyword) {
  if (parser.parseLSquare()) return failure();

  // Special case for empty list.
  if (succeeded(parser.parseOptionalRSquare())) return success();

  // Keep parsing the keyword and an optional comma following it. If the comma
  // is successfully parsed, then we have more keywords to parse.
  do {
    auto loc = parser.getCurrentLocation();
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || failed(processKeyword(loc, keyword)))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare()) return failure();

  return success();
}

/// Parses a TargetEnvAttr.
Attribute parseTargetAttr(DialectAsmParser &parser) {
  if (parser.parseLess()) return {};

  Builder &builder = parser.getBuilder();

  IntegerAttr versionAttr;
  {
    auto loc = parser.getCurrentLocation();
    StringRef version;
    if (parser.parseKeyword(&version) || parser.parseComma()) return {};

    if (auto versionSymbol = symbolizeVersion(version)) {
      versionAttr =
          builder.getI32IntegerAttr(static_cast<uint32_t>(*versionSymbol));
    } else {
      parser.emitError(loc, "unknown Vulkan version: ") << version;
      return {};
    }
  }

  IntegerAttr revisionAttr;
  {
    unsigned revision = 0;
    // TODO(antiagainst): it would be nice to parse rN instad of r(N).
    if (parser.parseKeyword("r") || parser.parseLParen() ||
        parser.parseInteger(revision) || parser.parseRParen() ||
        parser.parseComma())
      return {};
    revisionAttr = builder.getI32IntegerAttr(revision);
  }

  ArrayAttr extensionsAttr;
  {
    SmallVector<Attribute, 1> extensions;
    llvm::SMLoc errorloc;
    StringRef errorKeyword;

    MLIRContext *context = parser.getContext();
    auto processExtension = [&](llvm::SMLoc loc, StringRef extension) {
      if (std::optional<Extension> symbol = symbolizeExtension(extension)) {
        extensions.push_back(ExtensionAttr::get(context, *symbol));
        return success();
      }
      return errorloc = loc, errorKeyword = extension, failure();
    };
    if (parseKeywordList(parser, processExtension) || parser.parseComma()) {
      if (!errorKeyword.empty())
        parser.emitError(errorloc, "unknown Vulkan extension: ")
            << errorKeyword;
      return {};
    }

    extensionsAttr = builder.getArrayAttr(extensions);
  }

  // Parse vendor:device-type[:device-id]
  spirv::Vendor vendorID = spirv::Vendor::Unknown;
  spirv::DeviceType deviceType = spirv::DeviceType::Unknown;
  uint32_t deviceID = spirv::TargetEnvAttr::kUnknownDeviceID;
  {
    auto loc = parser.getCurrentLocation();
    StringRef vendorStr;
    if (parser.parseKeyword(&vendorStr)) return {};
    if (auto vendorSymbol = spirv::symbolizeVendor(vendorStr)) {
      vendorID = *vendorSymbol;
    } else {
      parser.emitError(loc, "unknown vendor: ") << vendorStr;
    }

    loc = parser.getCurrentLocation();
    StringRef deviceTypeStr;
    if (parser.parseColon() || parser.parseKeyword(&deviceTypeStr)) return {};
    if (auto deviceTypeSymbol = spirv::symbolizeDeviceType(deviceTypeStr)) {
      deviceType = *deviceTypeSymbol;
    } else {
      parser.emitError(loc, "unknown device type: ") << deviceTypeStr;
    }

    loc = parser.getCurrentLocation();
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseInteger(deviceID)) return {};
    }

    if (parser.parseComma()) return {};
  }

  CapabilitiesAttr capabilities;
  if (parser.parseAttribute(capabilities)) return {};

  if (parser.parseGreater()) return {};

  return TargetEnvAttr::get(versionAttr, revisionAttr, extensionsAttr, vendorID,
                            deviceType, deviceID, capabilities);
}
}  // anonymous namespace

Attribute VulkanDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  // Vulkan attributes do not have type.
  if (type) {
    parser.emitError(parser.getNameLoc(), "unexpected type");
    return {};
  }

  // Parse the kind keyword first.
  StringRef attrKind;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &attrKind, type, attr);
  if (result.has_value()) {
    if (failed(result.value())) return {};
    return attr;
  }

  if (attrKind == TargetEnvAttr::getKindName()) return parseTargetAttr(parser);

  parser.emitError(parser.getNameLoc(), "unknown Vulkan attriubte kind: ")
      << attrKind;
  return {};
}

//===----------------------------------------------------------------------===//
// Attribute Printing
//===----------------------------------------------------------------------===//

namespace {
void print(TargetEnvAttr targetEnv, DialectAsmPrinter &printer) {
  auto &os = printer.getStream();
  printer << TargetEnvAttr::getKindName() << "<"
          << stringifyVersion(targetEnv.getVersion()) << ", r("
          << targetEnv.getRevision() << "), [";
  interleaveComma(targetEnv.getExtensions(), os,
                  [&](Extension ext) { os << stringifyExtension(ext); });
  printer << "], " << spirv::stringifyVendor(targetEnv.getVendorID());
  printer << ":" << spirv::stringifyDeviceType(targetEnv.getDeviceType());
  auto deviceID = targetEnv.getDeviceID();
  if (deviceID != spirv::TargetEnvAttr::kUnknownDeviceID) {
    printer << ":" << targetEnv.getDeviceID();
  }
  printer << ", " << targetEnv.getCapabilitiesAttr() << ">";
}
}  // anonymous namespace

void VulkanDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer))) return;

  if (auto targetEnv = llvm::dyn_cast<TargetEnvAttr>(attr))
    return print(targetEnv, printer);

  assert(false && "unhandled Vulkan attribute kind");
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void VulkanDialect::registerAttributes() {
  addAttributes<TargetEnvAttr,
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.cpp.inc"
                >();
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
