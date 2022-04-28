// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

VulkanDialect::VulkanDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VulkanDialect>()) {
  registerAttributes();
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

    auto processExtension = [&](llvm::SMLoc loc, StringRef extension) {
      if (auto symbol = symbolizeExtension(extension)) {
        extensions.push_back(builder.getI32IntegerAttr(
            static_cast<uint32_t>(symbol.getValue())));
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

  DictionaryAttr capabilities;
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(capabilities)) return {};

    if (!capabilities.isa<CapabilitiesAttr>()) {
      parser.emitError(loc,
                       "capabilities must be a vulkan::CapabilitiesAttr "
                       "dictionary attribute");
      return {};
    }
  }

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
  if (parser.parseKeyword(&attrKind)) return {};

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
  interleaveComma(targetEnv.getExtensionsAttr(), os, [&](Attribute attr) {
    os << stringifyExtension(
        *symbolizeExtension(attr.cast<IntegerAttr>().getInt()));
  });
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
  if (auto targetEnv = attr.dyn_cast<TargetEnvAttr>())
    print(targetEnv, printer);
  else
    assert(false && "unhandled Vulkan attribute kind");
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
