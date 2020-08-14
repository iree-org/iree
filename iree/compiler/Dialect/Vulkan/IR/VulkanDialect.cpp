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

#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"

#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Vulkan {

static DialectRegistration<VulkanDialect> vkDialect;

VulkanDialect::VulkanDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VulkanDialect>()) {
  addAttributes<TargetEnvAttr>();
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
      if (symbolizeExtension(extension)) {
        extensions.push_back(builder.getStringAttr(extension));
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

  return TargetEnvAttr::get(versionAttr, revisionAttr, extensionsAttr,
                            capabilities);
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
    os << attr.cast<StringAttr>().getValue();
  });
  printer << "], " << targetEnv.getCapabilitiesAttr() << ">";
}
}  // anonymous namespace

void VulkanDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  if (auto targetEnv = attr.dyn_cast<TargetEnvAttr>())
    print(targetEnv, printer);
  else
    llvm_unreachable("unhandled Vulkan attribute kind");
}

}  // namespace Vulkan
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
