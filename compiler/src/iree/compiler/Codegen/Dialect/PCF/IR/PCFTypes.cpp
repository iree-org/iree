// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp.inc" // IWYU pragma: keep

namespace mlir::iree_compiler::IREE::PCF {

//===----------------------------------------------------------------------===//
// #pcf.sref<...>
//===----------------------------------------------------------------------===//

// static
Type ShapedRefType::parse(AsmParser &parser) {
  // Parse literal '<'
  if (parser.parseLess())
    return {};

  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  Attribute scope;

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  auto elemTypeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  auto commaLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseComma())) {
    parser.emitError(commaLoc, "expected comma after 'elementType'");
    return {};
  }

  Attribute syncScope;
  if (mlir::succeeded(parser.parseOptionalKeyword("sync"))) {
    if (mlir::failed(parser.parseLParen())) {
      return {};
    }

    auto scopeLoc = parser.getCurrentLocation();
    if (mlir::failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    // Special parsing for SyncOnParentAttr sync scope.
    syncScope = SyncOnParentAttr::get(parser.getContext());
    if (mlir::failed(parser.parseRParen())) {
      return {};
    }
  } else {
    auto scopeLoc = parser.getCurrentLocation();
    if (mlir::failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    if (!isa<ScopeAttr>(scope)) {
      parser.emitError(scopeLoc, "expected 'scope' parameter ")
          << scope << " to implement 'ScopeAttrInterface'";
      return {};
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
      auto syncLoc = parser.getCurrentLocation();
      if (failed(parser.parseAttribute(syncScope))) {
        parser.emitError(syncLoc, "failed to parse parameter 'sync_scope'");
        return {};
      }
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  MLIRContext *context = parser.getContext();
  return ShapedRefType::get(context, shape, elementType, cast<ScopeAttr>(scope),
                            syncScope);
}

void ShapedRefType::print(AsmPrinter &printer) const {
  printer << "<";

  auto shape = getShape();
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }

  printer << getElementType();
  printer << ", ";
  if (isParentScopeOnlySync()) {
    // Special case printer for parent only sync for convenience.
    printer << "sync";
    printer << "(" << getScope() << ")";
  } else if (getSyncScope()) {
    // Default for other sync scopes.
    printer << getScope() << ", " << getSyncScope();
  } else {
    // printer case with no sync scope.
    printer << getScope();
  }
  printer << ">";
}

ShapedRefType ShapedRefType::get(MLIRContext *context, ArrayRef<int64_t> shape,
                                 Type elementType, ScopeAttr scope) {
  return ShapedRefType::get(context, shape, elementType, scope, Attribute());
}

bool ShapedRefType::isParentScopeOnlySync() const {
  return isa_and_present<SyncOnParentAttr>(getSyncScope());
}

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

void PCFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::PCF
