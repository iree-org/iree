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

Type ShapedRefType::parse(AsmParser &parser) {
  if (parser.parseLess()) {
    return {};
  }

  SmallVector<int64_t> shape;
  Type elementType;
  Attribute scope;

  SMLoc shapeLoc = parser.getCurrentLocation();
  if (failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  SMLoc elemTypeLoc = parser.getCurrentLocation();
  if (failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  SMLoc commaLoc = parser.getCurrentLocation();
  if (failed(parser.parseComma())) {
    parser.emitError(commaLoc, "expected comma after 'elementType'");
    return {};
  }

  Attribute syncScope;
  if (succeeded(parser.parseOptionalKeyword("sync"))) {
    if (failed(parser.parseLParen())) {
      return {};
    }

    SMLoc scopeLoc = parser.getCurrentLocation();
    if (failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    // Special parsing for SyncOnReturnAttr sync scope.
    syncScope = SyncOnReturnAttr::get(parser.getContext());
    if (failed(parser.parseRParen())) {
      return {};
    }
  } else {
    SMLoc scopeLoc = parser.getCurrentLocation();
    if (failed(parser.parseAttribute(scope))) {
      parser.emitError(scopeLoc, "failed to parse parameter 'scope'");
      return {};
    }

    if (!isa<ScopeAttrInterface>(scope)) {
      parser.emitError(scopeLoc, "expected 'scope' parameter ")
          << scope << " to implement 'ScopeAttrInterfaceInterface'";
      return {};
    }

    if (succeeded(parser.parseOptionalComma())) {
      SMLoc syncLoc = parser.getCurrentLocation();
      if (failed(parser.parseAttribute(syncScope))) {
        parser.emitError(syncLoc, "failed to parse parameter 'sync_scope'");
        return {};
      }
    }
  }

  if (parser.parseGreater()) {
    return {};
  }

  MLIRContext *context = parser.getContext();
  return ShapedRefType::get(context, shape, elementType,
                            cast<ScopeAttrInterface>(scope), syncScope);
}

void ShapedRefType::print(AsmPrinter &printer) const {
  printer << "<";

  ArrayRef<int64_t> shape = getShape();
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }

  printer << getElementType();
  printer << ", ";
  if (isReturnOnlySync()) {
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
                                 Type elementType, ScopeAttrInterface scope) {
  return ShapedRefType::get(context, shape, elementType, scope, Attribute());
}

bool ShapedRefType::isReturnOnlySync() const {
  return isa_and_present<SyncOnReturnAttr>(getSyncScope());
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
