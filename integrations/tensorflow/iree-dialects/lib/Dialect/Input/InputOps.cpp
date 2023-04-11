// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputOps.h"

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

static ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                         StringAttr &symVisibilityAttr) {
  StringRef symVisibility;
  if (succeeded(parser.parseOptionalKeyword(&symVisibility,
                                            {"public", "private", "nested"}))) {
    symVisibilityAttr = parser.getBuilder().getStringAttr(symVisibility);
  }
  return success();
}

static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr symVisibilityAttr) {
  if (!symVisibilityAttr) {
    p << "public";
  } else {
    p << symVisibilityAttr.getValue();
  }
}

//===----------------------------------------------------------------------===//
// custom<TypeOrAttr>($type, $attr)
//===----------------------------------------------------------------------===//
// some.op custom<TypeOrAttr>($type, $attr)
// ->
// some.op : i32
// some.op = 42 : i32
// some.op : i32 = 42 : index

static ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                                   TypedAttr &attr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
    typeAttr = TypeAttr::get(attr.getType());
    return success();
  }

  Type type;
  if (failed(parser.parseColonType(type))) {
    return parser.emitError(parser.getCurrentLocation()) << "expected type";
  }
  typeAttr = TypeAttr::get(type);

  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
  }

  return success();
}

static void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                            TypedAttr attr) {
  if (!attr || attr.getType() != type.getValue()) {
    p << " : ";
    p.printAttribute(type);
  }
  if (attr) {
    p << " = ";
    p.printAttribute(attr);
  }
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, Type type,
                     std::optional<TypedAttr> initialValue,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  if (initialValue.has_value()) {
    result.addAttribute("initial_value", initialValue.value());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, Type type,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, std::nullopt, attrs);
}

// Returns true if the given |accessType| is compatible with the |globalType|.
// For example, this will return true if the global type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isGlobalTypeCompatible(Type globalType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (globalType.isa<ShapedType>() && accessType.isa<ShapedType>()) {
    return succeeded(mlir::verifyCompatibleShape(globalType, accessType)) &&
           globalType.cast<ShapedType>().getElementType() ==
               accessType.cast<ShapedType>().getElementType();
  }

  // Permissively allow any other types to be marked compatible as long as
  // neither are shaped type.
  return !globalType.isa<ShapedType>() && !accessType.isa<ShapedType>();
}

LogicalResult
GlobalLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  if (!globalOp) {
    return emitOpError() << "undefined global: " << getGlobal();
  }
  auto loadType = getResult().getType();
  if (!isGlobalTypeCompatible(globalOp.getType(), loadType)) {
    return emitOpError() << "global type mismatch; global " << getGlobal()
                         << " is " << globalOp.getType() << " but load is "
                         << loadType;
  }
  return success();
}

LogicalResult GlobalLoadIndirectOp::verify() {
  auto globalType = getGlobal().getType().cast<PtrType>().getTargetType();
  auto loadType = getResult().getType();
  if (!isGlobalTypeCompatible(globalType, loadType)) {
    return emitOpError() << "global type mismatch; global pointer is "
                         << globalType << " but load is " << loadType;
  }
  return success();
}

LogicalResult
GlobalStoreOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  if (!globalOp) {
    return emitOpError() << "undefined global: " << getGlobal();
  }
  auto storeType = getValue().getType();
  if (!isGlobalTypeCompatible(globalOp.getType(), storeType)) {
    return emitOpError() << "global type mismatch; global " << getGlobal()
                         << " is " << globalOp.getType() << " but store is "
                         << storeType;
  }
  return success();
}

LogicalResult GlobalStoreIndirectOp::verify() {
  auto globalType = getGlobal().getType().cast<PtrType>().getTargetType();
  auto storeType = getValue().getType();
  if (!isGlobalTypeCompatible(globalType, storeType)) {
    return emitOpError() << "global type mismatch; global pointer is "
                         << globalType << " but store is " << storeType;
  }
  return success();
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/Input/InputOps.cpp.inc"
