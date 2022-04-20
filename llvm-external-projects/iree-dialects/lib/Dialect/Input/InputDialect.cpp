// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

#include "iree-dialects/Dialect/Input/InputDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"

void IREEInputDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/Input/InputOps.cpp.inc"
      >();
}

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Input {

// ListType
Type ListType::parse(AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  Type elementType;
  if (parser.parseLess() || parser.parseType(elementType) ||
      parser.parseGreater())
    return Type();
  return get(ctxt, elementType);
}

void ListType::print(AsmPrinter &printer) const {
  printer << "<" << getElementType() << ">";
}

// PtrType
Type PtrType::parse(AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  Type targetType;
  if (parser.parseLess() || parser.parseType(targetType) ||
      parser.parseGreater())
    return Type();
  return get(ctxt, targetType);
}

void PtrType::print(AsmPrinter &printer) const {
  printer << "<" << getTargetType() << ">";
}

} // namespace Input
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
