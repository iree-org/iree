// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREE/IREEDialect.h"

#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree;

#include "iree-dialects/Dialect/IREE/IREEOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/IREE/IREEOpsTypes.cpp.inc"

void IREEDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/IREE/IREEOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/IREE/IREEOps.cpp.inc"
      >();
}

Type IREEDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  Type genType;
  if (succeeded(parser.parseKeyword(&typeTag)))
    generatedTypeParser(parser, typeTag, genType);
  return genType;
}

void IREEDialect::printType(Type type, DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
