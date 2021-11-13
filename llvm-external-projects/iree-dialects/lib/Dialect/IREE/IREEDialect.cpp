// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREE/IREEDialect.h"

#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
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

template <typename T>
static T parseShapedType(AsmParser &parser) {
  StringRef accessStr;
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&accessStr)) ||
      failed(parser.parseColon()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)) ||
      failed(parser.parseType(elementType)) || failed(parser.parseGreater())) {
    return {};
  }
  auto access = llvm::StringSwitch<TensorAccess>(accessStr)
                    .Case("readonly", TensorAccess::ReadOnly)
                    .Case("readwrite", TensorAccess::ReadWrite)
                    .Case("writeonly", TensorAccess::WriteOnly)
                    .Default(TensorAccess::ReadOnly);
  return T::get(access, shape, elementType);
}

static void printShapedType(const DeviceTensorType &type, AsmPrinter &p) {
  switch (type.getAccess()) {
    case TensorAccess::ReadOnly:
      p << "readonly";
      break;
    case TensorAccess::ReadWrite:
      p << "readwrite";
      break;
    case TensorAccess::WriteOnly:
      p << "writeonly";
      break;
    default:
      llvm_unreachable("unhandled access");
  }
  p << ":";
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      p << '?';
    } else {
      p << dim;
    }
    p << 'x';
  }
  p << type.getElementType();
}

//===----------------------------------------------------------------------===//
// DeviceTensorType
//===----------------------------------------------------------------------===//

LogicalResult DeviceTensorType::verify(
    function_ref<InFlightDiagnostic()> emitError, TensorAccess access,
    ArrayRef<int64_t> shape, Type elementType) {
  if (!TensorType::isValidElementType(elementType)) {
    return emitError() << "dispatch tensor elements must be int or float type";
  }
  if (any_of(shape, [](int64_t i) { return i < -1; })) {
    return emitError()
           << "dispatch tensor dimensions must be positive if defined";
  }
  return success();
}

Type DeviceTensorType::parse(AsmParser &parser) {
  return parseShapedType<DeviceTensorType>(parser);
}

void DeviceTensorType::print(AsmPrinter &p) const {
  p << "device.tensor<";
  printShapedType(*this, p);
  p << '>';
}
