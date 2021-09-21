// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.h"

#include "iree-dialects/Dialect/IREEPyDM/IR/Interfaces.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_pydm;

#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/IREEPyDM/IR/TypeInterfaces.cpp.inc"
#include "iree-dialects/Dialect/IREEPyDM/IR/Types.cpp.inc"

//------------------------------------------------------------------------------
// Dialect implementation
//------------------------------------------------------------------------------

using BuiltinIntegerType = mlir::IntegerType;

using PyBoolType = mlir::iree_pydm::BoolType;
using PyConstantOp = mlir::iree_pydm::ConstantOp;
using PyIntegerType = mlir::iree_pydm::IntegerType;
using PyRealType = mlir::iree_pydm::RealType;

void IREEPyDMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/IREEPyDM/IR/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.cpp.inc"
      >();
}

Operation *IREEPyDMDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  // Since we support materialization of builtin types too, explicitly
  // allow these.
  if (type.isa<PyBoolType, BytesType, PyIntegerType, PyRealType, StrType,
               BuiltinIntegerType>()) {
    return builder.create<iree_pydm::ConstantOp>(loc, type, value);
  }

  if (type.isa<NoneType>()) {
    return builder.create<iree_pydm::NoneOp>(loc, type);
  }

  if (type.isa<ExceptionResultType>() && value.isa<UnitAttr>()) {
    return builder.create<iree_pydm::SuccessOp>(loc, type);
  }

  llvm_unreachable("unhandled iree_pydm constant materialization");
}

Type IREEPyDMDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  if (succeeded(parser.parseKeyword(&typeTag))) {
    Type genType;
    auto parseResult =
        generatedTypeParser(getContext(), parser, typeTag, genType);
    if (parseResult.hasValue()) {
      if (*parseResult) {
        return Type();
      }
      return genType;
    }
  }

  parser.emitError(parser.getNameLoc(), "unknown dialect type");
  return Type();
}

void IREEPyDMDialect::printType(Type type, DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}

//------------------------------------------------------------------------------
// Python type implementation
//------------------------------------------------------------------------------

BuiltinTypeCode iree_pydm::BoolType::getTypeCode() const {
  return BuiltinTypeCode::Bool;
}

StringRef iree_pydm::BoolType::getPythonTypeName() const { return "bool"; }

Optional<int> iree_pydm::BoolType::getNumericPromotionOrder() const {
  return 1;
}

BuiltinTypeCode iree_pydm::BytesType::getTypeCode() const {
  return BuiltinTypeCode::Bytes;
}

StringRef iree_pydm::BytesType::getPythonTypeName() const { return "bytes"; }

BuiltinTypeCode iree_pydm::ExceptionResultType::getTypeCode() const {
  return BuiltinTypeCode::ExceptionResult;
}

StringRef iree_pydm::ExceptionResultType::getPythonTypeName() const {
  return "Exception";
}

BuiltinTypeCode iree_pydm::IntegerType::getTypeCode() const {
  return BuiltinTypeCode::Integer;
}

StringRef iree_pydm::IntegerType::getPythonTypeName() const { return "int"; }

Optional<int> iree_pydm::IntegerType::getNumericPromotionOrder() const {
  return 2;
}

BuiltinTypeCode iree_pydm::ListType::getTypeCode() const {
  return BuiltinTypeCode::List;
}

StringRef iree_pydm::ListType::getPythonTypeName() const { return "list"; }

BuiltinTypeCode iree_pydm::NoneType::getTypeCode() const {
  return BuiltinTypeCode::None;
}

StringRef iree_pydm::NoneType::getPythonTypeName() const { return "None"; }

BuiltinTypeCode iree_pydm::ObjectType::getTypeCode() const {
  return BuiltinTypeCode::Object;
}

StringRef iree_pydm::ObjectType::getPythonTypeName() const { return "object"; }

BuiltinTypeCode iree_pydm::RealType::getTypeCode() const {
  return BuiltinTypeCode::Real;
}

StringRef iree_pydm::RealType::getPythonTypeName() const { return "float"; }

Optional<int> iree_pydm::RealType::getNumericPromotionOrder() const {
  return 3;
}

BuiltinTypeCode iree_pydm::StrType::getTypeCode() const {
  return BuiltinTypeCode::Str;
}

StringRef iree_pydm::StrType::getPythonTypeName() const { return "str"; }

BuiltinTypeCode iree_pydm::TupleType::getTypeCode() const {
  return BuiltinTypeCode::Tuple;
}

StringRef iree_pydm::TupleType::getPythonTypeName() const { return "tuple"; }

BuiltinTypeCode iree_pydm::TypeType::getTypeCode() const {
  return BuiltinTypeCode::Type;
}

StringRef iree_pydm::TypeType::getPythonTypeName() const { return "type"; }

//------------------------------------------------------------------------------
// Union type implementation
//------------------------------------------------------------------------------

LogicalResult iree_pydm::UnionType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Type> alternatives) {
  int lastTypeCode = 0;
  for (Type alternative : alternatives) {
    if (auto pythonType =
            alternative.dyn_cast<iree_pydm::PythonTypeInterface>()) {
      int thisTypeCode = static_cast<int>(pythonType.getTypeCode());
      // TODO: This doesn't account for parameterized types.
      if (thisTypeCode <= lastTypeCode) {
        return emitError() << "expected total order of union to be normative. "
                              "got out of order: "
                           << alternative;
      }
    } else {
      return emitError() << "expected a python type in union. got: "
                         << alternative;
    }
  }

  return failure();
}
