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
    auto parseResult = generatedTypeParser(parser, typeTag, genType);
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
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef iree_pydm::BoolType::getPythonTypeName() const { return "bool"; }

Optional<NumericCategory> iree_pydm::BoolType::getNumericCategory() const {
  return NumericCategory::Bool;
}

Optional<int> iree_pydm::BoolType::getNumericSubTypeCode() const { return 0; }

Optional<int> iree_pydm::BoolType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
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

LogicalResult iree_pydm::IntegerType::verify(
    function_ref<InFlightDiagnostic()> emitError, Optional<int> bitWidth) {
  if (!bitWidth) return success();
  int w = abs(*bitWidth);
  if (w == 0 || w == 8 || w == 16 || w == 32 || w == 64) return success();
  return emitError() << "unsupported python integer bit width: " << w;
}

BuiltinTypeCode iree_pydm::IntegerType::getTypeCode() const {
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef iree_pydm::IntegerType::getPythonTypeName() const { return "int"; }

Optional<NumericCategory> iree_pydm::IntegerType::getNumericCategory() const {
  if (isWeak()) return NumericCategory::WeakInteger;
  if (getBitWidth() == 0) return NumericCategory::APSigned;
  if (isSigned()) return NumericCategory::Signed;
  return NumericCategory::Unsigned;
}

Optional<int> iree_pydm::IntegerType::getNumericSubTypeCode() const {
  if (isWeak()) return 0;
  IntegerSubTypeCode stc;
  switch (getBitWidth()) {
    case 8:
      stc = IntegerSubTypeCode::Integer8;
      break;
    case 16:
      stc = IntegerSubTypeCode::Integer16;
      break;
    case 32:
      stc = IntegerSubTypeCode::Integer32;
      break;
    case 64:
      stc = IntegerSubTypeCode::Integer64;
      break;
    default: {
      llvm_unreachable("unsupported numeric bitwidth");
    }
  }
  return static_cast<int>(stc);
}

Optional<int> iree_pydm::IntegerType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
}

bool iree_pydm::IntegerType::isWeak() const { return !getImpl()->bitWidth; }

unsigned iree_pydm::IntegerType::getBitWidth() const {
  return abs(*getImpl()->bitWidth);
}

bool iree_pydm::IntegerType::isSigned() const {
  return *getImpl()->bitWidth >= 0;
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

LogicalResult iree_pydm::RealType::verify(
    function_ref<InFlightDiagnostic()> emitError, FloatType floatType) {
  if (!floatType) return success();
  if (!floatType.isa<BFloat16Type, Float16Type, Float32Type, Float64Type>()) {
    return emitError() << "unsupported Python floating point type: "
                       << floatType;
  }
  return success();
}

BuiltinTypeCode iree_pydm::RealType::getTypeCode() const {
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef iree_pydm::RealType::getPythonTypeName() const { return "float"; }

Optional<NumericCategory> iree_pydm::RealType::getNumericCategory() const {
  if (isWeak()) return NumericCategory::WeakReal;
  return NumericCategory::Real;
}

Optional<int> iree_pydm::RealType::getNumericSubTypeCode() const {
  if (isWeak()) return 0;
  RealSubTypeCode stc =
      TypeSwitch<Type, RealSubTypeCode>(getFloatType())
          .Case([](BFloat16Type t) { return RealSubTypeCode::BF16; })
          .Case([](Float16Type t) { return RealSubTypeCode::FP16; })
          .Case([](Float32Type t) { return RealSubTypeCode::FP32; })
          .Case([](Float64Type t) { return RealSubTypeCode::FP64; })
          .Default([](Type t) {
            llvm_unreachable("unsupported float type");
            return RealSubTypeCode::FP64;
          });
  return static_cast<int>(stc);
}

Optional<int> iree_pydm::RealType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
}

bool iree_pydm::RealType::isWeak() const { return !getImpl()->floatType; }

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
