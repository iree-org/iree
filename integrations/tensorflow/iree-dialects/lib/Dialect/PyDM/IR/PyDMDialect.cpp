// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"

#include "iree-dialects/Dialect/PyDM/IR/PyDMInterfaces.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.cpp.inc"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOpInterfaces.cpp.inc"
#include "iree-dialects/Dialect/PyDM/IR/PyDMTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/PyDM/IR/PyDMTypes.cpp.inc"

//------------------------------------------------------------------------------
// Dialect implementation
//------------------------------------------------------------------------------

using BuiltinIntegerType = mlir::IntegerType;

using PyBoolType = PYDM::BoolType;
using PyConstantOp = PYDM::ConstantOp;
using PyIntegerType = PYDM::IntegerType;
using PyListType = PYDM::ListType;
using PyRealType = PYDM::RealType;
using PyObjectType = PYDM::ObjectType;
using PyUnionType = PYDM::UnionType;

void IREEPyDMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/PyDM/IR/PyDMTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.cpp.inc"
      >();
}

Operation *IREEPyDMDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  // Since we support materialization of builtin types too, explicitly
  // allow these.
  if (type.isa<PyBoolType, BytesType, PyIntegerType, PyRealType, StrType,
               BuiltinIntegerType>()) {
    return builder.create<PYDM::ConstantOp>(loc, type, value);
  }

  if (type.isa<NoneType>()) {
    return builder.create<PYDM::NoneOp>(loc, type);
  }

  if (type.isa<ExceptionResultType>() && value.isa<UnitAttr>()) {
    return builder.create<PYDM::SuccessOp>(loc, type);
  }

  assert(false && "unhandled iree_pydm constant materialization");
  return nullptr;
}

//------------------------------------------------------------------------------
// Python type implementation
//------------------------------------------------------------------------------

// BoolType
BuiltinTypeCode PYDM::BoolType::getTypeCode() const {
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef PYDM::BoolType::getPythonTypeName() const { return "bool"; }

Optional<NumericCategory> PYDM::BoolType::getNumericCategory() const {
  return NumericCategory::Bool;
}

Optional<int> PYDM::BoolType::getNumericSubTypeCode() const { return 0; }

Optional<int> PYDM::BoolType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
}

// BytesType
BuiltinTypeCode PYDM::BytesType::getTypeCode() const {
  return BuiltinTypeCode::Bytes;
}

StringRef PYDM::BytesType::getPythonTypeName() const { return "bytes"; }

// ExceptionResultType
BuiltinTypeCode PYDM::ExceptionResultType::getTypeCode() const {
  return BuiltinTypeCode::ExceptionResult;
}

StringRef PYDM::ExceptionResultType::getPythonTypeName() const {
  return "Exception";
}

// IntegerType
LogicalResult
PYDM::IntegerType::verify(function_ref<InFlightDiagnostic()> emitError,
                          Optional<int> bitWidth) {
  if (!bitWidth)
    return success();
  int w = abs(*bitWidth);
  if (w == 0 || w == 8 || w == 16 || w == 32 || w == 64)
    return success();
  return emitError() << "unsupported python integer bit width: " << w;
}

Type PyIntegerType::parse(mlir::AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  auto emitError = [&]() -> InFlightDiagnostic {
    return parser.emitError(parser.getCurrentLocation());
  };
  // Weak
  if (failed(parser.parseOptionalLess()))
    return get(ctxt);
  // AP
  if (succeeded(parser.parseOptionalStar())) {
    if (failed(parser.parseGreater()))
      return Type();
    return get(ctxt, None);
  }

  // Explicit
  bool isSigned;
  if (succeeded(parser.parseOptionalKeyword("unsigned"))) {
    isSigned = false;
  } else {
    isSigned = true;
  }

  int width;
  if (failed(parser.parseInteger(width)))
    return Type();
  if (failed(parser.parseGreater()))
    return Type();
  if (!isSigned)
    width = -width;
  return getChecked(emitError, ctxt, width);
}

void PyIntegerType::print(mlir::AsmPrinter &printer) const {
  auto w = getImpl()->bitWidth;
  if (w) {
    printer << "<";
    if (*w == 0) {
      printer << "*";
    } else if (*w > 0) {
      printer << *w;
    } else {
      printer << "unsigned " << (-*w);
    }
    printer << ">";
  }
}

BuiltinTypeCode PYDM::IntegerType::getTypeCode() const {
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef PYDM::IntegerType::getPythonTypeName() const { return "int"; }

Optional<NumericCategory> PYDM::IntegerType::getNumericCategory() const {
  if (isWeak())
    return NumericCategory::WeakInteger;
  if (getBitWidth() == 0)
    return NumericCategory::APSigned;
  if (isSigned())
    return NumericCategory::Signed;
  return NumericCategory::Unsigned;
}

Optional<int> PYDM::IntegerType::getNumericSubTypeCode() const {
  if (isWeak())
    return 0;
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
    stc = IntegerSubTypeCode::Integer8; // Arbitrarily picked value.
    assert(false && "unsupported numeric bitwidth");
  }
  }
  return static_cast<int>(stc);
}

Optional<int> PYDM::IntegerType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
}

bool PYDM::IntegerType::isWeak() const { return !getImpl()->bitWidth; }

unsigned PYDM::IntegerType::getBitWidth() const {
  return abs(*getImpl()->bitWidth);
}

bool PYDM::IntegerType::isSigned() const { return *getImpl()->bitWidth >= 0; }

BuiltinTypeCode PYDM::ListType::getTypeCode() const {
  return BuiltinTypeCode::List;
}

// ListType
void PyListType::print(mlir::AsmPrinter &printer) const {
  if (getImpl()->uniformElementType ||
      getImpl()->storageClass != CollectionStorageClass::Boxed) {
    printer << "<";
    switch (getImpl()->storageClass) {
    case CollectionStorageClass::Boxed:
      printer << "boxed";
      break;
    case CollectionStorageClass::Empty:
      printer << "empty";
      break;
    case CollectionStorageClass::Unboxed:
      printer << "unboxed";
      break;
    }

    if (getImpl()->uniformElementType) {
      printer << ",";
      printer << getImpl()->uniformElementType;
    }
    printer << ">";
  }
}

Type PyListType::parse(mlir::AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseOptionalLess())
    return get(ctxt, CollectionStorageClass::Boxed, nullptr);

  Type t;
  StringRef storageClassKeyword;
  if (parser.parseKeyword(&storageClassKeyword))
    return Type();
  if (parser.parseComma())
    return Type();
  if (parser.parseType(t))
    return Type();
  if (parser.parseGreater())
    return Type();

  CollectionStorageClass storageClass;
  if (storageClassKeyword == "boxed")
    storageClass = CollectionStorageClass::Boxed;
  else if (storageClassKeyword == "empty")
    storageClass = CollectionStorageClass::Empty;
  else if (storageClassKeyword == "unboxed")
    storageClass = CollectionStorageClass::Unboxed;
  else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected one of 'boxed', 'empty', 'unboxed'");
    return Type();
  }
  return get(ctxt, storageClass, t);
}

StringRef PYDM::ListType::getPythonTypeName() const { return "list"; }

BuiltinTypeCode PYDM::NoneType::getTypeCode() const {
  return BuiltinTypeCode::List;
}

bool PYDM::ListType::isRefinable() const {
  if (getStorageClass() == CollectionStorageClass::Empty)
    return false;

  if (!getUniformElementType())
    return true;

  if (auto pyType = getUniformElementType().dyn_cast<PythonTypeInterface>())
    return pyType.isRefinable();

  return false;
}

Type PYDM::ListType::getElementStorageType() const {
  switch (getStorageClass()) {
  case CollectionStorageClass::Boxed:
  case CollectionStorageClass::Empty:
    return ObjectType::get(getContext());
  case CollectionStorageClass::Unboxed:
    assert(getUniformElementType() &&
           "unboxed list should have uniform element type");
    return getUniformElementType();
  default:
    assert(false && "unsupported storage class");
    return {};
  }
}

// NoneType
StringRef PYDM::NoneType::getPythonTypeName() const { return "None"; }

// ObjectType
void PyObjectType::print(mlir::AsmPrinter &printer) const {
  if (getImpl()->primitiveType)
    printer << "<" << getImpl()->primitiveType << ">";
}

Type PyObjectType::parse(mlir::AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseOptionalLess())
    return get(ctxt, nullptr);

  Type t;
  if (parser.parseType(t))
    return Type();
  if (parser.parseGreater())
    return Type();
  if (auto primitiveType = t.dyn_cast<PrimitiveType>())
    return get(ctxt, primitiveType);
  else {
    parser.emitError(parser.getNameLoc(), "expected a primitive type");
    return Type();
  }
}

BuiltinTypeCode PYDM::ObjectType::getTypeCode() const {
  return BuiltinTypeCode::Object;
}

StringRef PYDM::ObjectType::getPythonTypeName() const { return "object"; }

bool PYDM::ObjectType::isRefinable() const {
  if (!getPrimitiveType())
    return true;

  if (auto pyType = getPrimitiveType().dyn_cast<PythonTypeInterface>())
    return pyType.isRefinable();

  return false;
}

// RealType
void PyRealType::print(mlir::AsmPrinter &printer) const {
  auto ft = getImpl()->floatType;
  if (ft)
    printer << "<" << ft << ">";
}

Type PyRealType::parse(mlir::AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();

  auto emitError = [&]() -> InFlightDiagnostic {
    return parser.emitError(parser.getCurrentLocation());
  };
  // Weak
  if (failed(parser.parseOptionalLess()))
    return get(ctxt);
  // Explicit
  FloatType subType;
  if (failed(parser.parseType(subType)))
    return Type();
  if (failed(parser.parseGreater()))
    return Type();
  return getChecked(emitError, ctxt, subType);
}

LogicalResult
PYDM::RealType::verify(function_ref<InFlightDiagnostic()> emitError,
                       FloatType floatType) {
  if (!floatType)
    return success();
  if (!floatType.isa<BFloat16Type, Float16Type, Float32Type, Float64Type>()) {
    return emitError() << "unsupported Python floating point type: "
                       << floatType;
  }
  return success();
}

BuiltinTypeCode PYDM::RealType::getTypeCode() const {
  return static_cast<BuiltinTypeCode>(
      makeNumericTypeCode(*getNumericCategory(), *getNumericSubTypeCode()));
}

StringRef PYDM::RealType::getPythonTypeName() const { return "float"; }

Optional<NumericCategory> PYDM::RealType::getNumericCategory() const {
  if (isWeak())
    return NumericCategory::WeakReal;
  return NumericCategory::Real;
}

Optional<int> PYDM::RealType::getNumericSubTypeCode() const {
  if (isWeak())
    return 0;
  RealSubTypeCode stc =
      TypeSwitch<Type, RealSubTypeCode>(getFloatType())
          .Case([](BFloat16Type t) { return RealSubTypeCode::BF16; })
          .Case([](Float16Type t) { return RealSubTypeCode::FP16; })
          .Case([](Float32Type t) { return RealSubTypeCode::FP32; })
          .Case([](Float64Type t) { return RealSubTypeCode::FP64; })
          .Default([](Type t) {
            assert(false && "unsupported float type");
            return RealSubTypeCode::FP64;
          });
  return static_cast<int>(stc);
}

Optional<int> PYDM::RealType::getNumericPromotionOrder() const {
  return static_cast<int>(getTypeCode());
}

bool PYDM::RealType::isWeak() const { return !getImpl()->floatType; }

// StrType
BuiltinTypeCode PYDM::StrType::getTypeCode() const {
  return BuiltinTypeCode::Str;
}

StringRef PYDM::StrType::getPythonTypeName() const { return "str"; }

// TupleType
BuiltinTypeCode PYDM::TupleType::getTypeCode() const {
  return BuiltinTypeCode::Tuple;
}

StringRef PYDM::TupleType::getPythonTypeName() const { return "tuple"; }

// TypeType
BuiltinTypeCode PYDM::TypeType::getTypeCode() const {
  return BuiltinTypeCode::Type;
}

StringRef PYDM::TypeType::getPythonTypeName() const { return "type"; }

Type PYDM::TupleType::getElementStorageType() const {
  // TODO: When it implements unboxed storage, switch here.
  return ObjectType::get(getContext());
}

//------------------------------------------------------------------------------
// Union type implementation
//------------------------------------------------------------------------------

void PyUnionType::print(mlir::AsmPrinter &printer) const {
  llvm::interleaveComma(getAlternatives(), printer);
}

Type PyUnionType::parse(mlir::AsmParser &parser) {
  MLIRContext *ctxt = parser.getContext();
  if (parser.parseOptionalLess())
    return get(ctxt, {});

  SmallVector<::mlir::Type> alternatives;

  do {
    Type type;
    if (parser.parseType(type))
      return Type();
    alternatives.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  return getChecked([&]() { return parser.emitError(parser.getNameLoc()); },
                    ctxt, alternatives);
}

LogicalResult
PYDM::UnionType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<Type> alternatives) {
  int lastTypeCode = 0;
  for (Type alternative : alternatives) {
    if (auto pythonType = alternative.dyn_cast<PYDM::PythonTypeInterface>()) {
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
