// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

#include "iree-dialects/Dialect/Input/InputDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputAttrs.cpp.inc"
#include "iree-dialects/Dialect/Input/InputEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"

void IREEInputDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/Input/InputAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/Input/InputOps.cpp.inc"
      >();
}

namespace mlir::iree_compiler::IREE::Input {

//===----------------------------------------------------------------------===//
// IREE ABI helpers for constructing buffer views
//===----------------------------------------------------------------------===//

// Keep these in sync with iree/hal/api.h
namespace {
enum class NumericalType : uint32_t {
  kUnknown = 0x00,
  kInteger = 0x10,
  kIntegerSigned = kInteger | 0x01,
  kIntegerUnsigned = kInteger | 0x02,
  kBoolean = kInteger | 0x03,
  kFloat = 0x20,
  kFloatIEEE = kFloat | 0x01,
  kFloatBrain = kFloat | 0x02,
  kFloatComplex = kFloat | 0x03,
};
} // namespace

static constexpr int32_t makeElementTypeValue(NumericalType numericalType,
                                              int32_t bitCount) {
  return (static_cast<uint32_t>(numericalType) << 24) | bitCount;
}

std::optional<int32_t> getElementTypeValue(Type type) {
  if (auto intType = llvm::dyn_cast_if_present<IntegerType>(type)) {
    NumericalType numericalType;
    if (intType.isInteger(1)) {
      return makeElementTypeValue(NumericalType::kBoolean, 8);
    } else if (intType.isSigned()) {
      numericalType = NumericalType::kIntegerSigned;
    } else if (intType.isUnsigned()) {
      numericalType = NumericalType::kIntegerUnsigned;
    } else {
      // There's no such thing as a signless integer in machine types but we
      // need to be able to round-trip the format through the ABI. Exact
      // numerical type equality comparisons may fail if the frontend assumes
      // signed/unsigned but the compiler is propagating signless.
      numericalType = NumericalType::kInteger;
    }
    return makeElementTypeValue(numericalType, intType.getWidth());
  } else if (auto floatType = llvm::dyn_cast_if_present<FloatType>(type)) {
    switch (APFloat::SemanticsToEnum(floatType.getFloatSemantics())) {
    case APFloat::S_IEEEhalf:
    case APFloat::S_IEEEsingle:
    case APFloat::S_IEEEdouble:
    case APFloat::S_IEEEquad:
      return makeElementTypeValue(NumericalType::kFloatIEEE,
                                  floatType.getWidth());
    case APFloat::S_BFloat:
      return makeElementTypeValue(NumericalType::kFloatBrain,
                                  floatType.getWidth());
    default:
      return std::nullopt;
    }
  } else if (auto complexType = llvm::dyn_cast_if_present<ComplexType>(type)) {
    return makeElementTypeValue(
        NumericalType::kFloatComplex,
        complexType.getElementType().getIntOrFloatBitWidth() * 2);
  }
  return std::nullopt;
}

std::optional<int32_t> getEncodingTypeValue(Attribute attr) {
  // TODO(#6762): encoding attribute handling/mapping to enums.
  assert(!attr && "encoding types other than default not yet supported");
  // Default to IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR for now.
  return 1;
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// PtrType
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// #iree_input.device.target
//===----------------------------------------------------------------------===//

DeviceTargetAttr DeviceTargetAttr::get(MLIRContext *context,
                                       StringRef deviceID) {
  return get(context, StringAttr::get(context, deviceID),
             DictionaryAttr::get(context));
}

Attribute DeviceTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr deviceIDAttr;
  DictionaryAttr configAttr;
  // `<"device-id"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(deviceIDAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), deviceIDAttr, configAttr);
}

void DeviceTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getDeviceID());
  auto configAttr = getConfiguration();
  if (configAttr && !configAttr.empty()) {
    os << ", ";
    p.printAttribute(configAttr);
  }
  os << ">";
}

//===----------------------------------------------------------------------===//
// #iree_input.executable.target
//===----------------------------------------------------------------------===//

ExecutableTargetAttr ExecutableTargetAttr::get(MLIRContext *context,
                                               StringRef backend,
                                               StringRef format) {
  return get(context, StringAttr::get(context, backend),
             StringAttr::get(context, format), DictionaryAttr::get(context));
}

Attribute ExecutableTargetAttr::parse(AsmParser &p, Type type) {
  StringAttr backendAttr;
  StringAttr formatAttr;
  DictionaryAttr configurationAttr;
  // `<"backend", "format"`
  if (failed(p.parseLess()) || failed(p.parseAttribute(backendAttr)) ||
      failed(p.parseComma()) || failed(p.parseAttribute(formatAttr))) {
    return {};
  }
  // `, {config}`
  if (succeeded(p.parseOptionalComma()) &&
      failed(p.parseAttribute(configurationAttr))) {
    return {};
  }
  // `>`
  if (failed(p.parseGreater())) {
    return {};
  }
  return get(p.getContext(), backendAttr, formatAttr, configurationAttr);
}

void ExecutableTargetAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<";
  p.printAttribute(getBackend());
  os << ", ";
  p.printAttribute(getFormat());
  auto config = getConfiguration();
  if (config && !config.empty()) {
    os << ", ";
    p.printAttribute(config);
  }
  os << ">";
}

//===----------------------------------------------------------------------===//
// #iree_input.executable.object
//===----------------------------------------------------------------------===//

Attribute ExecutableObjectAttr::parse(AsmParser &p, Type type) {
  NamedAttrList dict;
  // `<{` dict `}>`
  if (failed(p.parseLess()) || failed(p.parseOptionalAttrDict(dict)) ||
      failed(p.parseGreater())) {
    return {};
  }
  auto pathAttr = llvm::dyn_cast_if_present<StringAttr>(dict.get("path"));
  auto dataAttr =
      llvm::dyn_cast_if_present<DenseIntElementsAttr>(dict.get("data"));
  return get(p.getContext(), pathAttr, dataAttr);
}

void ExecutableObjectAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  if (auto pathAttr = getPath()) {
    os << "path = ";
    p.printAttribute(getPath());
  }
  if (auto dataAttr = getData()) {
    os << ", data = ";
    p.printAttribute(getData());
  }
  os << "}>";
}

//===----------------------------------------------------------------------===//
// #iree_input.executable.objects
//===----------------------------------------------------------------------===//

LogicalResult ExecutableObjectsAttr::verify(
    function_ref<mlir::InFlightDiagnostic()> emitError, ArrayAttr targetsAttr,
    ArrayAttr targetObjectsAttr) {
  if (targetsAttr.size() != targetObjectsAttr.size()) {
    return emitError() << "targets and objects must be 1:1";
  }
  for (auto targetAttr : targetsAttr) {
    if (!llvm::isa<IREE::Input::ExecutableTargetAttr>(targetAttr)) {
      return emitError()
             << "target keys must be #iree_input.executable.target attributes";
    }
  }
  for (auto objectsAttr : targetObjectsAttr) {
    auto objectsArrayAttr = llvm::dyn_cast<ArrayAttr>(objectsAttr);
    if (!objectsArrayAttr) {
      return emitError() << "target objects must be an array of "
                            "#iree_input.executable.object attributes";
    }
  }
  return success();
}

Attribute ExecutableObjectsAttr::parse(AsmParser &p, Type type) {
  // `<{` target = [objects, ...], ... `}>`
  SmallVector<Attribute> targetAttrs;
  SmallVector<Attribute> objectsAttrs;
  if (failed(p.parseLess()))
    return {};
  if (succeeded(p.parseLBrace()) && !succeeded(p.parseOptionalRBrace())) {
    do {
      Attribute targetAttr;
      ArrayAttr objectsAttr;
      if (failed(p.parseAttribute(targetAttr)) || failed(p.parseEqual()) ||
          failed(p.parseAttribute(objectsAttr))) {
        return {};
      }
      targetAttrs.push_back(targetAttr);
      objectsAttrs.push_back(objectsAttr);
    } while (succeeded(p.parseOptionalComma()));
    if (failed(p.parseRBrace()))
      return {};
  }
  if (failed(p.parseGreater()))
    return {};
  return get(p.getContext(), ArrayAttr::get(p.getContext(), targetAttrs),
             ArrayAttr::get(p.getContext(), objectsAttrs));
}

void ExecutableObjectsAttr::print(AsmPrinter &p) const {
  auto &os = p.getStream();
  os << "<{";
  llvm::interleaveComma(llvm::zip_equal(getTargets(), getTargetObjects()), os,
                        [&](std::tuple<Attribute, Attribute> keyValue) {
                          p.printAttribute(std::get<0>(keyValue));
                          os << " = ";
                          p.printAttributeWithoutType(std::get<1>(keyValue));
                        });
  os << "}>";
}

} // namespace mlir::iree_compiler::IREE::Input
