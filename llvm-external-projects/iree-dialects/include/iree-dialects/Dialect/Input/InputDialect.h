// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_DIALECT_H
#define IREE_DIALECTS_DIALECT_INPUT_DIALECT_H

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// Include generated dialect code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/Input/InputDialect.h.inc"

// Include generated enums code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/Input/InputEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputTypes.h.inc"

//===----------------------------------------------------------------------===//
// IREE ABI helpers for constructing buffer views
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::Input {

// Returns a stable identifier for the MLIR element type or nullopt if the
// type is unsupported in the ABI.
std::optional<int32_t> getElementTypeValue(Type type);

// Returns a stable identifier for the MLIR encoding type or empty optional
// (opaque) if the type is unsupported in the ABI.
std::optional<int32_t> getEncodingTypeValue(Attribute attr);

} // namespace mlir::iree_compiler::IREE::Input

//===----------------------------------------------------------------------===//
// Specialize templates in mlir namespace to support enum attributes
//===----------------------------------------------------------------------===//

namespace mlir {

template <>
struct FieldParser<
    std::optional<mlir::iree_compiler::IREE::Input::DescriptorSetLayoutFlags>> {
  static FailureOr<mlir::iree_compiler::IREE::Input::DescriptorSetLayoutFlags>
  parse(AsmParser &parser) {
    std::string value;
    if (parser.parseKeywordOrString(&value))
      return failure();
    auto result = mlir::iree_compiler::IREE::Input::symbolizeEnum<
        mlir::iree_compiler::IREE::Input::DescriptorSetLayoutFlags>(value);
    if (!result.has_value())
      return failure();
    return result.value();
  }
};

static inline AsmPrinter &operator<<(
    AsmPrinter &printer,
    std::optional<mlir::iree_compiler::IREE::Input::DescriptorSetLayoutFlags>
        param) {
  printer << (param.has_value()
                  ? mlir::iree_compiler::IREE::Input::stringifyEnum(
                        param.value())
                  : StringRef{""});
  return printer;
}

template <>
struct FieldParser<
    std::optional<mlir::iree_compiler::IREE::Input::DescriptorFlags>> {
  static FailureOr<mlir::iree_compiler::IREE::Input::DescriptorFlags>
  parse(AsmParser &parser) {
    std::string value;
    if (parser.parseKeywordOrString(&value))
      return failure();
    auto result = mlir::iree_compiler::IREE::Input::symbolizeEnum<
        mlir::iree_compiler::IREE::Input::DescriptorFlags>(value);
    if (!result.has_value())
      return failure();
    return result.value();
  }
};

static inline AsmPrinter &operator<<(
    AsmPrinter &printer,
    std::optional<mlir::iree_compiler::IREE::Input::DescriptorFlags> param) {
  printer << (param.has_value()
                  ? mlir::iree_compiler::IREE::Input::stringifyEnum(
                        param.value())
                  : StringRef{""});
  return printer;
}

static inline AsmPrinter &
operator<<(AsmPrinter &printer,
           mlir::iree_compiler::IREE::Input::DescriptorType param) {
  printer << mlir::iree_compiler::IREE::Input::stringifyEnum(param);
  return printer;
}

} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_INPUT_DIALECT_H
