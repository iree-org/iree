// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/AttributeUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::iree_compiler {

/// Parse a list of integer values and/or dynamic values ('?')
FailureOr<SmallVector<int64_t>> parseDynamicI64IntegerList(AsmParser &parser) {
  SmallVector<int64_t> integerVals;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&] {
        int64_t value = ShapedType::kDynamic;
        if (failed(parser.parseOptionalQuestion()) &&
            failed(parser.parseInteger(value))) {
          return failure();
        }
        integerVals.push_back(value);
        return success();
      }))) {
    return failure();
  }
  return integerVals;
}

/// Print a list of integer values and/or dynamic values ('?')
void printDynamicI64IntegerList(AsmPrinter &printer, ArrayRef<int64_t> vals) {
  printer << "[";
  llvm::interleaveComma(vals, printer, [&](int64_t val) {
    if (ShapedType::isDynamic(val)) {
      printer << "?";
    } else {
      printer << val;
    }
  });
  printer << "]";
}

/// Parse a list of integer values and/or dynamic values ('?') into an ArrayAttr
ParseResult parseDynamicI64ArrayAttr(AsmParser &parser, ArrayAttr &attr) {
  FailureOr<SmallVector<int64_t>> integerVals =
      parseDynamicI64IntegerList(parser);
  if (failed(integerVals)) {
    return failure();
  }
  auto integerValsAttr =
      llvm::map_to_vector(integerVals.value(), [&](int64_t val) -> Attribute {
        return IntegerAttr::get(IntegerType::get(parser.getContext(), 64), val);
      });
  attr = ArrayAttr::get(parser.getContext(), integerValsAttr);
  return success();
}

/// Print an ArrayAttr of integer values and/or dynamic values ('?')
void printDynamicI64ArrayAttr(AsmPrinter &printer, ArrayAttr attrs) {
  SmallVector<int64_t> intVals = llvm::map_to_vector(
      attrs, [&](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });
  return printDynamicI64IntegerList(printer, intVals);
}

/// Parse a list of integer values and/or dynamic values ('?') into a
/// DenseI64ArrayAttr
ParseResult parseDynamicI64DenseArrayAttr(AsmParser &parser,
                                          DenseI64ArrayAttr &attr) {
  FailureOr<SmallVector<int64_t>> integerVals =
      parseDynamicI64IntegerList(parser);
  if (failed(integerVals)) {
    return failure();
  }
  attr = DenseI64ArrayAttr::get(parser.getContext(), *integerVals);
  return success();
}

/// Print a DenseI64ArrayAttr as a list of integer values and/or dynamic values
/// ('?')
void printDynamicI64DenseArrayAttr(AsmPrinter &printer,
                                   DenseI64ArrayAttr attr) {
  printDynamicI64IntegerList(printer, attr.asArrayRef());
}

} // namespace mlir::iree_compiler
