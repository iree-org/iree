// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Experimental
//===----------------------------------------------------------------------===//

// For now we emit all cases and then select the first found (by selecting
// in reverse). So if selecting between case0, case1, and case2 we'd end up with
//   %case0 = ...
//   %case1 = ...
//   %case2 = ...
//   %0 = arith.select %case2, %c2, %c-1
//   %1 = arith.select %case1, %c1, %0
//   %2 = arith.select %case0, %c0, %1
//   // %2 is now -1 if nothing matched or the index of the match
Value buildIfElseTree(
    Location loc, size_t count,
    std::function<Value(Location, size_t, OpBuilder &)> caseBuilder,
    OpBuilder &builder) {
  SmallVector<Value> caseValues;
  caseValues.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    caseValues.push_back(caseBuilder(loc, i, builder));
  }
  Value result = builder.create<arith::ConstantIndexOp>(loc, -1);
  for (int i = count - 1; i >= 0; --i) {
    result = builder.create<arith::SelectOp>(
        loc, caseValues[i], builder.create<arith::ConstantIndexOp>(loc, i),
        result);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

ArrayAttr deduplicateArrayElements(ArrayAttr arrayAttr) {
  SetVector<Attribute> attrsSet(arrayAttr.begin(), arrayAttr.end());
  if (attrsSet.size() == arrayAttr.size())
    return arrayAttr;
  return ArrayAttr::get(arrayAttr.getContext(), attrsSet.takeVector());
}

// Finds the operand index in |operands| that |tiedResult| references.
// Returns TiedOpInterface::kUntiedIndex if no operand is found.
static int64_t
findTiedOperand(OpAsmParser::UnresolvedOperand tiedResult,
                ArrayRef<OpAsmParser::UnresolvedOperand> operands) {
  int64_t operandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
  for (int64_t i = 0; i < operands.size(); ++i) {
    if (operands[i].name == tiedResult.name &&
        operands[i].number == tiedResult.number) {
      operandIndex = i;
      break;
    }
  }
  return operandIndex;
}

static int64_t findTiedArgument(OpAsmParser::UnresolvedOperand tiedResult,
                                ArrayRef<OpAsmParser::Argument> arguments) {
  int64_t operandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
  for (int64_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i].ssaName.name == tiedResult.name &&
        arguments[i].ssaName.number == tiedResult.number) {
      operandIndex = i;
      break;
    }
  }
  return operandIndex;
}

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr) {
  StringRef symVisibility;
  if (succeeded(parser.parseOptionalKeyword(&symVisibility,
                                            {"public", "private", "nested"}))) {
    symVisibilityAttr = parser.getBuilder().getStringAttr(symVisibility);
  }
  return success();
}

void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
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

ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                            Attribute &attr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }

    if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
      typeAttr = TypeAttr::get(typedAttr.getType());
    }
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

void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                     Attribute attr) {
  bool needsSpace = false;
  auto typedAttr = dyn_cast_if_present<TypedAttr>(attr);
  if (!typedAttr || typedAttr.getType() != type.getValue()) {
    p << ": ";
    p.printAttribute(type);
    needsSpace = true; // subsequent attr value needs a space separator
  }
  if (attr) {
    if (needsSpace)
      p << ' ';
    p << "= ";
    p.printAttribute(attr);
  }
}

//===----------------------------------------------------------------------===//
// custom<SymbolAlias>($sym_name, $alias)
//===----------------------------------------------------------------------===//
//  @foo            sym_name: @foo, alias: @foo
//  @foo as("bar")  sym_name: @bar, alias: @foo

ParseResult parseSymbolAlias(OpAsmParser &parser, StringAttr &sym_name,
                             FlatSymbolRefAttr &alias) {
  if (failed(parser.parseAttribute(alias))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("as"))) {
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(sym_name)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    sym_name = StringAttr::get(parser.getContext(), alias.getValue());
  }
  return success();
}

void printSymbolAlias(OpAsmPrinter &p, Operation *op, StringAttr sym_name,
                      FlatSymbolRefAttr alias) {
  p.printAttributeWithoutType(alias);
  if (sym_name.getValue() != alias.getValue()) {
    p << " as(\"" << sym_name.getValue() << "\")";
  }
}

//===----------------------------------------------------------------------===//
// custom<TypeAlias>($encoding_type, $storage_type)
//===----------------------------------------------------------------------===//
// tensor<4xf32>
// tensor<4xf32> as tensor<2xf64>

ParseResult parseTypeAlias(OpAsmParser &parser, TypeAttr &encodingTypeAttr,
                           Type &storageType) {
  Type encodingType;
  if (failed(parser.parseType(encodingType)))
    return failure();
  storageType = encodingType;
  if (succeeded(parser.parseOptionalKeyword("as"))) {
    if (failed(parser.parseType(storageType)))
      return failure();
  }
  encodingTypeAttr = TypeAttr::get(encodingType);
  return success();
}

void printTypeAlias(OpAsmPrinter &p, Operation *op, TypeAttr encodingTypeAttr,
                    Type storageType) {
  if (encodingTypeAttr.getValue() != storageType) {
    p.printType(encodingTypeAttr.getValue());
    p << " as ";
  }
  p.printType(storageType);
}

//===----------------------------------------------------------------------===//
// custom<TypedValueList>(ref($type_value), $values)
//===----------------------------------------------------------------------===//

ParseResult
parseTypedValueList(OpAsmParser &parser, Type type,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
                    SmallVectorImpl<Type> &valueTypes) {
  if (failed(parser.parseOperandList(values, AsmParser::Delimiter::Square))) {
    return failure();
  }
  valueTypes.append(values.size(), type);
  return success();
}

void printTypedValueList(OpAsmPrinter &p, Operation *op, Type type,
                         OperandRange values, TypeRange valueTypes) {
  p << "[";
  p.printOperands(values);
  p << "]";
}

//===----------------------------------------------------------------------===//
// custom<RangeList>($offsets, $lengths)
//===----------------------------------------------------------------------===//
// [%offset for %length], [%offset for %length], ...

ParseResult
parseRangeList(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &lengths) {
  do {
    OpAsmParser::UnresolvedOperand offset;
    OpAsmParser::UnresolvedOperand length;
    if (failed(parser.parseLSquare()) || failed(parser.parseOperand(offset)) ||
        failed(parser.parseKeyword("for")) ||
        failed(parser.parseOperand(length)) || failed(parser.parseRSquare())) {
      return failure();
    }
    offsets.push_back(offset);
    lengths.push_back(length);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

void printRangeList(OpAsmPrinter &p, Operation *op, OperandRange offsets,
                    OperandRange lengths) {
  llvm::interleaveComma(llvm::zip_equal(offsets, lengths), p, [&](auto it) {
    auto offset = std::get<0>(it);
    auto length = std::get<1>(it);
    p << "[";
    p.printOperand(offset);
    p << " for ";
    p.printOperand(length);
    p << "]";
  });
}

//===----------------------------------------------------------------------===//
// custom<SizeAwareType>
//===----------------------------------------------------------------------===//
// type{%size}

ParseResult parseSizeAwareType(OpAsmParser &parser, Type &type,
                               OpAsmParser::UnresolvedOperand &size) {
  if (failed(parser.parseType(type)) || failed(parser.parseLBrace()) ||
      failed(parser.parseOperand(size)) || failed(parser.parseRBrace())) {
    return failure();
  }
  return success();
}

void printSizeAwareType(OpAsmPrinter &p, Operation *op, Type type, Value size) {
  p.printType(type);
  p << "{";
  p.printOperand(size);
  p << "}";
}

//===----------------------------------------------------------------------===//
// custom<OperandTypeList>
//===----------------------------------------------------------------------===//
// ()
// (type, type)

ParseResult parseOperandTypeList(OpAsmParser &parser,
                                 SmallVectorImpl<Type> &operandTypes) {
  if (failed(parser.parseLParen()))
    return failure();
  while (!succeeded(parser.parseOptionalRParen())) {
    Type type;
    if (failed(parser.parseType(type)))
      return failure();
    operandTypes.push_back(type);
  }
  return success();
}

void printOperandTypeList(OpAsmPrinter &p, Operation *op,
                          TypeRange operandTypes) {
  p << '(';
  llvm::interleaveComma(operandTypes, p.getStream());
  p << ')';
}

//===----------------------------------------------------------------------===//
// custom<TiedResultList>
//===----------------------------------------------------------------------===//
// type, %operand0, %operand1 as type

ParseResult
parseTiedResultList(OpAsmParser &parser,
                    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                    TypeRange operandTypes, SmallVectorImpl<Type> &resultTypes,
                    ArrayAttr &tiedOperands) {
  SmallVector<int64_t> tiedOperandIndices;
  do {
    OpAsmParser::UnresolvedOperand tiedResult;
    auto res = parser.parseOptionalOperand(tiedResult);
    Type type;
    int64_t tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
    if (res.has_value() && succeeded(res.value())) {
      tiedOperandIndex = findTiedOperand(tiedResult, operands);
      if (tiedOperandIndex == IREE::Util::TiedOpInterface::kUntiedIndex) {
        return parser.emitError(tiedResult.location,
                                "tied operand not found for result reference ")
               << tiedResult.name;
      }
      if (succeeded(parser.parseOptionalKeyword("as"))) {
        // Type _may_ differ from the operand.
        if (failed(parser.parseType(type)))
          return failure();
      } else {
        // Use the operands type.
        type = operandTypes[tiedOperandIndex];
      }
    } else if (failed(parser.parseType(type))) {
      return failure();
    }
    resultTypes.push_back(type);
    tiedOperandIndices.push_back(tiedOperandIndex);
  } while (succeeded(parser.parseOptionalComma()));
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

void printTiedResultList(OpAsmPrinter &p, Operation *op, ValueRange operands,
                         TypeRange operandTypes, TypeRange resultTypes,
                         ArrayAttr tiedOperands) {
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    auto resultType = resultTypes[i];
    auto tiedOperandIndex =
        tiedOp ? tiedOp.getTiedResultOperandIndex(i) : std::nullopt;
    bool printType = true;
    if (tiedOperandIndex.has_value()) {
      auto tiedOperand = op->getOperand(tiedOperandIndex.value());
      p.printOperand(tiedOperand);
      if (tiedOperand.getType() != resultType) {
        p << " as ";
      } else {
        // Type elided as it matches the operand.
        printType = false;
      }
    }
    if (printType) {
      p.printType(resultType);
    }
    if (i < resultTypes.size() - 1)
      p << ", ";
  }
}

//===----------------------------------------------------------------------===//
// custom<TiedFunctionResultList>
//===----------------------------------------------------------------------===//
// ()
// type
// (type, %operand0 {some.attr}, %operand1 as type)

static ParseResult
parseTiedFunctionResultListImpl(OpAsmParser &parser,
                                ArrayRef<OpAsmParser::Argument> arguments,
                                SmallVectorImpl<Type> &resultTypes,
                                SmallVectorImpl<DictionaryAttr> &resultAttrs,
                                ArrayAttr &tiedOperands, bool allowAttrs) {
  SmallVector<int64_t> tiedOperandIndices;
  do {
    OpAsmParser::UnresolvedOperand tiedResult;
    auto res = parser.parseOptionalOperand(tiedResult);
    Type type;
    int64_t tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
    if (res.has_value() && succeeded(res.value())) {
      tiedOperandIndex = findTiedArgument(tiedResult, arguments);
      if (tiedOperandIndex == IREE::Util::TiedOpInterface::kUntiedIndex) {
        return parser.emitError(tiedResult.location,
                                "tied operand not found for result reference ")
               << tiedResult.name;
      }
      if (succeeded(parser.parseOptionalKeyword("as"))) {
        // Type _may_ differ from the operand.
        if (failed(parser.parseType(type)))
          return failure();
      } else {
        // Use the operands type.
        type = arguments[tiedOperandIndex].type;
      }
    } else if (failed(parser.parseType(type))) {
      return failure();
    }
    DictionaryAttr resultAttrDict;
    if (allowAttrs) {
      NamedAttrList resultAttrList;
      if (succeeded(parser.parseOptionalAttrDict(resultAttrList))) {
        resultAttrDict = parser.getBuilder().getDictionaryAttr(resultAttrList);
      }
    }
    resultTypes.push_back(type);
    resultAttrs.push_back(resultAttrDict);
    tiedOperandIndices.push_back(tiedOperandIndex);
  } while (succeeded(parser.parseOptionalComma()));
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

static ParseResult parseTiedFunctionResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::Argument> arguments,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<DictionaryAttr> &resultAttrs, ArrayAttr &tiedOperands) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandTypes;
  operands.reserve(arguments.size());
  operandTypes.reserve(arguments.size());
  for (auto argument : arguments) {
    operands.push_back(argument.ssaName);
    operandTypes.push_back(argument.type);
  }
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalRParen())) {
      // Empty list/no results `()`.
    } else {
      // One or more result types.
      if (failed(parseTiedFunctionResultListImpl(parser, arguments, resultTypes,
                                                 resultAttrs, tiedOperands,
                                                 /*allowAttrs=*/true)) ||
          failed(parser.parseRParen())) {
        return failure();
      }
    }
  } else {
    // Single result with omitted `()`.
    if (failed(parseTiedFunctionResultListImpl(parser, arguments, resultTypes,
                                               resultAttrs, tiedOperands,
                                               /*allowAttrs=*/false))) {
      return failure();
    }
  }
  return success();
}

ParseResult parseTiedFunctionResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    ArrayRef<Type> operandTypes, SmallVectorImpl<Type> &resultTypes,
    ArrayAttr &tiedOperands) {
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalRParen())) {
      // Empty list/no results `()`.
    } else {
      // One or more result types.
      if (failed(parseTiedResultList(parser, operands, operandTypes,
                                     resultTypes, tiedOperands)) ||
          failed(parser.parseRParen())) {
        return failure();
      }
    }
  } else {
    // Single result with omitted `()`.
    if (failed(parseTiedResultList(parser, operands, operandTypes, resultTypes,
                                   tiedOperands))) {
      return failure();
    }
  }
  return success();
}

void printTiedFunctionResultList(OpAsmPrinter &p, Operation *op,
                                 ValueRange operands, TypeRange operandTypes,
                                 TypeRange resultTypes,
                                 ArrayAttr tiedOperands) {
  if (resultTypes.size() != 1)
    p << "(";
  printTiedResultList(p, op, operands, operandTypes, resultTypes, tiedOperands);
  if (resultTypes.size() != 1)
    p << ")";
}

//===----------------------------------------------------------------------===//
// custom<ShapedTypeList>
//===----------------------------------------------------------------------===//
// type{%size0}, type, type{%size1}

ParseResult
parseShapedTypeList(OpAsmParser &parser, SmallVectorImpl<Type> &types,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dims) {
  do {
    Type type;
    if (failed(parser.parseType(type)))
      return failure();
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      if (!shapedType.hasStaticShape()) {
        SmallVector<OpAsmParser::UnresolvedOperand> dynamicDims;
        if (failed(parser.parseLBrace()) ||
            failed(parser.parseOperandList(dynamicDims,
                                           shapedType.getNumDynamicDims(),
                                           OpAsmParser::Delimiter::None)) ||
            failed(parser.parseRBrace())) {
          return failure();
        }
        dims.append(dynamicDims);
      }
    } else if (isa<IREE::Util::SizeAwareTypeInterface>(type)) {
      OpAsmParser::UnresolvedOperand size;
      if (failed(parser.parseLBrace()) || failed(parser.parseOperand(size)) ||
          failed(parser.parseRBrace())) {
        return failure();
      }
      dims.push_back(size);
    }
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

void printShapedTypeList(OpAsmPrinter &p, Operation *op, TypeRange types,
                         ValueRange dims) {
  llvm::interleaveComma(types, p, [&](Type type) {
    p.printType(type);
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      if (!shapedType.hasStaticShape()) {
        if (dims.empty()) {
          p << "{<<INVALID>>}";
          return;
        }
        p << "{";
        llvm::interleaveComma(dims.take_front(shapedType.getNumDynamicDims()),
                              p, [&](Value value) { p.printOperand(value); });
        p << "}";
        dims = dims.drop_front(shapedType.getNumDynamicDims());
      }
    } else if (isa<IREE::Util::SizeAwareTypeInterface>(type)) {
      p << "{";
      p.printOperand(dims.front());
      p << "}";
      dims = dims.drop_front(1);
    }
  });
}

ParseResult
parseShapedTypeList(OpAsmParser &parser, SmallVectorImpl<Type> &types0,
                    SmallVectorImpl<Type> &types1,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dims) {
  if (failed(parseShapedTypeList(parser, types0, dims)))
    return failure();
  types1 = types0;
  return success();
}

void printShapedTypeList(OpAsmPrinter &p, Operation *op, TypeRange types0,
                         TypeRange types1, ValueRange dims) {
  printShapedTypeList(p, op, types0, dims);
}

//===----------------------------------------------------------------------===//
// custom<ShapedTiedResult>
//===----------------------------------------------------------------------===//
// type{%dim0, %dim1}
// %arg0 as type{%dim0}

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims) {
  ArrayAttr tiedOperands;
  return parseShapedTiedResult(parser, resultType, resultDims, tiedOperands);
}

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims,
    ArrayAttr &tiedOperands) {
  OpAsmParser::UnresolvedOperand tiedResult;
  auto res = parser.parseOptionalOperand(tiedResult);
  int64_t tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
  if (res.has_value() && succeeded(res.value())) {
    tiedOperandIndex = 0;
    if (failed(parser.parseKeyword("as")))
      return failure();
  }
  if (failed(parser.parseType(resultType)))
    return failure();
  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
    if (!shapedType.hasStaticShape()) {
      SmallVector<OpAsmParser::UnresolvedOperand> dynamicDims;
      if (failed(parser.parseLBrace()) ||
          failed(parser.parseOperandList(dynamicDims,
                                         shapedType.getNumDynamicDims(),
                                         OpAsmParser::Delimiter::None)) ||
          failed(parser.parseRBrace())) {
        return failure();
      }
      resultDims.append(dynamicDims);
    }
  } else if (auto sizedType =
                 dyn_cast<IREE::Util::SizeAwareTypeInterface>(resultType)) {
    OpAsmParser::UnresolvedOperand size;
    if (failed(parser.parseLBrace()) || failed(parser.parseOperand(size)) ||
        failed(parser.parseRBrace())) {
      return failure();
    }
    resultDims.push_back(size);
  }
  tiedOperands = parser.getBuilder().getIndexArrayAttr({tiedOperandIndex});
  return success();
}

void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims) {
  auto tiedOp = cast<IREE::Util::TiedOpInterface>(op);
  auto tiedOperandIndex = tiedOp.getTiedResultOperandIndex(0);
  if (tiedOperandIndex.has_value()) {
    auto tiedOperand = op->getOperand(tiedOperandIndex.value());
    p.printOperand(tiedOperand);
    p << " as ";
  }
  p.printType(resultType);
  if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
    if (!shapedType.hasStaticShape()) {
      if (resultDims.empty()) {
        p << "{<<INVALID>>}";
        return;
      }
      p << "{";
      llvm::interleaveComma(
          resultDims.take_front(shapedType.getNumDynamicDims()), p,
          [&](Value value) { p.printOperand(value); });
      p << "}";
      resultDims = resultDims.drop_front(shapedType.getNumDynamicDims());
    }
  } else if (auto sizedType =
                 dyn_cast<IREE::Util::SizeAwareTypeInterface>(resultType)) {
    p << "{";
    p.printOperand(resultDims.front());
    p << "}";
    resultDims = resultDims.drop_front(1);
  }
}

void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims, ArrayAttr tiedOperands) {
  printShapedTiedResult(p, op, resultType, resultDims);
}

//===----------------------------------------------------------------------===//
// custom<ShapedResultList>
//===----------------------------------------------------------------------===//
// type{%dim2}, %operand4

ParseResult parseShapedResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    TypeRange operandTypes,
    ArrayRef<OpAsmParser::UnresolvedOperand> operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims,
    ArrayAttr &tiedOperands) {
  SmallVector<int64_t> tiedOperandIndices;
  do {
    OpAsmParser::UnresolvedOperand tiedResult;
    auto res = parser.parseOptionalOperand(tiedResult);
    Type type;
    int64_t tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
    if (res.has_value() && succeeded(res.value())) {
      tiedOperandIndex = findTiedOperand(tiedResult, operands);
      if (tiedOperandIndex == IREE::Util::TiedOpInterface::kUntiedIndex) {
        return parser.emitError(tiedResult.location,
                                "tied operand not found for result reference ")
               << tiedResult.name;
      }
      if (succeeded(parser.parseOptionalKeyword("as"))) {
        // Type _may_ differ from the operand.
        if (failed(parser.parseType(type)))
          return failure();
      } else {
        // Use the operands type.
        type = operandTypes[tiedOperandIndex];
      }
    } else if (failed(parser.parseType(type))) {
      return failure();
    }
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      if (!shapedType.hasStaticShape()) {
        SmallVector<OpAsmParser::UnresolvedOperand> dynamicDims;
        if (failed(parser.parseLBrace()) ||
            failed(parser.parseOperandList(dynamicDims,
                                           shapedType.getNumDynamicDims(),
                                           OpAsmParser::Delimiter::None)) ||
            failed(parser.parseRBrace())) {
          return failure();
        }
        resultDims.append(dynamicDims);
      }
    } else if (auto sizedType =
                   dyn_cast<IREE::Util::SizeAwareTypeInterface>(type)) {
      OpAsmParser::UnresolvedOperand size;
      if (failed(parser.parseLBrace()) || failed(parser.parseOperand(size)) ||
          failed(parser.parseRBrace())) {
        return failure();
      }
      resultDims.push_back(size);
    }
    resultTypes.push_back(type);
    tiedOperandIndices.push_back(tiedOperandIndex);
  } while (succeeded(parser.parseOptionalComma()));
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

void printShapedResultList(OpAsmPrinter &p, Operation *op, ValueRange operands,
                           TypeRange operandTypes, ValueRange operandDims,
                           TypeRange resultTypes, ValueRange resultDims,
                           ArrayAttr tiedOperands) {
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    auto resultType = resultTypes[i];
    auto tiedOperandIndex =
        tiedOp ? tiedOp.getTiedResultOperandIndex(i) : std::nullopt;
    bool printType = true;
    if (tiedOperandIndex.has_value()) {
      auto tiedOperand = op->getOperand(tiedOperandIndex.value());
      p.printOperand(tiedOperand);
      if (tiedOperand.getType() != resultType) {
        p << " as ";
      } else {
        // Type elided as it matches the operand.
        printType = false;
      }
    }
    if (printType) {
      p.printType(resultType);
    }
    if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
      if (!shapedType.hasStaticShape()) {
        if (resultDims.empty()) {
          p << "{<<INVALID>>}";
          return;
        }
        p << "{";
        llvm::interleaveComma(
            resultDims.take_front(shapedType.getNumDynamicDims()), p,
            [&](Value value) { p.printOperand(value); });
        p << "}";
        resultDims = resultDims.drop_front(shapedType.getNumDynamicDims());
      }
    } else if (auto sizedType =
                   dyn_cast<IREE::Util::SizeAwareTypeInterface>(resultType)) {
      p << "{";
      p.printOperand(resultDims.front());
      p << "}";
      resultDims = resultDims.drop_front(1);
    }
    if (i < resultTypes.size() - 1)
      p << ", ";
  }
}

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionType>
//===----------------------------------------------------------------------===//
// (type, type{%dim0, %dim1}, type) -> (type{%dim2}, %operand4)

ParseResult parseShapedFunctionType(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims,
    ArrayAttr &tiedOperands) {
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parseShapedTypeList(parser, operandTypes, operandDims)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }
  if (failed(parser.parseArrow()))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (succeeded(parser.parseOptionalRParen())) {
      // Empty list/no results `()`.
    } else {
      // One or more result types.
      if (failed(parseShapedResultList(parser, operands, operandTypes,
                                       operandDims, resultTypes, resultDims,
                                       tiedOperands)) ||
          failed(parser.parseRParen())) {
        return failure();
      }
    }
  } else {
    // Single result with omitted `()`.
    if (failed(parseShapedResultList(parser, operands, operandTypes,
                                     operandDims, resultTypes, resultDims,
                                     tiedOperands))) {
      return failure();
    }
  }
  return success();
}

void printShapedFunctionType(OpAsmPrinter &p, Operation *op,
                             ValueRange operands, TypeRange operandTypes,
                             OperandRange operandDims, TypeRange resultTypes,
                             OperandRange resultDims, ArrayAttr tiedOperands) {
  p << "(";
  printShapedTypeList(p, op, operandTypes, operandDims);
  p << ") -> ";
  if (resultTypes.size() != 1)
    p << "(";
  printShapedResultList(p, op, operands, operandTypes, operandDims, resultTypes,
                        resultDims, tiedOperands);
  if (resultTypes.size() != 1)
    p << ")";
}

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionSignature>
//===----------------------------------------------------------------------===//
// (%arg0: type {some.attr = 54 : index}, %arg1: type) -> (type, %arg1 as type)

static ParseResult parseShapedFunctionArgumentList(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &args,
    SmallVectorImpl<Type> &types, ArrayAttr &attrs) {
  SmallVector<Attribute> argAttrsVec;
  do {
    OpAsmParser::UnresolvedOperand arg;
    Type type;
    NamedAttrList attrsVec;
    if (failed(parser.parseOperand(arg)) ||
        failed(parser.parseColonType(type)) ||
        failed(parser.parseOptionalAttrDict(attrsVec))) {
      return failure();
    }
    args.push_back(arg);
    types.push_back(type);
    argAttrsVec.push_back(parser.getBuilder().getDictionaryAttr(attrsVec));
  } while (succeeded(parser.parseOptionalComma()));
  if (!argAttrsVec.empty()) {
    attrs = parser.getBuilder().getArrayAttr(argAttrsVec);
  }
  return success();
}

static ParseResult parseShapedFunctionResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> args,
    TypeRange argTypes, SmallVectorImpl<Type> &resultTypes,
    ArrayAttr &resultAttrs, ArrayAttr &tiedOperands, bool allowResultAttrs) {
  SmallVector<Attribute> resultAttrsVec;
  SmallVector<int64_t> tiedOperandIndices;
  do {
    OpAsmParser::UnresolvedOperand tiedResult;
    auto res = parser.parseOptionalOperand(tiedResult);
    Type type;
    int64_t tiedOperandIndex = IREE::Util::TiedOpInterface::kUntiedIndex;
    if (res.has_value() && succeeded(res.value())) {
      tiedOperandIndex = findTiedOperand(tiedResult, args);
      if (tiedOperandIndex == IREE::Util::TiedOpInterface::kUntiedIndex) {
        return parser.emitError(tiedResult.location,
                                "tied operand not found for result reference ")
               << tiedResult.name;
      }
      if (succeeded(parser.parseOptionalKeyword("as"))) {
        // Type _may_ differ from the operand.
        if (failed(parser.parseType(type)))
          return failure();
      } else {
        // Use the operands type.
        type = argTypes[tiedOperandIndex];
      }
    } else if (failed(parser.parseType(type))) {
      return failure();
    }
    NamedAttrList attrs;
    if (allowResultAttrs && failed(parser.parseOptionalAttrDict(attrs))) {
      return failure();
    }
    resultTypes.push_back(type);
    resultAttrsVec.push_back(parser.getBuilder().getDictionaryAttr(attrs));
    tiedOperandIndices.push_back(tiedOperandIndex);
  } while (succeeded(parser.parseOptionalComma()));
  if (!resultAttrsVec.empty()) {
    resultAttrs = parser.getBuilder().getArrayAttr(resultAttrsVec);
  }
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

static void printShapedFunctionResultList(OpAsmPrinter &p, Operation *op,
                                          TypeRange argTypes,
                                          TypeRange resultTypes,
                                          ArrayAttr resultAttrs,
                                          ArrayAttr tiedOperands) {
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    auto resultType = resultTypes[i];
    auto tiedOperandIndex =
        IREE::Util::detail::getTiedResultOperandIndex(op, i);
    bool printType = true;
    if (tiedOperandIndex.has_value()) {
      p << "%arg" << tiedOperandIndex.value();
      if (argTypes[tiedOperandIndex.value()] != resultType) {
        p << " as ";
      } else {
        // Type elided as it matches the operand.
        printType = false;
      }
    }
    if (printType) {
      p.printType(resultType);
    }
    if (resultAttrs) {
      auto attrs =
          dyn_cast_if_present<DictionaryAttr>(resultAttrs.getValue()[i]);
      if (attrs && !attrs.empty()) {
        p.printOptionalAttrDict(attrs.getValue());
      }
    }
    if (i < resultTypes.size() - 1)
      p << ", ";
  }
}

ParseResult parseShapedFunctionSignature(OpAsmParser &parser,
                                         TypeAttr &functionTypeAttr,
                                         ArrayAttr &tiedOperands,
                                         ArrayAttr &argAttrs,
                                         ArrayAttr &resultAttrs) {
  SmallVector<OpAsmParser::UnresolvedOperand> args;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parseShapedFunctionArgumentList(parser, args, argTypes,
                                               argAttrs)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }
  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parseShapedFunctionResultList(
              parser, args, argTypes, resultTypes, resultAttrs, tiedOperands,
              /*allowResultAttrs=*/true)) ||
          failed(parser.parseRParen())) {
        return failure();
      }
    } else {
      if (failed(parseShapedFunctionResultList(
              parser, args, argTypes, resultTypes, resultAttrs, tiedOperands,
              /*allowResultAttrs=*/false))) {
        return failure();
      }
    }
  }
  functionTypeAttr = TypeAttr::get(
      FunctionType::get(parser.getContext(), argTypes, resultTypes));
  return success();
}

void printShapedFunctionSignature(OpAsmPrinter &p, Operation *op,
                                  TypeAttr functionTypeAttr,
                                  ArrayAttr tiedOperands, ArrayAttr argAttrs,
                                  ArrayAttr resultAttrs) {
  auto functionType = cast<FunctionType>(functionTypeAttr.getValue());
  p << "(";
  int argIndex = 0;
  llvm::interleaveComma(functionType.getInputs(), p, [&](auto type) {
    p << "%arg";
    p << argIndex;
    p << ": ";
    p.printType(type);
    if (argAttrs) {
      auto attrs =
          dyn_cast_if_present<DictionaryAttr>(argAttrs.getValue()[argIndex]);
      if (attrs && !attrs.empty())
        p.printOptionalAttrDict(attrs.getValue());
    }
    ++argIndex;
  });
  p << ")";
  auto resultTypes = functionType.getResults();
  if (!resultTypes.empty()) {
    p << " -> ";
    bool anyResultAttrs =
        resultAttrs && !resultAttrs.empty() &&
        llvm::any_of(resultAttrs.getAsValueRange<DictionaryAttr>(),
                     [](auto attr) { return !attr.empty(); });
    if (resultTypes.size() != 1 || anyResultAttrs)
      p << "(";
    printShapedFunctionResultList(p, op, functionType.getInputs(), resultTypes,
                                  resultAttrs, tiedOperands);
    if (resultTypes.size() != 1 || anyResultAttrs)
      p << ")";
  }
}

} // namespace mlir::iree_compiler

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// util.optimization_barrier
//===----------------------------------------------------------------------===//

void OptimizationBarrierOp::build(OpBuilder &builder, OperationState &state,
                                  ValueRange operands,
                                  ArrayRef<NamedAttribute> attributes) {
  state.addOperands(operands);
  state.addTypes(llvm::to_vector<2>(operands.getTypes()));
  state.addAttributes(attributes);
}

LogicalResult OptimizationBarrierOp::verify() {
  Operation *op = getOperation();
  if (op->getNumOperands() != op->getNumResults()) {
    return op->emitOpError()
           << "must have same number of operands and results, but has "
           << op->getNumOperands() << " and " << op->getNumResults()
           << ", respectively";
  }

  for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i).getType() != op->getResult(i).getType()) {
      op->emitOpError() << "must have same operand and result types, but they "
                           "differ at index "
                        << i;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// util.unfoldable_constant
//===----------------------------------------------------------------------===//

// Parsing/printing copied from std.constant

ParseResult UnfoldableConstantOp::parse(OpAsmParser &parser,
                                        OperationState &state) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseAttribute(valueAttr, "value", state.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!isa<SymbolRefAttr>(valueAttr))
    type = cast<TypedAttr>(valueAttr).getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, state.types);
}

void UnfoldableConstantOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1)
    p << ' ';
  p << getValue();

  // If the value is a symbol reference, print a trailing type.
  if (isa<SymbolRefAttr>(getValue()))
    p << " : " << getType();
}

//===----------------------------------------------------------------------===//
// Type manipulation
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  if (a == b) {
    // Both types are the same.
    return true;
  }
  if (isa<IREE::Util::ObjectType>(a) || isa<IREE::Util::ObjectType>(b)) {
    // Either type is an opaque object.
    return true;
  }
  // Don't currently allow casting between types as we don't have runtime
  // support for such operations (we don't generally care in the VM).
  return false;
}

LogicalResult CastOp::verify() {
  auto operandType = getOperand().getType();
  if (!IREE::Util::ObjectType::isCompatible(operandType)) {
    return this->emitOpError() << "operand type " << operandType
                               << " is not object cast compatible";
  }
  auto resultType = getResult().getType();
  if (!IREE::Util::ObjectType::isCompatible(resultType)) {
    return this->emitOpError()
           << "result type " << resultType << " is not object cast compatible";
  }
  return success();
}

Value CastOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getOperand());
}

Value CastOp::getTiedResultOperand(Value result) { return getOperand(); }

::std::optional<unsigned>
CastOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // operand
}

SmallVector<int64_t> CastOp::getTiedResultOperandIndices() {
  return {0}; // operand
}

//===----------------------------------------------------------------------===//
// Numeric ops
//===----------------------------------------------------------------------===//

std::optional<std::pair<int64_t, int64_t>>
NumericOptionalNarrowOp::getIntegerRange() {
  if (!getMinValue() || !getMaxValue())
    return {};
  bool signExtend = isSigned();
  // Note: Cannot sign extend 0 bit values.
  int64_t minValue = signExtend && getMinValue()->getBitWidth() > 0
                         ? getMinValue()->getSExtValue()
                         : getMinValue()->getZExtValue();
  int64_t maxValue = signExtend && getMaxValue()->getBitWidth() > 0
                         ? getMaxValue()->getSExtValue()
                         : getMaxValue()->getZExtValue();
  return std::make_pair(minValue, maxValue);
}

//===----------------------------------------------------------------------===//
// util.initializer
//===----------------------------------------------------------------------===//

void InitializerOp::build(OpBuilder &builder, OperationState &result,
                          ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("function_type", TypeAttr::get(FunctionType::get(
                                           builder.getContext(), {}, {})));
  result.addRegion();
  result.attributes.append(attrs.begin(), attrs.end());
}

ParseResult InitializerOp::parse(OpAsmParser &parser, OperationState &result) {
  result.addAttribute("function_type", TypeAttr::get(FunctionType::get(
                                           result.getContext(), {}, {})));
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }
  auto &body = *result.addRegion();
  if (failed(parser.parseRegion(body))) {
    return failure();
  }
  return success();
}

void InitializerOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{"function_type"});
  p << " ";
  p.printRegion(getBody());
}

Block *InitializerOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  return entry;
}

Block *InitializerOp::addBlock() {
  assert(!empty() && "function should at least have an entry block");
  push_back(new Block());
  return &back();
}

//===----------------------------------------------------------------------===//
// util.func
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<int64_t> tiedOperands,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs,
                      ArrayRef<DictionaryAttr> resAttrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type,
                builder.getIndexArrayAttr(tiedOperands), attrs, argAttrs,
                resAttrs);
  return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayAttr tiedOperands,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs,
                   ArrayRef<DictionaryAttr> resAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(SymbolTable::getVisibilityAttrName(),
                     builder.getStringAttr("private"));
  state.addAttribute("function_type", TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.attributes.erase(IREE::Util::TiedOpInterface::getStorageAttrName());
  state.addAttribute(IREE::Util::TiedOpInterface::getStorageAttrName(),
                     tiedOperands);
  state.addRegion();
  if (!argAttrs.empty() || !resAttrs.empty()) {
    assert(type.getNumInputs() == argAttrs.size());
    assert(type.getNumResults() == resAttrs.size());
    function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, resAttrs, builder.getStringAttr("arg_attrs"),
        builder.getStringAttr("res_attrs"));
  }
}

static ParseResult
parseFunctionArgumentList(OpAsmParser &parser,
                          SmallVectorImpl<OpAsmParser::Argument> &arguments) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        auto argPresent = parser.parseOptionalArgument(
            argument, /*allowType=*/true, /*allowAttrs=*/true);
        if (argPresent.has_value()) {
          if (failed(argPresent.value()))
            return failure(); // Present but malformed.
          if (!arguments.empty() && arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected type instead of SSA identifier");

        } else {
          argument.ssaName.location = parser.getCurrentLocation();
          if (!arguments.empty() && !arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected SSA identifier");
          NamedAttrList attrs;
          if (parser.parseType(argument.type) ||
              parser.parseOptionalAttrDict(attrs) ||
              parser.parseOptionalLocationSpecifier(argument.sourceLoc))
            return failure();
          argument.attrs = attrs.getDictionary(parser.getContext());
        }
        arguments.push_back(argument);
        return success();
      });
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  StringAttr symVisibilityAttr;
  if (failed(parseSymbolVisibility(parser, symVisibilityAttr)))
    return failure();
  if (symVisibilityAttr)
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        symVisibilityAttr);

  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> arguments;
  if (parseFunctionArgumentList(parser, arguments))
    return failure();

  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  ArrayAttr tiedOperands;
  if (succeeded(parser.parseOptionalArrow())) {
    if (failed(parseTiedFunctionResultList(parser, arguments, resultTypes,
                                           resultAttrs, tiedOperands)))
      return failure();
  }
  if (tiedOperands)
    result.addAttribute("tied_operands", tiedOperands);

  SmallVector<Type> argumentTypes;
  for (auto argument : arguments)
    argumentTypes.push_back(argument.type);
  result.addAttribute("function_type", TypeAttr::get(builder.getFunctionType(
                                           argumentTypes, resultTypes)));

  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();
  for (StringRef disallowed : {
           SymbolTable::getVisibilityAttrName(),
           SymbolTable::getSymbolAttrName(),
           StringRef("function_type"),
       }) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  assert(resultAttrs.size() == resultTypes.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, result, arguments, resultAttrs,
      builder.getStringAttr("arg_attrs"), builder.getStringAttr("res_attrs"));

  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  auto parseResult = parser.parseOptionalRegion(*body, arguments,
                                                /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  printSymbolVisibility(p, *this, getSymVisibilityAttr());
  p << ' ';
  p.printSymbolName(getSymName());
  printShapedFunctionSignature(p, *this, getFunctionTypeAttr(),
                               getTiedOperandsAttr(), getArgAttrsAttr(),
                               getResAttrsAttr());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     /*elidedAttrs=*/{
                                         "sym_name",
                                         "function_type",
                                         "tied_operands",
                                         "sym_visibility",
                                         "arg_attrs",
                                         "res_attrs",
                                     });
  if (!getBody().empty()) {
    p << ' ';
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  }
}

//===----------------------------------------------------------------------===//
// util.call
//===----------------------------------------------------------------------===//

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();

  // Only support calls to util.func.
  auto calleeOp = symbolTable.lookupNearestSymbolFrom<IREE::Util::FuncOp>(
      op, getCalleeAttr());
  if (!calleeOp) {
    return op->emitOpError("undefined/incompatible callee: ") << getCallee();
  }

  // Ensure that the arg/result types match.
  auto expectedType = getCalleeType();
  auto calleeType = calleeOp.getFunctionType();
  if (calleeType != expectedType) {
    return emitOpError("function type mismatch; expected ")
           << expectedType << " but callee is " << calleeType;
  }

  // Ensure tied operands are consistent.
  auto expectedTiedOperands = getTiedOperandsAttr();
  auto calleeTiedOperands = calleeOp.getTiedOperandsAttr();
  if (calleeTiedOperands != expectedTiedOperands) {
    return emitOpError("function tied operands mismatch; expected ")
           << expectedTiedOperands << " but callee is " << calleeTiedOperands;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// util.return
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *op = getOperation();
  auto parentOp = cast<mlir::FunctionOpInterface>(op->getParentOp());
  auto expectedTypes = parentOp.getResultTypes();
  if (getNumOperands() != expectedTypes.size()) {
    return emitOpError("has ")
           << getNumOperands()
           << " operands, but enclosing function-like op returns "
           << expectedTypes.size();
  }
  for (auto [i, expectedType, actualType] :
       llvm::enumerate(expectedTypes, getOperandTypes())) {
    if (expectedType != actualType) {
      return emitOpError() << "type of return operand " << i << " ("
                           << actualType
                           << ") doesn't match function result type ("
                           << expectedType << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// util.global
//===----------------------------------------------------------------------===//

// TODO(benvanik): move entirely to the interface.
// Returns true if the given |accessType| is compatible with the |globalType|.
// For example, this will return true if the global type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isGlobalTypeCompatible(Type globalType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (isa<ShapedType>(globalType) && isa<ShapedType>(accessType)) {
    return succeeded(mlir::verifyCompatibleShape(globalType, accessType));
  }

  if (auto knownType = dyn_cast<GlobalTypeInterface>(globalType)) {
    return knownType.isAccessStorageCompatible(accessType);
  }

  // Otherwise, the types must be the same.
  return globalType == accessType;
}

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

void GlobalAddressOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), Twine("ptr_" + getGlobal()).str());
}

void GlobalLoadOp::build(OpBuilder &builder, OperationState &state,
                         IREE::Util::GlobalOpInterface globalOp,
                         ArrayRef<NamedAttribute> attrs) {
  state.addTypes({globalOp.getGlobalType()});
  state.addAttribute("global", SymbolRefAttr::get(globalOp));
  state.attributes.append(attrs.begin(), attrs.end());
}

void GlobalLoadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getGlobal());
}

void GlobalLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // HACK: works around the lack of symbol side effects in mlir by only saying
  // we have a side-effect if the variable we are loading is mutable.
  auto globalOp =
      SymbolTable::lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  assert(globalOp);
  if (globalOp.getIsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult GlobalLoadIndirectOp::verify() {
  Operation *op = getOperation();
  auto globalType =
      cast<IREE::Util::PtrType>(getGlobal().getType()).getTargetType();
  auto loadType = getResult().getType();
  if (!isGlobalTypeCompatible(globalType, loadType)) {
    return op->emitOpError() << "global type mismatch; global pointer is "
                             << globalType << " but load is " << loadType;
  }
  return success();
}

void GlobalStoreOp::build(OpBuilder &builder, OperationState &state,
                          Value value, IREE::Util::GlobalOpInterface globalOp,
                          ArrayRef<NamedAttribute> attrs) {
  state.addOperands({value});
  state.addAttribute("global", SymbolRefAttr::get(globalOp));
  state.attributes.append(attrs.begin(), attrs.end());
}

LogicalResult GlobalStoreIndirectOp::verify() {
  Operation *op = getOperation();
  auto globalType =
      cast<IREE::Util::PtrType>(getGlobal().getType()).getTargetType();
  auto storeType = getValue().getType();
  if (!isGlobalTypeCompatible(globalType, storeType)) {
    return op->emitOpError() << "global type mismatch; global pointer is "
                             << globalType << " but store is " << storeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// !util.list<T>
//===----------------------------------------------------------------------===//

static ParseResult parseListTypeGet(OpAsmParser &parser, Type &listType,
                                    Type &elementType) {
  if (failed(parser.parseType(listType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected !util.list<T> type");
  }
  auto listElementType = cast<ListType>(listType).getElementType();
  if (succeeded(parser.parseOptionalArrow())) {
    // Use overridden type - required for variants only.
    if (failed(parser.parseType(elementType))) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "expected an element type when specifying list access types");
    }
    if (!ListType::canImplicitlyCast(listElementType, elementType)) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "list access types must match the same base type as the list element "
          "type (when not variant)");
    }
  } else {
    // Use list element type as the result element type.
    elementType = listElementType;
  }
  return success();
}

static void printListTypeGet(OpAsmPrinter &printer, Operation *, Type listType,
                             Type elementType) {
  printer.printType(listType);
  auto listElementType = cast<ListType>(listType).getElementType();
  if (listElementType != elementType) {
    printer.printArrowTypeList(ArrayRef<Type>{elementType});
  }
}

static ParseResult parseListTypeSet(OpAsmParser &parser, Type &listType,
                                    Type &elementType) {
  Type leadingType;
  if (failed(parser.parseType(leadingType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected element type or !util.list<T> type");
  }
  if (succeeded(parser.parseOptionalArrow())) {
    elementType = leadingType;
    if (failed(parser.parseType(listType)) || !isa<ListType>(listType)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
  } else {
    if (!isa<ListType>(leadingType)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
    listType = leadingType;
    elementType = cast<ListType>(listType).getElementType();
  }
  return success();
}

static void printListTypeSet(OpAsmPrinter &printer, Operation *, Type listType,
                             Type elementType) {
  auto listElementType = cast<ListType>(listType).getElementType();
  if (listElementType != elementType) {
    printer.printType(elementType);
    printer.printArrowTypeList(ArrayRef<Type>{listType});
  } else {
    printer.printType(listType);
  }
}

LogicalResult ListGetOp::verify() {
  Operation *op = getOperation();
  auto listType = cast<IREE::Util::ListType>(getList().getType());
  auto elementType = listType.getElementType();
  auto resultType = getResult().getType();
  if (!ListType::canImplicitlyCast(elementType, resultType)) {
    return op->emitError() << "list contains " << elementType
                           << " and cannot be accessed as " << resultType;
  }
  return success();
}

LogicalResult ListSetOp::verify() {
  Operation *op = getOperation();
  auto listType = cast<IREE::Util::ListType>(getList().getType());
  auto elementType = listType.getElementType();
  auto valueType = getValue().getType();
  if (!ListType::canImplicitlyCast(valueType, elementType)) {
    return op->emitError() << "list contains " << elementType
                           << " and cannot be mutated as " << valueType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// !util.buffer
//===----------------------------------------------------------------------===//

void BufferConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName().value_or("buffer_cst"));
}

LogicalResult BufferConstantOp::verify() {
  if (!isa<IREE::Util::SerializableAttrInterface>(getValue())) {
    return emitOpError("unsupported non-serializable constant attribute type");
  }
  if (auto minAlignmentAttr = getAlignmentAttr()) {
    int64_t minAlignment = minAlignmentAttr.getInt();
    if (minAlignment > 0 && !llvm::isPowerOf2_64(minAlignment)) {
      return emitOpError("invalid alignment; must be a power of two");
    }
  }
  return success();
}

void BufferAllocOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

LogicalResult BufferAllocOp::verify() {
  if (auto minAlignmentAttr = getAlignmentAttr()) {
    int64_t minAlignment = minAlignmentAttr.getInt();
    if (minAlignment > 0 && !llvm::isPowerOf2_64(minAlignment)) {
      return emitOpError("invalid alignment; must be a power of two");
    }
  }
  return success();
}

void BufferSliceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

SubrangeOperand BufferSliceOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getSource(), getSourceSize(), getSourceOffset(),
                           getResultSize()};
  } else {
    assert(false && "only source is a subrange");
    return {};
  }
}

void BufferSliceOp::setSubrangeOperand(unsigned operandIndex,
                                       SubrangeOperand operand) {
  assert(operandIndex == 0 && "only source is a subrange");
  getSourceMutable().assign(operand.resource);
  getSourceSizeMutable().assign(operand.resourceSize);
  getSourceOffsetMutable().assign(operand.offset);
  getResultSizeMutable().assign(operand.length);
}

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer_span");
}

Value BufferSubspanOp::getViewSource() { return getSource(); }

Value BufferSubspanOp::getTiedResult(unsigned resultIndex) {
  return IREE::Util::TiedOpInterface::findTiedBaseValue(getSource());
}

SubrangeOperand BufferSubspanOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getSource(), getSourceSize(), getSourceOffset(),
                           getResultSize()};
  } else {
    assert(false && "only source is a subrange");
    return {};
  }
}

void BufferSubspanOp::setSubrangeOperand(unsigned operandIndex,
                                         SubrangeOperand operand) {
  assert(operandIndex == 0 && "only source is a subrange");
  getSourceMutable().assign(operand.resource);
  getSourceSizeMutable().assign(operand.resourceSize);
  getSourceOffsetMutable().assign(operand.offset);
  getResultSizeMutable().assign(operand.length);
}

::std::optional<unsigned>
BufferSubspanOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // source
}

SmallVector<int64_t> BufferSubspanOp::getTiedResultOperandIndices() {
  return {0}; // source
}

// static
IREE::Util::BufferSubspanOp BufferSubspanOp::findSubspanOp(Value value) {
  while (value) {
    auto *definingOp = value.getDefiningOp();
    if (!definingOp) {
      // Defined as a block argument - stop walk.
      break;
    } else if (auto subviewOp =
                   dyn_cast<IREE::Util::BufferSubspanOp>(definingOp)) {
      // Found!
      return subviewOp;
    } else if (auto tiedOp =
                   dyn_cast<IREE::Util::TiedOpInterface>(definingOp)) {
      // Continue walking up through the tied operand.
      value = tiedOp.getTiedResultOperand(value);
    } else {
      break;
    }
  }
  return {};
}

void BufferSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer_size");
}

void BufferStorageOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer_storage");
  setNameFn(getOffset(), "buffer_offset");
}

SubrangeOperand BufferCopyOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getSource(), getSourceSize(), getSourceOffset(),
                           getLength()};
  } else if (operandIndex == 3) {
    return SubrangeOperand{getTarget(), getTargetSize(), getTargetOffset(),
                           getLength()};
  } else {
    assert(false && "only source/target are subranges");
    return {};
  }
}

void BufferCopyOp::setSubrangeOperand(unsigned operandIndex,
                                      SubrangeOperand operand) {
  if (operandIndex == 0) {
    getSourceMutable().assign(operand.resource);
    getSourceSizeMutable().assign(operand.resourceSize);
    getSourceOffsetMutable().assign(operand.offset);
    getLengthMutable().assign(operand.length);
  } else if (operandIndex == 3) {
    getTargetMutable().assign(operand.resource);
    getTargetSizeMutable().assign(operand.resourceSize);
    getTargetOffsetMutable().assign(operand.offset);
    getLengthMutable().assign(operand.length);
  } else {
    assert(false && "only source/target are subranges");
  }
}

SubrangeOperand BufferCompareOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getLhs(), getLhsSize(), getLhsOffset(), getLength()};
  } else if (operandIndex == 3) {
    return SubrangeOperand{getRhs(), getRhsSize(), getRhsOffset(), getLength()};
  } else {
    assert(false && "only lhs/rhs are subranges");
    return {};
  }
}

void BufferCompareOp::setSubrangeOperand(unsigned operandIndex,
                                         SubrangeOperand operand) {
  if (operandIndex == 0) {
    getLhsMutable().assign(operand.resource);
    getLhsSizeMutable().assign(operand.resourceSize);
    getLhsOffsetMutable().assign(operand.offset);
    getLengthMutable().assign(operand.length);
  } else if (operandIndex == 3) {
    getRhsMutable().assign(operand.resource);
    getRhsSizeMutable().assign(operand.resourceSize);
    getRhsOffsetMutable().assign(operand.offset);
    getLengthMutable().assign(operand.length);
  } else {
    assert(false && "only lhs/rhs are subranges");
  }
}

SubrangeOperand BufferFillOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 1) {
    return SubrangeOperand{getTarget(), getTargetSize(), getTargetOffset(),
                           getLength()};
  } else {
    assert(false && "only target is a subrange");
    return {};
  }
}

void BufferFillOp::setSubrangeOperand(unsigned operandIndex,
                                      SubrangeOperand operand) {
  assert(operandIndex == 1 && "only target is a subrange");
  getTargetMutable().assign(operand.resource);
  getTargetSizeMutable().assign(operand.resourceSize);
  getTargetOffsetMutable().assign(operand.offset);
  getLengthMutable().assign(operand.length);
}

SubrangeOperand BufferLoadOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getSource(), getSourceSize(), getSourceOffset(),
                           getLength()};
  } else {
    assert(false && "only source is a subrange");
    return {};
  }
}

void BufferLoadOp::setSubrangeOperand(unsigned operandIndex,
                                      SubrangeOperand operand) {
  assert(operandIndex == 0 && "only source is a subrange");
  getSourceMutable().assign(operand.resource);
  getSourceSizeMutable().assign(operand.resourceSize);
  getSourceOffsetMutable().assign(operand.offset);
  getLengthMutable().assign(operand.length);
}

SubrangeOperand BufferStoreOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 1) {
    return SubrangeOperand{getTarget(), getTargetSize(), getTargetOffset(),
                           getLength()};
  } else {
    assert(false && "only target is a subrange");
    return {};
  }
}

void BufferStoreOp::setSubrangeOperand(unsigned operandIndex,
                                       SubrangeOperand operand) {
  assert(operandIndex == 1 && "only target is a subrange");
  getTargetMutable().assign(operand.resource);
  getTargetSizeMutable().assign(operand.resourceSize);
  getTargetOffsetMutable().assign(operand.offset);
  getLengthMutable().assign(operand.length);
}

SubrangeOperand BufferHashOp::getSubrangeOperand(unsigned operandIndex) {
  if (operandIndex == 0) {
    return SubrangeOperand{getSource(), getSourceSize(), getSourceOffset(),
                           getLength()};
  } else {
    assert(false && "only source is a subrange");
    return {};
  }
}

void BufferHashOp::setSubrangeOperand(unsigned operandIndex,
                                      SubrangeOperand operand) {
  assert(operandIndex == 0 && "only source is a subrange");
  getSourceMutable().assign(operand.resource);
  getSourceSizeMutable().assign(operand.resourceSize);
  getSourceOffsetMutable().assign(operand.offset);
  getLengthMutable().assign(operand.length);
}

} // namespace mlir::iree_compiler::IREE::Util

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilOps.cpp.inc"
