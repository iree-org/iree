// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputOps.h"

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

#include "iree-dialects/Dialect/Input/InputOpInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// IREE::Input::TiedOpInterface
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::Input::detail {

std::optional<unsigned> getTiedResultOperandIndex(Operation *op,
                                                  unsigned resultIndex) {
  auto storageAttr =
      op->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr)
    return std::nullopt;
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty())
    return std::nullopt;
  if (auto tiedOp = dyn_cast<TiedOpInterface>(op)) {
    auto indexAndLength = tiedOp.getTiedResultsIndexAndLength();
    if (resultIndex < indexAndLength.first)
      return std::nullopt;
    resultIndex -= indexAndLength.first;
    if (resultIndex >= indexAndLength.second)
      return std::nullopt;
  }
  int64_t value = llvm::cast<IntegerAttr>(valueAttrs[resultIndex]).getInt();
  if (value == TiedOpInterface::kUntiedIndex)
    return std::nullopt;
  if (auto tiedOp = dyn_cast<TiedOpInterface>(op)) {
    unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
    return tiedOperandsOffset + static_cast<unsigned>(value);
  } else {
    return static_cast<unsigned>(value);
  }
}

SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) {
  SmallVector<int64_t> indices;
  auto storageAttr =
      op->getAttrOfType<ArrayAttr>(TiedOpInterface::getStorageAttrName());
  if (!storageAttr)
    return indices;
  auto valueAttrs = storageAttr.getValue();
  if (valueAttrs.empty())
    return indices;
  auto tiedOp = cast<TiedOpInterface>(op);
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
  indices.resize(resultRange.second);
  for (unsigned i = 0; i < valueAttrs.size(); ++i) {
    int64_t index = llvm::cast<IntegerAttr>(valueAttrs[i]).getInt();
    indices[i] = index != TiedOpInterface::kUntiedIndex
                     ? tiedOperandsOffset + index
                     : TiedOpInterface::kUntiedIndex;
  }
  return indices;
}

void setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                               std::optional<unsigned> operandIndex) {
  auto tiedOp = cast<TiedOpInterface>(op);
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  resultIndex -= resultRange.first;

  auto indices = getTiedResultOperandIndices(op);
  if (indices.empty()) {
    indices.resize(resultRange.second, TiedOpInterface::kUntiedIndex);
  } else {
    // Well, getTiedResultOperandIndices() returns indices into the full range
    // of the op, but in the attribute, we expect to store ranges into the range
    // returned by `getTiedOperandsIndexAndLength`.
    unsigned tiedOperandsOffset = tiedOp.getTiedOperandsIndexAndLength().first;
    for (auto &index : indices) {
      if (index != TiedOpInterface::kUntiedIndex)
        index -= tiedOperandsOffset;
    }
  }

  indices[resultIndex] = operandIndex.value_or(TiedOpInterface::kUntiedIndex);
  op->setAttr(TiedOpInterface::getStorageAttrName(),
              Builder(op).getIndexArrayAttr(indices));
}

bool isOperandTied(Operation *op, unsigned operandIndex) {
  auto tiedOp = dyn_cast<TiedOpInterface>(op);
  if (!tiedOp)
    return false;
  auto tiedIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedIndices.size(); ++i) {
    if (tiedIndices[i] == operandIndex) {
      return true;
    }
  }
  return false;
}

SmallVector<Value> getOperandTiedResults(Operation *op, unsigned operandIndex) {
  auto tiedOp = dyn_cast<TiedOpInterface>(op);
  if (!tiedOp)
    return {};
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  SmallVector<Value> results;
  auto tiedIndices = tiedOp.getTiedResultOperandIndices();
  for (unsigned i = 0; i < tiedIndices.size(); ++i) {
    if (tiedIndices[i] == operandIndex) {
      results.push_back(op->getResult(resultRange.first + i));
    }
  }
  return results;
}

LogicalResult verifyTiedOp(TiedOpInterface tiedOp) {
  auto tiedOperandIndices = tiedOp.getTiedResultOperandIndices();
  if (tiedOperandIndices.empty())
    return success();
  auto resultRange = tiedOp.getTiedResultsIndexAndLength();
  if (tiedOperandIndices.size() != resultRange.second) {
    return tiedOp.emitError("op results/tied operand indices mismatch");
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::Input::detail

Value TiedOpInterface::findTiedBaseValue(Value derivedValue) {
  Value baseValue = derivedValue;
  while (auto definingOp =
             dyn_cast_or_null<TiedOpInterface>(baseValue.getDefiningOp())) {
    auto tiedValue = definingOp.getTiedResultOperand(baseValue);
    if (!tiedValue)
      break;
    baseValue = tiedValue;
  }
  return baseValue;
}

bool TiedOpInterface::hasAnyTiedUses(Value value) {
  return llvm::any_of(value.getUses(), [](auto &use) {
    if (auto tiedOp = dyn_cast<TiedOpInterface>(use.getOwner())) {
      return tiedOp.isOperandTied(use.getOperandNumber());
    }
    return false;
  });
}

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

static ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                         StringAttr &symVisibilityAttr) {
  StringRef symVisibility;
  if (succeeded(parser.parseOptionalKeyword(&symVisibility,
                                            {"public", "private", "nested"}))) {
    symVisibilityAttr = parser.getBuilder().getStringAttr(symVisibility);
  }
  return success();
}

static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
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

static ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                                   TypedAttr &attr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
    typeAttr = TypeAttr::get(attr.getType());
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

static void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                            TypedAttr attr) {
  if (!attr || attr.getType() != type.getValue()) {
    p << " : ";
    p.printAttribute(type);
  }
  if (attr) {
    p << " = ";
    p.printAttribute(attr);
  }
}

//===----------------------------------------------------------------------===//
// custom<ShapedTiedResult>
//===----------------------------------------------------------------------===//
// type{%dim0, %dim1}
// %arg0 as type{%dim0}

static ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims,
    ArrayAttr &tiedOperands) {
  OpAsmParser::UnresolvedOperand tiedResult;
  auto res = parser.parseOptionalOperand(tiedResult);
  int64_t tiedOperandIndex = TiedOpInterface::kUntiedIndex;
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
  }
  tiedOperands = parser.getBuilder().getIndexArrayAttr({tiedOperandIndex});
  return success();
}

static ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims) {
  ArrayAttr tiedOperands;
  return parseShapedTiedResult(parser, resultType, resultDims, tiedOperands);
}

void printShapedTiedResult(OpAsmPrinter &p, TiedOpInterface op, Type resultType,
                           ValueRange resultDims) {
  auto tiedOperandIndex = op.getTiedResultOperandIndex(0);
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
  }
}

static void printShapedTiedResult(OpAsmPrinter &p, TiedOpInterface op,
                                  Type resultType, ValueRange resultDims,
                                  ArrayAttr tiedOperands) {
  printShapedTiedResult(p, op, resultType, resultDims);
}

//===----------------------------------------------------------------------===//
// custom<ShapedOperandList>($values, type($values), $value_dims)
//===----------------------------------------------------------------------===//
// %value : type{%dynamic_dims}, ...

ParseResult parseShapedOperandList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    SmallVectorImpl<Type> &valueTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valueDims) {
  do {
    values.emplace_back();
    valueTypes.emplace_back();
    if (failed(parser.parseOperand(values.back())) ||
        failed(parser.parseColon()) ||
        failed(parser.parseType(valueTypes.back())))
      return failure();
    if (int64_t dynamicDimCount =
            cast<ShapedType>(valueTypes.back()).getNumDynamicDims()) {
      if (failed(parser.parseOperandList(valueDims, dynamicDimCount,
                                         AsmParser::Delimiter::Braces)))
        return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

void printShapedOperandList(OpAsmPrinter &p, Operation *op, ValueRange values,
                            TypeRange valueTypes, ValueRange valueDims) {
  llvm::interleaveComma(llvm::zip_equal(values, valueTypes), p, [&](auto it) {
    auto [value, valueType] = it;
    p << value;
    p << " : ";
    p << valueType;
    if (int64_t dynamicDimCount =
            cast<ShapedType>(valueType).getNumDynamicDims()) {
      p << "{";
      llvm::interleaveComma(valueDims.take_front(dynamicDimCount), p);
      valueDims = valueDims.drop_front(dynamicDimCount);
      p << "}";
    }
  });
}

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionType>
//===----------------------------------------------------------------------===//
// (type, type{%dim0, %dim1}, type) -> (type{%dim2}, %operand4)

static ParseResult
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
    }
    types.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));
  return success();
}

// Finds the operand index in |operands| that |tiedResult| references.
// Returns TiedOpInterface::kUntiedIndex if no operand is found.
static int64_t
findTiedOperand(OpAsmParser::UnresolvedOperand tiedResult,
                ArrayRef<OpAsmParser::UnresolvedOperand> operands) {
  int64_t operandIndex = TiedOpInterface::kUntiedIndex;
  for (int64_t i = 0; i < operands.size(); ++i) {
    if (operands[i].name == tiedResult.name &&
        operands[i].number == tiedResult.number) {
      operandIndex = i;
      break;
    }
  }
  return operandIndex;
}

static ParseResult parseShapedResultList(
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
    int64_t tiedOperandIndex = TiedOpInterface::kUntiedIndex;
    if (res.has_value() && succeeded(res.value())) {
      tiedOperandIndex = findTiedOperand(tiedResult, operands);
      if (tiedOperandIndex == TiedOpInterface::kUntiedIndex) {
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
    }
    resultTypes.push_back(type);
    tiedOperandIndices.push_back(tiedOperandIndex);
  } while (succeeded(parser.parseOptionalComma()));
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

static void printShapedResultList(OpAsmPrinter &p, TiedOpInterface tiedOp,
                                  ValueRange operands, TypeRange operandTypes,
                                  ValueRange operandDims, TypeRange resultTypes,
                                  ValueRange resultDims,
                                  ArrayAttr tiedOperands) {
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    auto resultType = resultTypes[i];
    auto tiedOperandIndex = tiedOp.getTiedResultOperandIndex(i);
    bool printType = true;
    if (tiedOperandIndex.has_value()) {
      auto tiedOperand = tiedOp->getOperand(tiedOperandIndex.value());
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
    }
    if (i < resultTypes.size() - 1)
      p << ", ";
  }
}

static ParseResult parseShapedFunctionType(
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

static void printShapedFunctionType(OpAsmPrinter &p, TiedOpInterface tiedOp,
                                    ValueRange operands, TypeRange operandTypes,
                                    OperandRange operandDims,
                                    TypeRange resultTypes,
                                    OperandRange resultDims,
                                    ArrayAttr tiedOperands) {
  p << "(";
  llvm::interleaveComma(operandTypes, p, [&](Type type) {
    p.printType(type);
    if (auto shapedType = dyn_cast<ShapedType>(type)) {
      if (!shapedType.hasStaticShape()) {
        if (operandDims.empty()) {
          p << "{<<INVALID>>}";
          return;
        }
        p << "{";
        llvm::interleaveComma(
            operandDims.take_front(shapedType.getNumDynamicDims()), p,
            [&](Value value) { p.printOperand(value); });
        p << "}";
        operandDims = operandDims.drop_front(shapedType.getNumDynamicDims());
      }
    }
  });
  p << ") -> ";
  if (resultTypes.size() != 1)
    p << "(";
  printShapedResultList(p, tiedOp, operands, operandTypes, operandDims,
                        resultTypes, resultDims, tiedOperands);
  if (resultTypes.size() != 1)
    p << ")";
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

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

// Returns true if the given |accessType| is compatible with the |globalType|.
// For example, this will return true if the global type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isGlobalTypeCompatible(Type globalType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (globalType.isa<ShapedType>() && accessType.isa<ShapedType>()) {
    return succeeded(mlir::verifyCompatibleShape(globalType, accessType)) &&
           globalType.cast<ShapedType>().getElementType() ==
               accessType.cast<ShapedType>().getElementType();
  }

  // Permissively allow any other types to be marked compatible as long as
  // neither are shaped type.
  return !globalType.isa<ShapedType>() && !accessType.isa<ShapedType>();
}

LogicalResult
GlobalLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  if (!globalOp) {
    return emitOpError() << "undefined global: " << getGlobal();
  }
  auto loadType = getResult().getType();
  if (!isGlobalTypeCompatible(globalOp.getType(), loadType)) {
    return emitOpError() << "global type mismatch; global " << getGlobal()
                         << " is " << globalOp.getType() << " but load is "
                         << loadType;
  }
  return success();
}

LogicalResult GlobalLoadIndirectOp::verify() {
  auto globalType = getGlobal().getType().cast<PtrType>().getTargetType();
  auto loadType = getResult().getType();
  if (!isGlobalTypeCompatible(globalType, loadType)) {
    return emitOpError() << "global type mismatch; global pointer is "
                         << globalType << " but load is " << loadType;
  }
  return success();
}

LogicalResult
GlobalStoreOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto globalOp =
      symbolTable.lookupNearestSymbolFrom<GlobalOp>(*this, getGlobalAttr());
  if (!globalOp) {
    return emitOpError() << "undefined global: " << getGlobal();
  }
  auto storeType = getValue().getType();
  if (!isGlobalTypeCompatible(globalOp.getType(), storeType)) {
    return emitOpError() << "global type mismatch; global " << getGlobal()
                         << " is " << globalOp.getType() << " but store is "
                         << storeType;
  }
  return success();
}

LogicalResult GlobalStoreIndirectOp::verify() {
  auto globalType = getGlobal().getType().cast<PtrType>().getTargetType();
  auto storeType = getValue().getType();
  if (!isGlobalTypeCompatible(globalType, storeType)) {
    return emitOpError() << "global type mismatch; global pointer is "
                         << globalType << " but store is " << storeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// !iree_input.byte_vuffer
//===----------------------------------------------------------------------===//

void ByteBufferConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName().value_or("bytes_cst"));
}

LogicalResult ByteBufferConstantOp::verify() {
  if (auto minAlignmentAttr = getAlignmentAttr()) {
    int64_t minAlignment = minAlignmentAttr.getInt();
    if (minAlignment > 0 && !llvm::isPowerOf2_64(minAlignment)) {
      return emitOpError("invalid alignment; must be a power of two");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree_input.buffer.subspan
//===----------------------------------------------------------------------===//

void BufferSubspanOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "buffer");
}

//===----------------------------------------------------------------------===//
// iree_input.buffer_view.create
//===----------------------------------------------------------------------===//

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value sourceBuffer, Value sourceOffset,
                               Value sourceLength, int32_t elementType,
                               int32_t encodingType, ValueRange shape) {
  build(builder, state, sourceBuffer, sourceOffset, sourceLength,
        builder.createOrFold<arith::ConstantIntOp>(state.location, elementType,
                                                   32),
        builder.createOrFold<arith::ConstantIntOp>(state.location, encodingType,
                                                   32),
        shape);
}

void BufferViewCreateOp::build(OpBuilder &builder, OperationState &state,
                               Value sourceBuffer, Value sourceOffset,
                               Value sourceLength, Value elementType,
                               Value encodingType, ValueRange shape) {
  state.addOperands(
      {sourceBuffer, sourceOffset, sourceLength, elementType, encodingType});
  state.addOperands(shape);
  state.addTypes({BufferViewType::get(builder.getContext())});
}

//===----------------------------------------------------------------------===//
// iree_input.tensor.update
//===----------------------------------------------------------------------===//

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return TiedOpInterface::findTiedBaseValue(getTarget());
}

std::optional<unsigned>
TensorUpdateOp::getTiedResultOperandIndex(unsigned resultIndex) {
  return {0}; // $target
}

SmallVector<int64_t> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0}; // $target
}

//===----------------------------------------------------------------------===//
// iree_input.tensor.trace
//===----------------------------------------------------------------------===//

void TensorTraceOp::build(OpBuilder &builder, OperationState &state,
                          StringRef key, ValueRange values) {
  SmallVector<Value> dynamicDims;
  for (auto value : values) {
    auto valueType = cast<ShapedType>(value.getType());
    for (unsigned i = 0; i < valueType.getRank(); ++i) {
      if (valueType.isDynamicDim(i)) {
        dynamicDims.push_back(
            builder.createOrFold<tensor::DimOp>(state.location, value, i));
      }
    }
  }
  build(builder, state, key, values, dynamicDims);
}

//===----------------------------------------------------------------------===//
// iree_input.dispatch
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &state,
                       ExecutableExportOp exportOp, ValueRange workload,
                       TypeRange resultTypes, ValueRange resultDims,
                       ValueRange operands, ValueRange operandDims,
                       ArrayAttr tiedOperands,
                       ArrayRef<NamedAttribute> attributes) {
  StringRef executableOpSymName =
      exportOp->getParentOp()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  auto entryPoint =
      SymbolRefAttr::get(builder.getContext(), executableOpSymName,
                         {SymbolRefAttr::get(exportOp)});
  state.addAttribute("entry_point", entryPoint);
  state.addOperands(workload);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(TiedOpInterface::getStorageAttrName());
  state.addAttribute(TiedOpInterface::getStorageAttrName(), tiedOperands);
  state.attributes.erase(getOperandSegmentSizeAttr());
  state.addAttribute(getOperandSegmentSizeAttr(),
                     builder.getDenseI32ArrayAttr({
                         static_cast<int32_t>(workload.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
}

LogicalResult DispatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  auto exportOp = symbolTable.lookupNearestSymbolFrom<ExecutableExportOp>(
      op, getEntryPoint());
  if (!exportOp) {
    return op->emitOpError() << "undefined entry point: " << getEntryPoint();
  }

  // TODO(ezhulenev): verify that the exported function has matching operands.
  return success();
}

std::pair<unsigned, unsigned> DispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1); // $arguments
}

//===----------------------------------------------------------------------===//
// iree_input.optimization_barrier
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

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/Input/InputOps.cpp.inc"
