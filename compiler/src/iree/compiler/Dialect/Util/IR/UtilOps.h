// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Experimental
//===----------------------------------------------------------------------===//

// NOTE: this is a placeholder for a util.tree_switch (or something) op that
// looks like scf.index_switch but with a region per case. For now we emit a
// sequence of arith.select ops and return the index of the first condition that
// is true. Would be nicer with some range template magic instead of an index.
// Returns an index of -1 if no case matches.
Value buildIfElseTree(
    Location loc, size_t count,
    std::function<Value(Location, size_t, OpBuilder &)> caseBuilder,
    OpBuilder &builder);

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Removes duplicate attributes in the array (if any).
ArrayAttr deduplicateArrayElements(ArrayAttr arrayAttr);

// Finds the operand index in |operands| that |tiedResult| references.
// Returns TiedOpInterface::kUntiedIndex if no operand is found.
int64_t findTiedOperand(OpAsmParser::UnresolvedOperand tiedResult,
                        ArrayRef<OpAsmParser::UnresolvedOperand> operands);

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr);
void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                           StringAttr symVisibilityAttr);

//===----------------------------------------------------------------------===//
// custom<TypeOrAttr>($type, $attr)
//===----------------------------------------------------------------------===//
// some.op custom<TypeOrAttr>($type, $attr)
// ->
// some.op : i32
// some.op = 42 : i32
// some.op : i32 = 42 : index

ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                            Attribute &attr);
void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                     Attribute attr);

//===----------------------------------------------------------------------===//
// custom<SymbolAlias>($sym_name, $alias)
//===----------------------------------------------------------------------===//
//  @foo            sym_name: @foo, alias: @foo
//  @foo as("bar")  sym_name: @bar, alias: @foo

ParseResult parseSymbolAlias(OpAsmParser &parser, StringAttr &sym_name,
                             FlatSymbolRefAttr &alias);
void printSymbolAlias(OpAsmPrinter &p, Operation *op, StringAttr sym_name,
                      FlatSymbolRefAttr alias);

//===----------------------------------------------------------------------===//
// custom<TypeAlias>($encoding_type, $storage_type)
//===----------------------------------------------------------------------===//
// some.op custom<TypeAlias>($encoding_type, $storage_type)
// ->
// some.op tensor<4xf32>
// some.op tensor<4xf32> as tensor<2xf64>
// some.op tensor<4xf32> as tensor<?xf32>{...}

ParseResult parseTypeAlias(OpAsmParser &parser, TypeAttr &encodingTypeAttr,
                           Type &storageType);
void printTypeAlias(OpAsmPrinter &p, Operation *op, TypeAttr encodingTypeAttr,
                    Type storageType);

//===----------------------------------------------------------------------===//
// custom<SizeAwareType>
//===----------------------------------------------------------------------===//
// type{%size}

ParseResult parseSizeAwareType(OpAsmParser &parser, Type &type,
                               OpAsmParser::UnresolvedOperand &size);
void printSizeAwareType(OpAsmPrinter &p, Operation *op, Type type, Value size);

//===----------------------------------------------------------------------===//
// custom<OperandTypeList>
//===----------------------------------------------------------------------===//
// ()
// (type, type)

ParseResult parseOperandTypeList(OpAsmParser &parser,
                                 SmallVectorImpl<Type> &operandTypes);
void printOperandTypeList(OpAsmPrinter &p, Operation *op,
                          TypeRange operandTypes);

//===----------------------------------------------------------------------===//
// custom<TiedResultList>
//===----------------------------------------------------------------------===//
// type, %operand4

ParseResult
parseTiedResultList(OpAsmParser &parser,
                    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                    TypeRange operandTypes, SmallVectorImpl<Type> &resultTypes,
                    ArrayAttr &tiedOperands);
void printTiedResultList(OpAsmPrinter &p, Operation *op, ValueRange operands,
                         TypeRange operandTypes, TypeRange resultTypes,
                         ArrayAttr tiedOperands);

//===----------------------------------------------------------------------===//
// custom<TiedFunctionResultList>
//===----------------------------------------------------------------------===//
// ()
// type
// (type, %operand0, %operand1 as type)

ParseResult parseTiedFunctionResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    ArrayRef<Type> operandTypes, SmallVectorImpl<Type> &resultTypes,
    ArrayAttr &tiedOperands);
void printTiedFunctionResultList(OpAsmPrinter &p, Operation *op,
                                 ValueRange operands, TypeRange operandTypes,
                                 TypeRange resultTypes, ArrayAttr tiedOperands);

//===----------------------------------------------------------------------===//
// custom<ShapedTiedResult>
//===----------------------------------------------------------------------===//
// type{%dim0, %dim1}
// %arg0 as type{%dim0}

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims);
inline ParseResult
parseShapedTiedResult(OpAsmParser &parser, Type &resultType,
                      OpAsmParser::UnresolvedOperand &resultDim) {
  SmallVector<OpAsmParser::UnresolvedOperand, 1> resultDims;
  if (failed(parseShapedTiedResult(parser, resultType, resultDims))) {
    return failure();
  }
  assert(resultDims.size() == 1 && "requires one dim");
  resultDim = std::move(resultDims.front());
  return success();
}
void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims);

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &resultDims,
    ArrayAttr &tiedOperands);
void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims, ArrayAttr tiedOperands);

inline ParseResult
parseShapedTiedResult(OpAsmParser &parser, Type &resultType,
                      OpAsmParser::UnresolvedOperand &resultDim,
                      ArrayAttr &tiedOperands) {
  SmallVector<OpAsmParser::UnresolvedOperand> resultDims;
  if (failed(parseShapedTiedResult(parser, resultType, resultDims,
                                   tiedOperands))) {
    return failure();
  }
  assert(resultDims.size() == 1 && "requires one dim");
  resultDim = std::move(resultDims.front());
  return success();
}
inline void printShapedTiedResult(OpAsmPrinter &p, Operation *op,
                                  Type resultType, Value resultDim,
                                  ArrayAttr tiedOperands) {
  printShapedTiedResult(p, op, resultType, ValueRange{resultDim}, tiedOperands);
}
//===----------------------------------------------------------------------===//
// custom<ShapedTypeList>
//===----------------------------------------------------------------------===//
// i32, type{%size}, type{%dim0, %dim1}

ParseResult
parseShapedTypeList(OpAsmParser &parser, SmallVectorImpl<Type> &types,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dims);
void printShapedTypeList(OpAsmPrinter &p, Operation *op, TypeRange types,
                         ValueRange dims);
ParseResult
parseShapedTypeList(OpAsmParser &parser, SmallVectorImpl<Type> &types0,
                    SmallVectorImpl<Type> &types1,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dims);
void printShapedTypeList(OpAsmPrinter &p, Operation *op, TypeRange types0,
                         TypeRange types1, ValueRange dims);

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
    ArrayAttr &tiedOperands);
void printShapedResultList(OpAsmPrinter &p, Operation *op, ValueRange operands,
                           TypeRange operandTypes, ValueRange operandDims,
                           TypeRange resultTypes, ValueRange resultDims,
                           ArrayAttr tiedOperands);

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
    ArrayAttr &tiedOperands);
void printShapedFunctionType(OpAsmPrinter &p, Operation *op,
                             ValueRange operands, TypeRange operandTypes,
                             OperandRange operandDims, TypeRange resultTypes,
                             OperandRange resultDims, ArrayAttr tiedOperands);

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionSignature>
//===----------------------------------------------------------------------===//
// (%arg0: type {some.attr = 54 : index}, %arg1: type) -> (type, %arg1 as type)

ParseResult parseShapedFunctionSignature(OpAsmParser &parser,
                                         TypeAttr &functionTypeAttr,
                                         ArrayAttr &tiedOperands,
                                         ArrayAttr &argAttrs,
                                         ArrayAttr &resultAttrs);
void printShapedFunctionSignature(OpAsmPrinter &p, Operation *op,
                                  TypeAttr functionTypeAttr,
                                  ArrayAttr tiedOperands, ArrayAttr argAttrs,
                                  ArrayAttr resultAttrs);

} // namespace mlir::iree_compiler

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_
