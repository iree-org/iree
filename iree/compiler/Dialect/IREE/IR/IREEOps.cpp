// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

//===----------------------------------------------------------------------===//
// util.do_not_optimize
//===----------------------------------------------------------------------===//

void DoNotOptimizeOp::build(OpBuilder &builder, OperationState &state,
                            ValueRange operands,
                            ArrayRef<NamedAttribute> attributes) {
  state.addOperands(operands);
  state.addTypes(llvm::to_vector<2>(operands.getTypes()));
  state.addAttributes(attributes);
}

ParseResult parseDoNotOptimizeOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> args;
  // Operands and results have the same types.
  auto &operandTypes = state.types;

  if (failed(parser.parseLParen()) || failed(parser.parseOperandList(args)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseOptionalAttrDict(state.attributes)) ||
      failed(parser.parseOptionalColonTypeList(state.types)) ||
      failed(parser.resolveOperands(
          args, operandTypes, parser.getCurrentLocation(), state.operands))) {
    return failure();
  }

  return success();
}

void printDoNotOptimizeOp(OpAsmPrinter &p, Operation *op) {
  p << "util.do_not_optimize";
  p << "(";
  p.printOperands(op->getOperands());
  p << ")";
  p.printOptionalAttrDict(op->getAttrs());

  if (op->getNumOperands() != 0) {
    p << " : ";
    interleaveComma(op->getOperandTypes(), p);
  }
}

static LogicalResult verifyDoNotOptimizeOp(DoNotOptimizeOp op) {
  if (op.getNumOperands() != op.getNumResults()) {
    return op.emitOpError()
           << "must have same number of operands and results, but has "
           << op.getNumOperands() << " and " << op.getNumResults()
           << ", respectively";
  }

  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    if (op.getOperand(i).getType() != op.getResult(i).getType()) {
      op.emitOpError() << "must have same operand and result types, but they "
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

ParseResult parseUnfoldableConstantOp(OpAsmParser &parser,
                                      OperationState &state) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(state.attributes) ||
      parser.parseAttribute(valueAttr, "value", state.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, state.types);
}

void printUnfoldableConstantOp(OpAsmPrinter &p, Operation *op) {
  auto constOp = cast<IREE::UnfoldableConstantOp>(op);
  p << "util.unfoldable_constant ";
  p.printOptionalAttrDict(constOp->getAttrs(), /*elidedAttrs=*/{"value"});

  if (constOp->getAttrs().size() > 1) p << ' ';
  p << constOp.value();

  // If the value is a symbol reference, print a trailing type.
  if (constOp.value().isa<SymbolRefAttr>()) p << " : " << constOp.getType();
}

namespace {

struct ExpandUnfoldableConstantOp
    : public OpRewritePattern<UnfoldableConstantOp> {
  using OpRewritePattern<IREE::UnfoldableConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnfoldableConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto stdConst = rewriter.create<ConstantOp>(op.getLoc(), op.value());
    rewriter.replaceOpWithNewOp<DoNotOptimizeOp>(op, stdConst.getResult());
    return success();
  }
};

}  // namespace

void UnfoldableConstantOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ExpandUnfoldableConstantOp>(context);
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

static ParseResult parseListTypeGet(OpAsmParser &parser, Type &listType,
                                    Type &elementType) {
  if (failed(parser.parseType(listType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected !util.list<T> type");
  }
  auto listElementType = listType.cast<ListType>().getElementType();
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
  auto listElementType = listType.cast<ListType>().getElementType();
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
    if (failed(parser.parseType(listType)) || !listType.isa<ListType>()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
  } else {
    if (!leadingType.isa<ListType>()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
    listType = leadingType;
    elementType = listType.cast<ListType>().getElementType();
  }
  return success();
}

static void printListTypeSet(OpAsmPrinter &printer, Operation *, Type listType,
                             Type elementType) {
  auto listElementType = listType.cast<ListType>().getElementType();
  if (listElementType != elementType) {
    printer.printType(elementType);
    printer.printArrowTypeList(ArrayRef<Type>{listType});
  } else {
    printer.printType(listType);
  }
}

static LogicalResult verifyListGetOp(ListGetOp &op) {
  auto listType = op.list().getType().cast<IREE::ListType>();
  auto elementType = listType.getElementType();
  auto resultType = op.result().getType();
  if (!ListType::canImplicitlyCast(elementType, resultType)) {
    return op.emitError() << "list contains " << elementType
                          << " and cannot be accessed as " << resultType;
  }
  return success();
}

static LogicalResult verifyListSetOp(ListSetOp &op) {
  auto listType = op.list().getType().cast<IREE::ListType>();
  auto elementType = listType.getElementType();
  auto valueType = op.value().getType();
  if (!ListType::canImplicitlyCast(valueType, elementType)) {
    return op.emitError() << "list contains " << elementType
                          << " and cannot be mutated as " << valueType;
  }
  return success();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/IREE/IR/IREEOps.cpp.inc"
