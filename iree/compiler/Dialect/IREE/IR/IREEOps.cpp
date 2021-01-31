// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
// iree.do_not_optimize
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
  p << "iree.do_not_optimize";
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
// iree.unfoldable_constant
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
  p << "iree.unfoldable_constant ";
  p.printOptionalAttrDict(constOp.getAttrs(), /*elidedAttrs=*/{"value"});

  if (constOp.getAttrs().size() > 1) p << ' ';
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

static ParseResult parseListType(OpAsmParser &parser, Type &listType,
                                 Type &elementType) {
  if (failed(parser.parseType(listType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected !iree.list<> type");
  }
  elementType = listType.cast<ListType>().getElementType();
  return success();
}

static ParseResult parseListType(OpAsmParser &parser, Type &listType,
                                 SmallVectorImpl<Type> &elementTypes) {
  if (failed(parser.parseType(listType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected !iree.list<> type");
  }
  for (size_t i = 0; i < elementTypes.size(); ++i) {
    elementTypes[i] = listType.cast<ListType>().getElementType();
  }
  return success();
}

static void printListType(OpAsmPrinter &printer, Operation *, Type listType,
                          Type elementType) {
  printer.printType(listType);
}

static void printListType(OpAsmPrinter &printer, Operation *, Type listType,
                          TypeRange elementTypes) {
  printer.printType(listType);
}

static LogicalResult verifyListGetOp(ListGetOp &op) {
  auto listType = op.list().getType().cast<IREE::ListType>();
  auto elementType = listType.getElementType();
  auto resultType = op.result().getType();
  if (resultType != elementType) {
    return op.emitError() << "list contains " << elementType
                          << " and cannot be accessed as " << resultType;
  }
  return success();
}

static LogicalResult verifyListSetOp(ListSetOp &op) {
  auto listType = op.list().getType().cast<IREE::ListType>();
  auto elementType = listType.getElementType();
  auto valueType = op.value().getType();
  if (valueType != elementType) {
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
