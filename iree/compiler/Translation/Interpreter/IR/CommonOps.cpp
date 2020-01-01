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

#include "iree/compiler/Translation/Interpreter/IR/CommonOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREEInterp {

//===----------------------------------------------------------------------===//
// iree_interp.constant
//===----------------------------------------------------------------------===//

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  Type type;
  if (parser.parseLSquare() ||
      parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  return parser.addTypeToList(type, result.types);
}

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << "iree_interp.constant[";
  p.printAttribute(op.getValue());
  p << "] ";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});

  p << " : ";
  p.printType(op.getType());
}

namespace {

// TODO(gcmn) this is duplicated from MemRefUtils to avoid a circular
// dependency. Extract op-dependent parts of memref utils to allow reuse.
MemRefType convertTypeToMemRef(Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return MemRefType::get({}, type, {}, 0);
  } else if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return MemRefType::get(memRefType.getShape(), memRefType.getElementType());
  } else {
    llvm_unreachable("Unconvertable type");
  }
}

}  // namespace

void ConstantOp::build(Builder *builder, OperationState &state,
                       ElementsAttr value) {
  auto type = convertTypeToMemRef(value.getType());
  return build(builder, state, type, value);
}

// TODO(b/134575149): enable folder when we store the correct type.
// OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
//   assert(operands.empty() && "constant has no operands");
//   return getValue();
// }

//===----------------------------------------------------------------------===//
// iree_interp.tensor_to_memref
//===----------------------------------------------------------------------===//

static ParseResult parseTensorToMemRefOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  Type resultType;
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(operand)) ||
      failed(parser.parseColonType(operandType)) ||
      failed(parser.resolveOperand(operand, operandType, state.operands)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.addTypeToList(resultType, state.types))) {
    return failure();
  }
  return success();
}

static void printTensorToMemRefOp(OpAsmPrinter &p, TensorToMemRefOp &op) {
  p << "iree_interp.tensor_to_memref(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult TensorToMemRefOp::fold(ArrayRef<Attribute> operands) {
  if (auto memrefToTensorOp = dyn_cast_or_null<IREEInterp::MemRefToTensorOp>(
          getOperand()->getDefiningOp())) {
    return memrefToTensorOp.getOperand();
  }

  return {};
}

void TensorToMemRefOp::build(Builder *builder, OperationState &state,
                             Value arg) {
  build(builder, state, convertTypeToMemRef(arg->getType()), arg);
}

//===----------------------------------------------------------------------===//
// iree_interp.memref_to_tensor
//===----------------------------------------------------------------------===//

static ParseResult parseMemRefToTensorOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  Type resultType;
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(operand)) ||
      failed(parser.parseColonType(operandType)) ||
      failed(parser.resolveOperand(operand, operandType, state.operands)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.addTypeToList(resultType, state.types))) {
    return failure();
  }
  return success();
}

static void printMemRefToTensorOp(OpAsmPrinter &p, MemRefToTensorOp &op) {
  p << "iree_interp.memref_to_tensor(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult MemRefToTensorOp::fold(ArrayRef<Attribute> operands) {
  if (auto tensorToMemRefOp = dyn_cast_or_null<IREEInterp::TensorToMemRefOp>(
          getOperand()->getDefiningOp())) {
    return tensorToMemRefOp.getOperand();
  }

  return {};
}

void MemRefToTensorOp::build(Builder *builder, OperationState &state,
                             Value arg) {
  // TODO(gcmn) Use getTensorType from MemRefUtils when circular dependency can
  // be avoided.
  auto memRefType = arg->getType().cast<MemRefType>();
  auto tensorType =
      RankedTensorType::get(memRefType.getShape(), memRefType.getElementType());
  build(builder, state, tensorType, arg);
}

//===----------------------------------------------------------------------===//
// iree_interp.scalar_to_memref
//===----------------------------------------------------------------------===//

static ParseResult parseScalarToMemRefOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  Type resultType;
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(operand)) ||
      failed(parser.parseColonType(operandType)) ||
      failed(parser.resolveOperand(operand, operandType, state.operands)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.addTypeToList(resultType, state.types))) {
    return failure();
  }
  return success();
}

static void printScalarToMemRefOp(OpAsmPrinter &p, ScalarToMemRefOp &op) {
  p << "iree_interp.scalar_to_memref(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult ScalarToMemRefOp::fold(ArrayRef<Attribute> operands) {
  if (auto memrefToScalarOp = dyn_cast_or_null<IREEInterp::MemRefToScalarOp>(
          getOperand()->getDefiningOp())) {
    return memrefToScalarOp.getOperand();
  }

  return {};
}

void ScalarToMemRefOp::build(Builder *builder, OperationState &state,
                             Value arg) {
  build(builder, state, convertTypeToMemRef(arg->getType()), arg);
}

//===----------------------------------------------------------------------===//
// iree_interp.memref_to_scalar
//===----------------------------------------------------------------------===//

static ParseResult parseMemRefToScalarOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  Type resultType;
  if (failed(parser.parseLParen()) || failed(parser.parseOperand(operand)) ||
      failed(parser.parseColonType(operandType)) ||
      failed(parser.resolveOperand(operand, operandType, state.operands)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseColonType(resultType)) ||
      failed(parser.addTypeToList(resultType, state.types))) {
    return failure();
  }
  return success();
}

static void printMemRefToScalarOp(OpAsmPrinter &p, MemRefToScalarOp &op) {
  p << "iree_interp.memref_to_scalar(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult MemRefToScalarOp::fold(ArrayRef<Attribute> operands) {
  if (auto scalarToMemRefOp = dyn_cast_or_null<IREEInterp::ScalarToMemRefOp>(
          getOperand()->getDefiningOp())) {
    return scalarToMemRefOp.getOperand();
  }

  return {};
}

void MemRefToScalarOp::build(Builder *builder, OperationState &state,
                             Value arg) {
  build(builder, state, getElementTypeOrSelf(arg), arg);
}

#define GET_OP_CLASSES
#include "iree/compiler/Translation/Interpreter/IR/CommonOps.cpp.inc"

}  // namespace IREEInterp
}  // namespace iree_compiler
}  // namespace mlir
