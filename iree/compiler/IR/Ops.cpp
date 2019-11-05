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

#include "iree/compiler/IR/Ops.h"

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
namespace IREE {

//===----------------------------------------------------------------------===//
// iree.constant
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
  p << "iree.constant[";
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
// iree.tensor_to_memref
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
  p << "iree.tensor_to_memref(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult TensorToMemRefOp::fold(ArrayRef<Attribute> operands) {
  if (auto memrefToTensorOp = dyn_cast_or_null<IREE::MemRefToTensorOp>(
          getOperand()->getDefiningOp())) {
    return memrefToTensorOp.getOperand();
  }

  return {};
}

void TensorToMemRefOp::build(Builder *builder, OperationState &state,
                             Value *arg) {
  build(builder, state, convertTypeToMemRef(arg->getType()), arg);
}

//===----------------------------------------------------------------------===//
// iree.memref_to_tensor
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
  p << "iree.memref_to_tensor(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult MemRefToTensorOp::fold(ArrayRef<Attribute> operands) {
  if (auto tensorToMemRefOp = dyn_cast_or_null<IREE::TensorToMemRefOp>(
          getOperand()->getDefiningOp())) {
    return tensorToMemRefOp.getOperand();
  }

  return {};
}

void MemRefToTensorOp::build(Builder *builder, OperationState &state,
                             Value *arg) {
  // TODO(gcmn) Use getTensorType from MemRefUtils when circular dependency can
  // be avoided.
  auto memRefType = arg->getType().cast<MemRefType>();
  auto tensorType =
      RankedTensorType::get(memRefType.getShape(), memRefType.getElementType());
  build(builder, state, tensorType, arg);
}

//===----------------------------------------------------------------------===//
// iree.scalar_to_memref
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
  p << "iree.scalar_to_memref(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult ScalarToMemRefOp::fold(ArrayRef<Attribute> operands) {
  if (auto memrefToScalarOp = dyn_cast_or_null<IREE::MemRefToScalarOp>(
          getOperand()->getDefiningOp())) {
    return memrefToScalarOp.getOperand();
  }

  return {};
}

void ScalarToMemRefOp::build(Builder *builder, OperationState &state,
                             Value *arg) {
  build(builder, state, convertTypeToMemRef(arg->getType()), arg);
}

//===----------------------------------------------------------------------===//
// iree.memref_to_scalar
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
  p << "iree.memref_to_scalar(";
  p.printOperand(op.getOperand());
  p << " : ";
  p.printType(op.getOperand()->getType());
  p << ") : ";
  p.printType(op.getType());
}

OpFoldResult MemRefToScalarOp::fold(ArrayRef<Attribute> operands) {
  if (auto scalarToMemRefOp = dyn_cast_or_null<IREE::ScalarToMemRefOp>(
          getOperand()->getDefiningOp())) {
    return scalarToMemRefOp.getOperand();
  }

  return {};
}

void MemRefToScalarOp::build(Builder *builder, OperationState &state,
                             Value *arg) {
  build(builder, state, getElementTypeOrSelf(arg), arg);
}

//===----------------------------------------------------------------------===//
// iree.dispatch_region
//===----------------------------------------------------------------------===//

void DispatchRegionOp::build(Builder *builder, OperationState &state,
                             ArrayRef<Type> resultTypes, Value *workload,
                             ArrayRef<Value *> operands,
                             ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands({workload});
  state.addOperands(operands);
  state.addAttributes(attributes);
  state.addRegion();
  state.setOperandListToResizable();
}

ParseResult parseDispatchRegionOp(OpAsmParser &parser, OperationState &state) {
  // Parse required workload.
  OpAsmParser::OperandType workloadArg;
  Type workloadArgType;
  if (failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workloadArg)) ||
      failed(parser.parseColonType(workloadArgType)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperand(workloadArg, workloadArgType,
                                   state.operands))) {
    return failure();
  }

  // Parse (optional) args.
  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
  SmallVector<Type, 16> regionArgTypes;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    SmallVector<OpAsmParser::OperandType, 16> regionOperands;
    auto argsLoc = parser.getCurrentLocation();
    do {
      // Reserve entries in the lists.
      regionArgs.emplace_back();
      regionOperands.emplace_back();
      regionArgTypes.emplace_back();
      if (failed(parser.parseRegionArgument(regionArgs.back())) ||
          failed(parser.parseEqual()) ||
          failed(parser.parseOperand(regionOperands.back())) ||
          failed(parser.parseColonType(regionArgTypes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen()) ||
        failed(parser.resolveOperands(regionOperands, regionArgTypes, argsLoc,
                                      state.operands))) {
      return failure();
    }
  }
  state.setOperandListToResizable();

  // Parse (optional) results.
  if (failed(parser.parseOptionalColonTypeList(state.types))) {
    return failure();
  }

  // Parse region body.
  Region *body = state.addRegion();
  if (failed(parser.parseRegion(*body, regionArgs, regionArgTypes)) ||
      failed(parser.parseOptionalAttrDict(state.attributes))) {
    return failure();
  }
  return success();
}

void printDispatchRegionOp(OpAsmPrinter &p, DispatchRegionOp op) {
  p << "iree.dispatch_region";

  // Print the workload argument.
  p << "[";
  p.printOperand(op.getWorkload());
  p << " : ";
  p.printType(op.getWorkload()->getType());
  p << "]";

  // Print the data argument remapping.
  p << "(";
  interleaveComma(
      llvm::zip(op.getBody().front().getArguments(), op.getArgOperands()), p,
      [&](std::tuple<BlockArgument *, Value *> it) {
        p << *std::get<0>(it) << " = " << *std::get<1>(it);
        p << " : ";
        p << std::get<1>(it)->getType();
      });
  p << ")";

  // Print the result types, if any.
  if (op.getNumResults() > 0) {
    p << " : ";
    interleaveComma(op.getResultTypes(), p);
  }

  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{});
}

//===----------------------------------------------------------------------===//
// iree.reduction_region
//===----------------------------------------------------------------------===//

void ReductionRegionOp::build(Builder *builder, OperationState &state,
                              ArrayRef<Type> resultTypes, Value *workload,
                              ArrayRef<Value *> operands,
                              ArrayRef<Value *> initialValues,
                              ArrayRef<int64_t> dimensions,
                              ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands({workload});
  state.addOperands(operands);
  state.addOperands(initialValues);
  state.addAttribute(
      "dimensions",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(dimensions.size())},
                                builder->getIntegerType(64)),
          dimensions));
  state.addAttributes(attributes);
  state.addRegion();
  state.setOperandListToResizable();
}

void ReductionRegionOp::build(
    Builder *builder, OperationState &state, ArrayRef<Type> resultTypes,
    Value *workload, ArrayRef<Value *> operands,
    ArrayRef<Value *> initialValues, ArrayRef<int64_t> windowDimensions,
    ArrayRef<int64_t> windowStrides, ArrayRef<int64_t> baseDilations,
    ArrayRef<int64_t> windowDilations, PaddingMode paddingMode,
    ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands({workload});
  state.addOperands(operands);
  state.addOperands(initialValues);
  state.addAttribute(
      "window_dimensions",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(windowDimensions.size())},
                                builder->getIntegerType(64)),
          windowDimensions));
  state.addAttribute(
      "window_strides",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(windowStrides.size())},
                                builder->getIntegerType(64)),
          windowStrides));
  state.addAttribute(
      "base_dilations",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(baseDilations.size())},
                                builder->getIntegerType(64)),
          baseDilations));
  state.addAttribute(
      "window_dilations",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(windowDilations.size())},
                                builder->getIntegerType(64)),
          windowDilations));
  state.addAttribute("padding_mode", builder->getI32IntegerAttr(
                                         static_cast<int32_t>(paddingMode)));
  state.addAttributes(attributes);
  state.addRegion();
  state.setOperandListToResizable();
}

ParseResult parseReductionRegionOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType workloadArg;
  Type workloadArgType;
  if (failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workloadArg)) ||
      failed(parser.parseColonType(workloadArgType)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperand(workloadArg, workloadArgType,
                                   state.operands))) {
    return failure();
  }

  SmallVector<OpAsmParser::OperandType, 8> reductionOperands;
  Type reductionType;
  auto operandsLoc = parser.getCurrentLocation();
  if (failed(parser.parseLParen()) ||
      failed(parser.parseOperandList(reductionOperands)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseColonType(reductionType)) ||
      failed(parser.resolveOperands(
          reductionOperands, reductionType.cast<FunctionType>().getInputs(),
          operandsLoc, state.operands))) {
    return failure();
  }
  for (auto type : reductionType.cast<FunctionType>().getResults()) {
    state.types.push_back(type);
  }
  state.setOperandListToResizable();

  SmallVector<OpAsmParser::OperandType, 8> regionArgs;
  SmallVector<Type, 8> regionArgTypes;
  if (failed(parser.parseKeyword("invocation")) ||
      failed(parser.parseLParen())) {
    return failure();
  }
  do {
    Type argType;
    SmallVector<OpAsmParser::OperandType, 2> reductionRegionArgs;
    OpAsmParser::OperandType initialValue;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseOperandList(reductionRegionArgs, 2)) ||
        failed(parser.parseRParen()) || failed(parser.parseEqual()) ||
        failed(parser.parseOperand(initialValue)) ||
        failed(parser.parseColonType(argType)) ||
        failed(parser.resolveOperand(initialValue, argType, state.operands))) {
      return failure();
    }
    regionArgs.push_back(reductionRegionArgs[0]);
    regionArgTypes.push_back(argType);
    regionArgs.push_back(reductionRegionArgs[1]);
    regionArgTypes.push_back(argType);
  } while (succeeded(parser.parseOptionalComma()));
  if (failed(parser.parseRParen())) {
    return failure();
  }

  // Parse region body.
  Region *body = state.addRegion();
  if (failed(parser.parseRegion(*body, regionArgs, regionArgTypes)) ||
      failed(parser.parseOptionalAttrDict(state.attributes))) {
    return failure();
  }

  return success();
}

void printReductionRegionOp(OpAsmPrinter &p, ReductionRegionOp op) {
  p << "iree.reduction_region";

  // Print the workload argument.
  p << "[";
  p.printOperand(op.getWorkload());
  p << " : ";
  p.printType(op.getWorkload()->getType());
  p << "]";

  p << "(";
  p.printOperands(op.getODSOperands(1));
  p << ")";
  if (op.getNumResults() > 0) {
    p << " : (";
    interleaveComma(op.getODSOperands(1), p,
                    [&](Value *operand) { p.printType(operand->getType()); });
    p << ")";
    p << " -> (";
    interleaveComma(op.getResultTypes(), p);
    p << ")";
  }
  p << "\n";

  p << "      invocation(";
  auto &entryBlock = op.getBody().getBlocks().front();
  int regionArgIndex = 0;
  interleaveComma(op.getODSOperands(2), p, [&](Value *operand) {
    p << "(";
    p.printOperand(entryBlock.getArgument(regionArgIndex++));
    p << ", ";
    p.printOperand(entryBlock.getArgument(regionArgIndex++));
    p << ") = ";
    p.printOperand(operand);
    p << " : ";
    p.printType(operand->getType());
  });
  p << ") ";

  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{});
}

//===----------------------------------------------------------------------===//
// iree.return
//===----------------------------------------------------------------------===//

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, state.operands));
}

static void printReturnOp(OpAsmPrinter &p, ReturnOp op) {
  p << "iree.return";
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// iree.load_input
//===----------------------------------------------------------------------===//

ParseResult parseLoadInputOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operand;
  Type argType;
  if (parser.parseLParen() || parser.parseOperand(operand) ||
      parser.parseColonType(argType) || parser.parseRParen() ||
      parser.resolveOperand(operand, argType, state.operands)) {
    return failure();
  }
  Type outputType;
  if (parser.parseColonType(outputType) ||
      parser.addTypeToList(outputType, state.types)) {
    return failure();
  }
  return success();
}

void printLoadInputOp(OpAsmPrinter &printer, Operation *op) {
  auto *inputValue = op->getOperand(0);
  auto *outputValue = op->getResult(0);
  printer << op->getName() << '(';
  printer.printOperand(inputValue);
  printer << " : ";
  printer.printType(inputValue->getType());
  printer << ") : ";
  printer.printType(outputValue->getType());
}

//===----------------------------------------------------------------------===//
// iree.store_output
//===----------------------------------------------------------------------===//

ParseResult parseStoreOutputOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType op0, op1;
  Type argType0, argType1;
  if (parser.parseLParen() || parser.parseOperand(op0) ||
      parser.parseColonType(argType0) || parser.parseComma() ||
      parser.resolveOperand(op0, argType0, state.operands) ||
      parser.parseOperand(op1) || parser.parseColonType(argType1) ||
      parser.parseRParen() ||
      parser.resolveOperand(op1, argType1, state.operands)) {
    return failure();
  }
  return success();
}

void printStoreOutputOp(OpAsmPrinter &printer, Operation *op) {
  auto *inputValue = op->getOperand(0);
  auto *outputValue = op->getOperand(1);
  printer << op->getName() << '(';
  printer.printOperand(inputValue);
  printer << " : ";
  printer.printType(inputValue->getType());
  printer << ", ";
  printer.printOperand(outputValue);
  printer << " : ";
  printer.printType(outputValue->getType());
  printer << ")";
}

#define GET_OP_CLASSES
#include "iree/compiler/IR/Ops.cpp.inc"

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
