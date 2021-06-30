// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
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
namespace linalg_ext {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// Common methods from Linalg dialect.
//===----------------------------------------------------------------------===//

static ParseResult parseCommonStructuredOpParts(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<Type> &inputTypes, SmallVectorImpl<Type> &outputTypes) {
  llvm::SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> inputsOperands, outputsOperands;

  parser.parseOptionalAttrDict(result.attributes);

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen()) return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(inputsOperands.size()),
                           static_cast<int32_t>(outputsOperands.size())}));
  return success();
}

static ParseResult parseNamedStructuredOpResults(
    OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes)) return failure();
  return success();
}

template <typename NamedStructuredOpType>
static void printCommonStructuredOpParts(OpAsmPrinter &p,
                                         NamedStructuredOpType op) {
  if (!op.inputs().empty()) {
    p << " ins(" << op.inputs() << " : " << op.inputs().getTypes() << ")";
  }
  if (!op.outputs().empty()) {
    p << " outs(" << op.outputs() << " : " << op.outputs().getTypes() << ")";
  }
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty()) return;
  p.printOptionalArrowTypeList(resultTypes);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

static ParseResult parseSortOp(OpAsmParser &parser, OperationState &result) {
  DictionaryAttr dictAttr;
  parser.parseOptionalAttribute(dictAttr, "_", result.attributes);
  if (dictAttr) {
    result.attributes.assign(dictAttr.getValue().begin(),
                             dictAttr.getValue().end());
  }

  // Parsing is shared with named ops, except for the region.
  SmallVector<Type> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  SmallVector<OpAsmParser::OperandType> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  SmallVector<Type> operandTypes, regionTypes;
  if (parser.parseRegion(*region, regionOperands, regionTypes))
    return failure();
  result.addRegion(std::move(region));

  SmallVector<Type> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);
  return success();
}

static void printSortOp(OpAsmPrinter &p, SortOp op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operand_segment_sizes"});
  printCommonStructuredOpParts(p, op);
  if (!op.region().empty()) {
    p.printRegion(op.region());
  }
  printNamedStructuredOpResults(p, op.result_tensors().getTypes());
}

static LogicalResult verifySortOp(SortOp op) {
  if (op.getNumInputs()) {
    return op.emitOpError("does not expect to take any inputs");
  }

  Block &block = op.region().front();
  size_t numOutputs = op.getNumOutputs();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op.emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int rank = op.getRank(op.getOutputOperand(0));
  if (rank > 1 && !op.dimensionAttr()) {
    return op.emitOpError("dimension must be specified if rank > 1");
  }
  int dimension = 0;
  if (op.dimensionAttr()) {
    dimension = op.dimension().getValue();
  }
  if (dimension < 0 || dimension >= rank) {
    return op.emitOpError("dimension must be within (0, ") << rank << "]";
  }

  for (auto indexedOperand : llvm::enumerate(op.inputs())) {
    int index = indexedOperand.index();
    Type elemType =
        indexedOperand.value().getType().cast<ShapedType>().getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op.emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op.emitOpError("should yield exactly one operand");
  }
  auto ty = yieldOp.getOperand(0).getType().dyn_cast<IntegerType>();
  if (!ty || ty.getWidth() != 1) {
    return op.emitOpError("should yield i1 type");
  }

  return success();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
