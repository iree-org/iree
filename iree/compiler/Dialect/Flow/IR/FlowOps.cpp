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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOpUtils.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of constant that can be inlined into a "
                   "dispatch region"),
    llvm::cl::init(256));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Op utilities used within the Flow dialect
//===----------------------------------------------------------------------===//

// Returns true if the given |accessType| is compatible with the |variableType|.
// For example, this will return true if the variable type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isVariableTypeCompatible(Type variableType, Type accessType) {
  return succeeded(mlir::verifyCompatibleShape(variableType, accessType));
}

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// of the dynamic dimensions in |values|.
static LogicalResult verifyOpDynamicDims(Operation *op, ValueRange values,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
      requiredCount += shapedType.getNumDynamicDims();
    }
  }
  if (dynamicDims.size() != requiredCount) {
    return op->emitOpError()
           << "value set has " << requiredCount
           << " dynamic dimensions but only " << dynamicDims.size()
           << " dimension values are attached";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// custom<TiedResult>
//===----------------------------------------------------------------------===//
// type{%dim0, %dim1}
// %arg0

static ParseResult parseTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands) {
  if (failed(parser.parseType(resultType))) return failure();
  if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    if (!shapedType.hasStaticShape()) {
      SmallVector<OpAsmParser::OperandType, 4> dynamicDims;
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
  tiedOperands = parser.getBuilder().getIndexArrayAttr({0});
  return success();
}

static void printTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                            ValueRange resultDims, ArrayAttr tiedOperands) {
  p.printType(resultType);
  if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
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
    }
  }
}

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionType>
//===----------------------------------------------------------------------===//
// (type, type{%dim0, %dim1}, type) -> (type{%dim2}, %operand4)

static ParseResult parseShapedOperandList(
    OpAsmParser &parser, SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::OperandType> &dims) {
  do {
    Type type;
    if (failed(parser.parseType(type))) return failure();
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      if (!shapedType.hasStaticShape()) {
        SmallVector<OpAsmParser::OperandType, 4> dynamicDims;
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

// Ties the |tiedResult| parsed operand back to a previously parsed operand.
// The type and any dynamic dimensions of the operand will be used for the
// result values and the operand index will be appended to |tiedOperandIndices|.
static ParseResult tieOperand(
    OpAsmParser::OperandType tiedResult, OpAsmParser &parser,
    ArrayRef<OpAsmParser::OperandType> operands, TypeRange operandTypes,
    ArrayRef<OpAsmParser::OperandType> operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    SmallVectorImpl<int64_t> &tiedOperandIndices) {
  int64_t operandIndex = TiedOpInterface::kUntiedIndex;
  for (int64_t i = 0; i < operands.size(); ++i) {
    if (operands[i].name == tiedResult.name) {
      operandIndex = i;
      break;
    }
  }
  if (operandIndex == TiedOpInterface::kUntiedIndex) {
    return parser.emitError(tiedResult.location,
                            "tied operand not found for result reference ")
           << tiedResult.name;
  }

  auto resultType = operandTypes[operandIndex];
  resultTypes.push_back(resultType);
  tiedOperandIndices.push_back(operandIndex);

  auto shapedType = resultType.dyn_cast<ShapedType>();
  if (shapedType) {
    unsigned dimsIndex = 0;
    for (unsigned i = 0; i < operandIndex; ++i) {
      if (auto shapedType = operandTypes[i].dyn_cast<ShapedType>()) {
        dimsIndex += shapedType.getNumDynamicDims();
      }
    }
    resultDims.append(llvm::to_vector<4>(
        operandDims.slice(dimsIndex, shapedType.getNumDynamicDims())));
  }

  return success();
}

static ParseResult parseShapedResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::OperandType> operands,
    TypeRange operandTypes, ArrayRef<OpAsmParser::OperandType> operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands) {
  SmallVector<int64_t, 4> tiedOperandIndices;
  do {
    OpAsmParser::OperandType tiedResult;
    auto res = parser.parseOptionalOperand(tiedResult);
    if (res.hasValue() && succeeded(res.getValue())) {
      if (failed(tieOperand(tiedResult, parser, operands, operandTypes,
                            operandDims, resultTypes, resultDims,
                            tiedOperandIndices))) {
        return failure();
      }
    } else {
      Type type;
      if (failed(parser.parseType(type))) return failure();
      if (auto shapedType = type.dyn_cast<ShapedType>()) {
        if (!shapedType.hasStaticShape()) {
          SmallVector<OpAsmParser::OperandType, 4> dynamicDims;
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
      tiedOperandIndices.push_back(TiedOpInterface::kUntiedIndex);
    }
  } while (succeeded(parser.parseOptionalComma()));
  if (!tiedOperandIndices.empty()) {
    tiedOperands = parser.getBuilder().getIndexArrayAttr(tiedOperandIndices);
  }
  return success();
}

static ParseResult parseShapedFunctionType(
    OpAsmParser &parser, ArrayRef<OpAsmParser::OperandType> operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands) {
  if (failed(parser.parseLParen())) return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parseShapedOperandList(parser, operandTypes, operandDims)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }
  if (failed(parser.parseArrow())) return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parseShapedResultList(parser, operands, operandTypes,
                                     operandDims, resultTypes, resultDims,
                                     tiedOperands)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    if (failed(parseShapedResultList(parser, operands, operandTypes,
                                     operandDims, resultTypes, resultDims,
                                     tiedOperands))) {
      return failure();
    }
  }
  return success();
}

static void printShapedFunctionType(OpAsmPrinter &p, Operation *op,
                                    ValueRange operands, TypeRange operandTypes,
                                    OperandRange operandDims,
                                    TypeRange resultTypes,
                                    OperandRange resultDims,
                                    ArrayAttr tiedOperands) {
  p << "(";
  llvm::interleaveComma(operandTypes, p, [&](Type type) {
    p.printType(type);
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
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
  if (resultTypes.size() != 1) p << "(";
  auto tiedOp = cast<TiedOpInterface>(op);
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    auto tiedOperand = tiedOp.getTiedResultOperandIndex(i);
    if (tiedOperand.hasValue()) {
      p.printOperand(op->getOperand(tiedOperand.getValue()));
    } else {
      auto type = resultTypes[i];
      p.printType(type);
      if (auto shapedType = type.dyn_cast<ShapedType>()) {
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
    if (i < resultTypes.size() - 1) p << ", ";
  }
  if (resultTypes.size() != 1) p << ")";
}

//===----------------------------------------------------------------------===//
// flow.variable
//===----------------------------------------------------------------------===//

static ParseResult parseVariableOp(OpAsmParser &parser,
                                   OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("mutable"))) {
    result->addAttribute("is_mutable", UnitAttr::get(result->getContext()));
  }

  if (succeeded(parser.parseOptionalKeyword("init"))) {
    FlatSymbolRefAttr initializerAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(initializerAttr, "initializer",
                                     result->attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  }

  if (failed(parser.parseOptionalColon())) {
    Attribute initialValueAttr;
    if (failed(parser.parseAttribute(initialValueAttr, "initial_value",
                                     result->attributes))) {
      return failure();
    }
    result->addAttribute("type", TypeAttr::get(initialValueAttr.getType()));
  } else {
    Type type;
    if (failed(parser.parseType(type))) {
      return failure();
    }
    result->addAttribute("type", TypeAttr::get(type));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  return success();
}

static void printVariableOp(OpAsmPrinter &p, VariableOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  if (op.is_mutable()) {
    p << " mutable";
  }
  if (op.initializer().hasValue()) {
    p << " init(";
    p.printSymbolName(op.initializer().getValue());
    p << ')';
  }
  if (op.initial_value().hasValue()) {
    p << ' ';
    p.printAttribute(op.initial_value().getValue());
  } else {
    p << " : ";
    p.printType(op.type());
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), /*elidedAttrs=*/{
                                         "sym_name",
                                         "type",
                                         "is_mutable",
                                         "initializer",
                                         "initial_value",
                                     });
}

static LogicalResult verifyVariableOp(VariableOp op) {
  if (op.initializer().hasValue() && op.initial_value().hasValue()) {
    return op.emitOpError()
           << "variables can have either an initializer or an initial value";
  } else if (op.initializer().hasValue()) {
    // Ensure initializer returns the same type as the variable.
    auto *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue());
    if (!symbolOp) {
      return op.emitOpError() << "initializer function "
                              << op.initializer().getValue() << " not found";
    }
    auto initializerOp = dyn_cast<FuncOp>(symbolOp);
    if (initializerOp.getNumArguments() != 0 ||
        initializerOp.getNumResults() != 1 ||
        initializerOp.getType().getResult(0) != op.type()) {
      return op.emitOpError()
             << "initializer type mismatch; variable " << op.sym_name()
             << " is " << op.type() << " but initializer function "
             << initializerOp.getName() << " is " << initializerOp.getType();
    }
  } else if (op.initial_value().hasValue()) {
    // Ensure the value is something we can store in the variable
    if (!isVariableTypeCompatible(op.type(), op.initial_value()->getType())) {
      return op.emitOpError()
             << "initial value type mismatch; variable " << op.sym_name()
             << " is " << op.type() << " but initial value provided is "
             << op.initial_value()->getType();
    }
  }
  return success();
}

void VariableOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name, bool isMutable, FuncOp initializer,
                       ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  if (isMutable) {
    state.addAttribute("is_mutable", builder.getUnitAttr());
  }
  state.addAttribute("initializer", builder.getSymbolRefAttr(initializer));
  state.addAttribute("type", TypeAttr::get(initializer.getType().getResult(0)));
  state.attributes.append(attrs.begin(), attrs.end());
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, Type type,
                       Attribute initialValue, ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  result.addAttribute("initial_value", initialValue);
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void VariableOp::build(OpBuilder &builder, OperationState &result,
                       StringRef name, bool isMutable, Type type,
                       ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

//===----------------------------------------------------------------------===//
// flow.variable.load
//===----------------------------------------------------------------------===//

void VariableLoadOp::build(OpBuilder &builder, OperationState &state,
                           VariableOp variableOp,
                           ArrayRef<NamedAttribute> attrs) {
  state.addTypes({variableOp.type()});
  state.addAttribute("variable", builder.getSymbolRefAttr(variableOp));
  state.attributes.append(attrs.begin(), attrs.end());
}

void VariableLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // HACK: works around the lack of symbol side effects in mlir by only saying
  // we have a side-effect if the variable we are loading is mutable.
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(*this, variable());
  assert(symbolOp);
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  if (variableOp.is_mutable()) {
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

static LogicalResult verifyVariableLoadOp(VariableLoadOp &op) {
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, op.variable());
  if (!symbolOp) {
    return op.emitOpError() << "undefined variable: " << op.variable();
  }
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  auto loadType = op.result().getType();
  if (!isVariableTypeCompatible(variableOp.type(), loadType)) {
    return op.emitOpError()
           << "variable type mismatch; variable " << op.variable() << " is "
           << variableOp.type() << " but load is " << loadType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.variable.load.indirect
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableLoadIndirectOp(VariableLoadIndirectOp &op) {
  auto variableType =
      op.variable().getType().cast<IREE::PtrType>().getTargetType();
  auto loadType = op.result().getType();
  if (!isVariableTypeCompatible(variableType, loadType)) {
    return op.emitOpError() << "variable type mismatch; variable pointer is "
                            << variableType << " but load is " << loadType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.variable.store
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableStoreOp(VariableStoreOp &op) {
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, op.variable());
  if (!symbolOp) {
    return op.emitOpError() << "undefined variable: " << op.variable();
  }
  auto variableOp = dyn_cast<VariableOp>(symbolOp);
  auto storeType = op.value().getType();
  if (!isVariableTypeCompatible(variableOp.type(), storeType)) {
    return op.emitOpError()
           << "variable type mismatch; variable " << op.variable() << " is "
           << variableOp.type() << " but store is " << storeType;
  }
  if (!variableOp.is_mutable()) {
    return op.emitOpError() << "variable " << op.variable()
                            << " is not mutable and cannot be stored to";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.variable.store.indirect
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariableStoreIndirectOp(
    VariableStoreIndirectOp &op) {
  auto variableType =
      op.variable().getType().cast<IREE::PtrType>().getTargetType();
  auto storeType = op.value().getType();
  if (!isVariableTypeCompatible(variableType, storeType)) {
    return op.emitOpError() << "variable type mismatch; variable pointer is "
                            << variableType << " but store is " << storeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.region
//===----------------------------------------------------------------------===//

/// Inlines operation |op| into the |dispatchRegionOp| by making all operands,
/// as well as values caputred implicitly by the regions of the operation, that
/// are outside the dispatch region operands of the dispatch region as well.
static Operation *inlineOpIntoDispatchRegion(OpBuilder &builder,
                                             DispatchRegionOp dispatchRegionOp,
                                             Operation *op,
                                             BlockAndValueMapping &map) {
  llvm::SetVector<Value> capturedInputs(op->getOperands().begin(),
                                        op->getOperands().end());
  getUsedValuesDefinedAbove(op->getRegions(), capturedInputs);
  Block *block = builder.getInsertionBlock();
  for (Value capturedInput : capturedInputs) {
    if (map.contains(capturedInput)) continue;
    dispatchRegionOp.getOperation()->insertOperands(
        dispatchRegionOp.getOperation()->getNumOperands(), {capturedInput});
    Value newBlockArgument = block->addArgument(capturedInput.getType());
    map.map(capturedInput, newBlockArgument);
  }

  return builder.clone(*op, map);
}

llvm::Optional<std::pair<DispatchRegionOp, Operation *>>
DispatchRegionOp::formFromAnchorOp(Value workload, Operation *anchorOp,
                                   OpBuilder &builder) {
  builder.setInsertionPoint(anchorOp);
  auto loc = anchorOp->getLoc();
  // Map anchor into new dispatch region.
  auto drOp = builder.create<DispatchRegionOp>(
      loc, llvm::to_vector<1>(anchorOp->getResultTypes()), workload,
      ArrayRef<Value>());
  auto *drBlock = new Block();
  drOp.body().push_back(drBlock);
  BlockAndValueMapping mapping;
  builder.setInsertionPointToEnd(drBlock);
  Operation *newAnchorOp =
      inlineOpIntoDispatchRegion(builder, drOp, anchorOp, mapping);

  // Insert terminator
  builder.create<IREE::Flow::ReturnOp>(loc, newAnchorOp->getResults());

  // Replace anchor uses with region result.
  for (auto it : llvm::enumerate(anchorOp->getResults())) {
    it.value().replaceAllUsesWith(drOp.getResult(it.index()));
  }
  anchorOp->erase();
  return std::make_pair(drOp, newAnchorOp);
}

// Clones an operation with new result types.
// The original operation will be erased and a new operation constructed
// in its place.
static Operation *cloneWithNewResultTypes(Operation *op,
                                          TypeRange newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(newResultTypes);
  state.addSuccessors(op->getSuccessors());
  state.addAttributes(op->getAttrs());
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    state.addRegion();
  }
  Operation *newOp = Operation::create(state);
  for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
    newOp->getRegion(i).takeBody(op->getRegion(i));
  }
  return newOp;
}

ResultRange DispatchRegionOp::appendResults(DispatchRegionOp &self,
                                            ValueRange addlResults,
                                            OpBuilder &builder) {
  Block &block = self.body().front();

  unsigned origNumResults = self.getNumResults();
  llvm::SmallVector<Type, 4> newTypes(self.getResultTypes().begin(),
                                      self.getResultTypes().end());
  for (auto r : addlResults) newTypes.push_back(r.getType());

  // Changing the arity of the results requires replacing the dispatch region.
  builder.setInsertionPoint(self);
  auto newDrOp = llvm::cast<DispatchRegionOp>(
      builder.insert(cloneWithNewResultTypes(self, newTypes)));
  self.replaceAllUsesWith(newDrOp->getResults().take_front(origNumResults));
  self.erase();
  self = newDrOp;

  // Add results to the terminator.
  auto terminator = block.getTerminator();
  llvm::SmallVector<Value, 4> returns(terminator->getOperands());
  returns.append(addlResults.begin(), addlResults.end());
  terminator->setOperands(returns);

  return self->getResults().slice(origNumResults, addlResults.size());
}

Operation *DispatchRegionOp::inlineOp(Operation *origOp, OpBuilder &builder,
                                      bool positionAtEnd) {
  Block &block = body().front();
  if (positionAtEnd) {
    builder.setInsertionPoint(block.getTerminator());
  } else {
    builder.setInsertionPointToStart(&block);
  }
  // Map existing dr args.
  BlockAndValueMapping mapping;
  for (unsigned i = 0, e = block.getNumArguments(); i < e; ++i) {
    mapping.map(args()[i], block.getArgument(i));
  }

  // Also map any terminator operands to support inlining at the end.
  for (auto it : llvm::enumerate(block.getTerminator()->getOperands())) {
    mapping.map(getResult(it.index()), it.value());
  }

  // Remember the values corresponding to original op results.
  llvm::SmallVector<Value, 4> origOpResultValues;
  for (Value result : origOp->getResults()) {
    origOpResultValues.push_back(mapping.lookupOrNull(result));
  }

  Operation *inlinedOp =
      inlineOpIntoDispatchRegion(builder, *this, origOp, mapping);

  // Replace any results from the orig with results from the clone.
  for (unsigned i = 0, e = origOp->getNumResults(); i < e; ++i) {
    Value resultTo = origOpResultValues[i];
    if (resultTo) {
      resultTo.replaceAllUsesWith(inlinedOp->getResult(i));
    }
  }

  return inlinedOp;
}

void DispatchRegionOp::build(OpBuilder &builder, OperationState &state,
                             ArrayRef<Type> resultTypes, Value workload,
                             ValueRange args,
                             ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands({workload});
  state.addOperands(args);
  state.addAttributes(attributes);
  state.addRegion();
}

ParseResult parseDispatchRegionOp(OpAsmParser &parser, OperationState *result) {
  // Parse required workload.
  OpAsmParser::OperandType workloadArg;
  Type workloadArgType;
  if (failed(parser.parseLSquare()) ||
      failed(parser.parseOperand(workloadArg)) ||
      failed(parser.parseColonType(workloadArgType)) ||
      failed(parser.parseRSquare()) ||
      failed(parser.resolveOperand(workloadArg, workloadArgType,
                                   result->operands))) {
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
                                      result->operands))) {
      return failure();
    }
  }

  // Parse (optional) results.
  if (failed(parser.parseOptionalArrowTypeList(result->types))) {
    return failure();
  }

  // Parse region body.
  Region *body = result->addRegion();
  if (failed(parser.parseRegion(*body, regionArgs, regionArgTypes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes))) {
    return failure();
  }
  return success();
}

void printDispatchRegionOp(OpAsmPrinter &p, DispatchRegionOp op) {
  p << op.getOperationName();

  // Print the workload argument.
  p << "[";
  p.printOperand(op.workload());
  p << " : ";
  p.printType(op.workload().getType());
  p << "]";

  // Print the data argument remapping.
  p << "(";
  interleaveComma(llvm::zip(op.body().getArguments(), op.args()), p,
                  [&](std::tuple<BlockArgument, Value> it) {
                    p << std::get<0>(it) << " = " << std::get<1>(it);
                    p << " : ";
                    p << std::get<1>(it).getType();
                  });
  p << ")";

  // Print the result types, if any.
  if (op.getNumResults() > 0) {
    p << " -> (";
    interleaveComma(op.getResultTypes(), p);
    p << ")";
  }

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{});
}

Operation::operand_range DispatchRegionOp::getClosureOperands() {
  return args();
}

Operation::result_range DispatchRegionOp::getClosureResults() {
  return results();
}

// TODO(#4897): allow non-splat constants - current paths can't handle them.
static bool canDispatchRegionContainOpIssue4897(Operation *op) {
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    auto constantValueAttr = constantOp.getValue();
    auto constantType = constantOp.getType();
    if (constantValueAttr.isa<SplatElementsAttr>()) {
      return true;
    } else if (auto denseAttr =
                   constantValueAttr.dyn_cast<DenseElementsAttr>()) {
      return denseAttr.isSplat();
    } else if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  return false;
}

// Inline operations that the dispatch region can handle natively.
static bool canDispatchRegionContainOp(Operation *op) {
  // Inline constant operations that are splat or small constants.
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    auto constantValueAttr = constantOp.getValue();
    auto constantType = constantOp.getType();
    if (constantValueAttr.isa<SplatElementsAttr>()) {
      return true;
    } else if (auto denseAttr =
                   constantValueAttr.dyn_cast<DenseElementsAttr>()) {
      // TODO(GH-4897): Non-splat constants seems to have an issue on the LLLVM
      // side. Uncomment after that is fixed.
      auto shapedType = constantOp.getType().cast<ShapedType>();
      uint64_t estimatedByteLength =
          (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) /
          8;
      return denseAttr.isSplat() ||
             estimatedByteLength <= clInlineConstantByteLength;
    } else if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  return false;
}

bool DispatchRegionOp::canClosureContainOp(Operation *op) {
  return canDispatchRegionContainOpIssue4897(op);
}

ClosureOpInterface
DispatchRegionOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices) {
  SmallVector<Type, 4> newResultTypes = llvm::to_vector<4>(getResultTypes());
  SmallVector<Value, 4> newOperandsValues = llvm::to_vector<4>(args());
  excludeClosureOperandsAndResults(newOperandsValues, excludedOperandIndices,
                                   newResultTypes, excludedResultIndices);
  auto newOp = OpBuilder(getContext())
                   .create<DispatchRegionOp>(getLoc(), newResultTypes,
                                             workload(), newOperandsValues,
                                             getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  eraseRegionResults(newBody, excludedResultIndices);
  newBody.front().eraseArguments(excludedOperandIndices);
  return newOp;
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

void DispatchTensorLoadOp::build(OpBuilder &builder, OperationState &state,
                                 RankedTensorType returnType, Value source,
                                 ArrayRef<NamedAttribute> attributes) {
  build(builder, state, returnType, source, ArrayRef<Value>(),
        ArrayRef<Value>(), ArrayRef<Value>(), builder.getI64ArrayAttr({}),
        builder.getI64ArrayAttr({}), builder.getI64ArrayAttr({}));
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

void DispatchWorkgroupsOp::build(OpBuilder &builder, OperationState &state,
                                 ValueRange workgroupCount,
                                 TypeRange resultTypes, ValueRange resultDims,
                                 ValueRange operands, ValueRange operandDims,
                                 ArrayRef<int64_t> tiedOperands,
                                 ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(workgroupCount);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(TiedOpInterface::getStorageAttrName());
  state.addAttribute(TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(workgroupCount.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));

  auto *body = state.addRegion();
  assert(body->begin() == body->end());
  {
    OpBuilder::InsertionGuard g(builder);
    builder.createBlock(body);  // createBlock implicitly moves IP, RAII away...
  }

  llvm::BitVector operandAliases(llvm::size(operands), false);
  llvm::BitVector resultAliases(llvm::size(resultTypes), false);
  for (unsigned resultIndex = 0; resultIndex < tiedOperands.size();
       ++resultIndex) {
    int64_t tiedOperandIndex = tiedOperands[resultIndex];
    if (tiedOperandIndex != TiedOpInterface::kUntiedIndex) {
      operandAliases[tiedOperandIndex] = true;
      resultAliases[resultIndex] = true;
    }
  }

  for (auto operand : llvm::enumerate(operands)) {
    Type type = operand.value().getType();
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchTensorType::get(operandAliases[operand.index()]
                                         ? TensorAccess::ReadWrite
                                         : TensorAccess::ReadOnly,
                                     tensorType);
    }
    body->addArgument(type);
  }
  for (auto resultType : llvm::enumerate(resultTypes)) {
    if (resultAliases[resultType.index()]) {
      // Already handled by an aliased operand.
      continue;
    }
    Type type = resultType.value();
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchTensorType::get(TensorAccess::WriteOnly, tensorType);
    }
    body->addArgument(type);
  }
  assert(std::next(body->begin()) == body->end());
}

static ParseResult parseDispatchWorkgroupBody(OpAsmParser &parser,
                                              TypeRange operandTypes,
                                              TypeRange resultTypes,
                                              Region &body) {
  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
  SmallVector<Type, 16> regionArgTypes;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Reserve entries in the lists.
      regionArgs.emplace_back();
      regionArgTypes.emplace_back();
      if (failed(parser.parseRegionArgument(regionArgs.back())) ||
          failed(parser.parseColonType(regionArgTypes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }
  return parser.parseRegion(body, regionArgs, regionArgTypes,
                            /*enableNameShadowing=*/true);
}

static void printDispatchWorkgroupBody(OpAsmPrinter &p, Operation *op,
                                       TypeRange operandTypes,
                                       TypeRange resultTypes, Region &body) {
  p << "(";
  interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p << arg;
    p << ": ";
    p << arg.getType();
  });
  p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

static LogicalResult verifyDispatchWorkgroupsOp(DispatchWorkgroupsOp op) {
  if (op.workgroup_count().empty()) {
    return op.emitOpError() << "at least one workgroup dimension is required";
  }
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

Value DispatchWorkgroupsOp::buildOperandRankedShape(unsigned idx,
                                                    OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value DispatchWorkgroupsOp::buildResultRankedShape(unsigned idx,
                                                   OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
}

Operation::operand_range DispatchWorkgroupsOp::getClosureOperands() {
  return operands();
}

Operation::result_range DispatchWorkgroupsOp::getClosureResults() {
  return results();
}

bool DispatchWorkgroupsOp::canClosureContainOp(Operation *op) {
  return canDispatchRegionContainOp(op);
}

ClosureOpInterface
DispatchWorkgroupsOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices) {
  SmallVector<Type, 4> newResultTypes = llvm::to_vector<4>(getResultTypes());
  SmallVector<Value, 4> newResultDims = llvm::to_vector<4>(result_dims());
  SmallVector<Value, 4> newOperandsValues = llvm::to_vector<4>(operands());
  SmallVector<Value, 4> newOperandDims = llvm::to_vector<4>(operand_dims());
  excludeClosureOperandsAndResults(newOperandsValues, newOperandDims,
                                   excludedOperandIndices, newResultTypes,
                                   newResultDims, excludedResultIndices);

  auto newTiedOperandIndices =
      llvm::to_vector<4>(getTiedResultOperandIndices());

  // TODO(benvanik): all this offset stuff is confusing and should be reworked.
  // We should probably have absolute indices and relative indices, or just one
  // or the other, and not be crossing the streams. The way things are offset
  // is the same as variadic ODS operands for consistency, but just like ODS
  // operands half of the code assumes its within a particular ODS operand and
  // half the code assumes it's within the flattened set of all Operation
  // operands.
  unsigned tiedOperandOffset = getTiedOperandsIndexAndLength().first;
  for (unsigned i = 0; i < newTiedOperandIndices.size(); ++i) {
    if (newTiedOperandIndices[i] != TiedOpInterface::kUntiedIndex) {
      newTiedOperandIndices[i] -= tiedOperandOffset;
    }
  }

  // This need to happen *after* accounting for tied operand offset, given that
  // all excluded operand/result indices are relative ranges.
  excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);

  auto newOp = OpBuilder(getContext())
                   .create<DispatchWorkgroupsOp>(
                       getLoc(), workgroup_count(), newResultTypes,
                       newResultDims, newOperandsValues, newOperandDims,
                       newTiedOperandIndices, getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  newBody.front().eraseArguments(excludedOperandIndices);
  unsigned baseResultIndex = newBody.front().getNumArguments();
  newBody.front().eraseArguments(llvm::to_vector<4>(llvm::map_range(
      excludedResultIndices,
      [&](unsigned index) { return baseResultIndex + index; })));
  return newOp;
}

std::pair<unsigned, unsigned>
DispatchWorkgroupsOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroup.*
//===----------------------------------------------------------------------===//

void DispatchWorkgroupRankOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), "workgroup_rank");
}

static void getAsmResultNamesForDispatchWorkgroupInfoOp(
    StringRef prefix, const APInt &dimension, Value result,
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result, (prefix + std::to_string(dimension.getZExtValue())).str());
}

void DispatchWorkgroupIDOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_id_", dimension(),
                                              result(), setNameFn);
}

void DispatchWorkgroupCountOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_count_", dimension(),
                                              result(), setNameFn);
}

void DispatchWorkgroupSizeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  getAsmResultNamesForDispatchWorkgroupInfoOp("workgroup_size_", dimension(),
                                              result(), setNameFn);
}

template <typename T>
static LogicalResult verifyDispatchWorkgroupInfoOp(T op) {
  size_t dimCount = 0;
  if (auto dispatchOp = op->template getParentOfType<DispatchWorkgroupsOp>()) {
    dimCount = dispatchOp.workgroup_count().size();
  }
  uint64_t dimension = op.dimension().getZExtValue();
  if (dimCount != 0 && (dimension < 0 || dimension >= dimCount)) {
    return op.emitOpError()
           << "dimension " << dimension
           << " out of bounds of dispatch dimensions; expected [0, "
           << (dimCount - 1) << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.shape
//===----------------------------------------------------------------------===//

void DispatchShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // TODO(benvanik): since we know these are arguments, we could map them based
  // on index (so we get arg0_shape, ret0_shape, etc).
  setNameFn(result(), "shape");
}

LogicalResult DispatchShapeOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto dispatchTensorType = operands[0].getType().cast<DispatchTensorType>();
  auto shape = dispatchTensorType.getShape();
  auto rankedShapeType = Shape::RankedShapeType::get(shape, context);
  inferredReturnTypes.assign({rankedShapeType});
  return success();
}

//===----------------------------------------------------------------------===//
// flow.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(OpBuilder &builder, OperationState &state,
                         StringRef name) {
  ensureTerminator(*state.addRegion(), builder, state.location);
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
}

static ParseResult parseExecutableOp(OpAsmParser &parser,
                                     OperationState *result) {
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  // Parse the module body.
  auto *body = result->addRegion();
  if (failed(parser.parseRegion(*body, llvm::None, llvm::None))) {
    return failure();
  }

  // Ensure that this module has a valid terminator.
  ExecutableOp::ensureTerminator(*body, parser.getBuilder(), result->location);
  return success();
}

static void printExecutableOp(OpAsmPrinter &p, ExecutableOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static LogicalResult verifyExecutableOp(ExecutableOp op) {
  // TODO(benvanik): check export name conflicts.
  return success();
}

//===----------------------------------------------------------------------===//
// flow.dispatch.entry
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchEntryOp(OpAsmParser &parser,
                                        OperationState *result) {
  FlatSymbolRefAttr functionRefAttr;
  if (failed(parser.parseAttribute(functionRefAttr, "function_ref",
                                   result->attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    StringAttr exportNameAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(exportNameAttr, "sym_name",
                                     result->attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    result->addAttribute("sym_name", parser.getBuilder().getStringAttr(
                                         functionRefAttr.getValue()));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  return success();
}

static void printDispatchEntryOp(OpAsmPrinter &p, DispatchEntryOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.function_ref());
  if (op.sym_name() != op.function_ref()) {
    p << " as(\"" << op.sym_name() << "\")";
  }
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(), /*elidedAttrs=*/{"function_ref", "sym_name"});
}

//===----------------------------------------------------------------------===//
// flow.dispatch
//===----------------------------------------------------------------------===//

void DispatchOp::build(OpBuilder &builder, OperationState &state,
                       DispatchEntryOp entryPoint, ValueRange workgroupCount,
                       TypeRange resultTypes, ValueRange resultDims,
                       ValueRange operands, ValueRange operandDims,
                       ArrayAttr tiedOperands,
                       ArrayRef<NamedAttribute> attributes) {
  StringRef executableOpSymName =
      entryPoint->getParentOp()
          ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  state.addAttribute(
      "entry_point",
      builder.getSymbolRefAttr(executableOpSymName,
                               {builder.getSymbolRefAttr(entryPoint)}));

  state.addOperands(workgroupCount);
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(TiedOpInterface::getStorageAttrName());
  state.addAttribute(TiedOpInterface::getStorageAttrName(), tiedOperands);
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(workgroupCount.size()),
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
}

StringRef DispatchOp::executable() { return entry_point().getRootReference(); }

FunctionType DispatchOp::getEntryPointType() {
  SmallVector<Type, 8> argTypes(operand_type_range{operands()});
  return FunctionType::get(getContext(), argTypes, getResultTypes());
}

static LogicalResult verifyDispatchOp(DispatchOp op) {
  if (op.workgroup_count().empty()) {
    return op.emitOpError() << "at least one workgroup dimension is required";
  }
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

Value DispatchOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value DispatchOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
}

std::pair<unsigned, unsigned> DispatchOp::getTiedOperandsIndexAndLength() {
  return getODSOperandIndexAndLength(1);
}

//===----------------------------------------------------------------------===//
// flow.tensor.reshape
//===----------------------------------------------------------------------===//

Value TensorReshapeOp::buildOperandRankedShape(unsigned idx,
                                               OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorReshapeOp::buildResultRankedShape(unsigned idx,
                                              OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
}

Value TensorReshapeOp::getTiedResult(unsigned resultIndex) {
  return IREE::TiedOpInterface::findTiedBaseValue(source());
}

::llvm::Optional<unsigned> TensorReshapeOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // source
}

SmallVector<int64_t, 4> TensorReshapeOp::getTiedResultOperandIndices() {
  return {0};  // source
}

//===----------------------------------------------------------------------===//
// flow.tensor.*
//===----------------------------------------------------------------------===//

Value TensorLoadOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorLoadOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return {};
}

Value TensorStoreOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                         builder);
}

Value TensorStoreOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), target_dims(),
                                         builder);
}

Value TensorSplatOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return {};
}

Value TensorSplatOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
}

Value TensorCloneOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), operand(), operand_dims(),
                                         builder);
}

Value TensorCloneOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), operand_dims(),
                                         builder);
}

Value TensorSliceOp::buildOperandRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), source(), source_dims(),
                                         builder);
}

Value TensorSliceOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), result(), result_dims(),
                                         builder);
}

//===----------------------------------------------------------------------===//
// flow.tensor.update
//===----------------------------------------------------------------------===//

void TensorUpdateOp::build(OpBuilder &builder, OperationState &state,
                           Value target, ValueRange startIndices,
                           Value update) {
  auto targetDims =
      Shape::buildOrFindDynamicDimsForValue(state.location, target, builder);
  auto updateDims =
      Shape::buildOrFindDynamicDimsForValue(state.location, update, builder);
  build(builder, state, target.getType(), target, targetDims, startIndices,
        update, updateDims, builder.getIndexArrayAttr({0}));
}

static LogicalResult verifyTensorUpdateOp(TensorUpdateOp op) {
  if (failed(verifyOpDynamicDims(op, {op.update()}, op.update_dims())) ||
      failed(verifyOpDynamicDims(op, {op.target()}, op.target_dims()))) {
    return failure();
  }
  return success();
}

Value TensorUpdateOp::buildOperandRankedShape(unsigned idx,
                                              OpBuilder &builder) {
  switch (idx) {
    case 0:
      return Shape::buildRankedShapeForValue(getLoc(), update(), update_dims(),
                                             builder);
    case 2:
      return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                             builder);
    default:
      llvm_unreachable("unshaped operand");
  }
}

Value TensorUpdateOp::buildResultRankedShape(unsigned idx, OpBuilder &builder) {
  return Shape::buildRankedShapeForValue(getLoc(), target(), target_dims(),
                                         builder);
}

Value TensorUpdateOp::getTiedResult(unsigned resultIndex) {
  return IREE::TiedOpInterface::findTiedBaseValue(target());
}

::llvm::Optional<unsigned> TensorUpdateOp::getTiedResultOperandIndex(
    unsigned resultIndex) {
  return {0};  // target
}

SmallVector<int64_t, 4> TensorUpdateOp::getTiedResultOperandIndices() {
  return {0};  // target
}

//===----------------------------------------------------------------------===//
// flow.ex.stream.fragment
//===----------------------------------------------------------------------===//

void ExStreamFragmentOp::build(OpBuilder &builder, OperationState &state,
                               TypeRange resultTypes, ValueRange resultDims,
                               ValueRange operands, ValueRange operandDims,
                               ArrayRef<int64_t> tiedOperands,
                               ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addOperands(operandDims);
  state.addOperands(resultDims);
  state.addAttributes(attributes);
  state.attributes.erase(TiedOpInterface::getStorageAttrName());
  state.addAttribute(TiedOpInterface::getStorageAttrName(),
                     builder.getIndexArrayAttr(tiedOperands));
  state.attributes.erase("operand_segment_sizes");
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr({
                         static_cast<int32_t>(operands.size()),
                         static_cast<int32_t>(operandDims.size()),
                         static_cast<int32_t>(resultDims.size()),
                     }));
  state.addRegion();
}

static LogicalResult verifyExStreamFragmentOp(ExStreamFragmentOp op) {
  if (failed(verifyOpDynamicDims(op, op.operands(), op.operand_dims())) ||
      failed(verifyOpDynamicDims(op, op.results(), op.result_dims()))) {
    return failure();
  }
  return success();
}

static ParseResult parseStreamFragmentBody(OpAsmParser &parser,
                                           TypeRange operandTypes,
                                           TypeRange resultTypes,
                                           ArrayAttr tiedOperands,
                                           Region &body) {
  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 16> regionArgs;
  SmallVector<Type, 16> regionArgTypes;
  if (failed(parser.parseLParen())) {
    return failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    do {
      // Reserve entries in the lists.
      regionArgs.emplace_back();
      regionArgTypes.emplace_back();
      if (failed(parser.parseRegionArgument(regionArgs.back())) ||
          failed(parser.parseColonType(regionArgTypes.back()))) {
        return failure();
      }
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRParen())) {
      return failure();
    }
  }

  SmallVector<Type, 4> regionResultTypes;
  if (failed(parser.parseArrowTypeList(regionResultTypes))) return failure();

  if (regionArgs.size() != operandTypes.size()) {
    return parser.emitError(loc, "region operand list mismatch");
  }
  if (regionResultTypes.size() != resultTypes.size()) {
    return parser.emitError(loc, "region result list mismatch");
  }

  return parser.parseRegion(body, regionArgs, regionArgTypes,
                            /*enableNameShadowing=*/true);
}

static void printStreamFragmentBody(OpAsmPrinter &p, Operation *op,
                                    TypeRange operandTypes,
                                    TypeRange resultTypes,
                                    ArrayAttr tiedOperands, Region &body) {
  p << "(";
  llvm::interleaveComma(body.getArguments(), p, [&](BlockArgument arg) {
    p << arg;
    p << ": ";
    p << arg.getType();
  });
  p << ") -> ";
  if (resultTypes.size() != 1) p << "(";
  for (unsigned i = 0; i < resultTypes.size(); ++i) {
    p.printType(resultTypes[i]);
    if (i < resultTypes.size() - 1) p << ", ";
  }
  if (resultTypes.size() != 1) p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

Value ExStreamFragmentOp::buildOperandRankedShape(unsigned idx,
                                                  OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getOperands(),
                                               operand_dims(), builder);
}

Value ExStreamFragmentOp::buildResultRankedShape(unsigned idx,
                                                 OpBuilder &builder) {
  return Shape::buildRankedShapeForValueInList(getLoc(), idx, getResults(),
                                               result_dims(), builder);
}

Operation::operand_range ExStreamFragmentOp::getClosureOperands() {
  return operands();
}

Operation::result_range ExStreamFragmentOp::getClosureResults() {
  return results();
}

bool ExStreamFragmentOp::canClosureContainOp(Operation *op) {
  // NOTE: we widen support on new stream ops only - the legacy path isn't worth
  // upgrading to support more.
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    return constantOp.getType().isIntOrIndexOrFloat();
  }
  if (auto loadOp = dyn_cast<VariableLoadOp>(op)) {
    // Only allow loads of immutable variables to move into the stream.
    // As they are immutable it's always safe to do so as no synchronization at
    // the stream entry/exit boundary is required.
    //
    // Loads of mutable variables may sometimes be safe to move in as well
    // however that is best done when we have better cross-stream
    // synchronization support and can make those guarantees structurally.
    auto variableOp =
        SymbolTable::lookupNearestSymbolFrom<VariableOp>(op, loadOp.variable());
    return variableOp.is_mutable() == false;
  }
  return false;
}

ClosureOpInterface
ExStreamFragmentOp::cloneReplacementExcludingOperandsAndResults(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices) {
  SmallVector<Type, 4> newResultTypes = llvm::to_vector<4>(getResultTypes());
  SmallVector<Value, 4> newResultDims = llvm::to_vector<4>(result_dims());
  SmallVector<Value, 4> newOperandsValues = llvm::to_vector<4>(operands());
  SmallVector<Value, 4> newOperandDims = llvm::to_vector<4>(operand_dims());
  excludeClosureOperandsAndResults(newOperandsValues, newOperandDims,
                                   excludedOperandIndices, newResultTypes,
                                   newResultDims, excludedResultIndices);

  auto newTiedOperandIndices =
      llvm::to_vector<4>(getTiedResultOperandIndices());
  excludeTiedOperandAndResultIndices(
      excludedOperandIndices, excludedResultIndices, newTiedOperandIndices);
  assert(getTiedOperandsIndexAndLength().first == 0 &&
         "operands must be the first ODS group");

  auto newOp = OpBuilder(getContext())
                   .create<ExStreamFragmentOp>(
                       getLoc(), newResultTypes, newResultDims,
                       newOperandsValues, newOperandDims, newTiedOperandIndices,
                       getOperation()->getAttrs());
  auto &newBody = newOp.getClosureBodyRegion();
  newBody.takeBody(getClosureBodyRegion());
  eraseRegionResults(newBody, excludedResultIndices);
  newBody.front().eraseArguments(excludedOperandIndices);
  return newOp;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc"
