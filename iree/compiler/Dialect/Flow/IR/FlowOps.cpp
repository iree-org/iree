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
#include "llvm/ADT/StringExtras.h"
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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Returns true if the given |accessType| is compatible with the |variableType|.
// For example, this will return true if the variable type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isVariableTypeCompatible(Type variableType, Type accessType) {
  return succeeded(mlir::verifyCompatibleShape(variableType, accessType));
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

void DispatchRegionOp::dceOperandsAndResults(DispatchRegionOp &op) {
  OpBuilder builder(op.getContext());
  ClosureOpDce dce(op, op.body().front(), /*variadicOffset=*/1);
  op = llvm::cast<DispatchRegionOp>(dce.optimize(builder));
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
    p << " -> ";
    if (op.getNumResults() > 1) p << "(";
    interleaveComma(op.getResultTypes(), p);
    if (op.getNumResults() > 1) p << ")";
  }

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{});
}

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

void DispatchWorkgroupsOp::build(OpBuilder &builder, OperationState &state,
                                 ValueRange workgroupCount,
                                 TypeRange resultTypes, ValueRange operands,
                                 ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(workgroupCount);
  state.addOperands(operands);
  state.addAttributes(attributes);
  state.addAttribute(
      "operand_segment_sizes",
      builder.getI32VectorAttr({static_cast<int32_t>(workgroupCount.size()),
                                static_cast<int32_t>(operands.size())}));

  auto *body = state.addRegion();
  assert(body->begin() == body->end());
  {
    OpBuilder::InsertionGuard g(builder);
    builder.createBlock(body);  // createBlock implicitly moves IP, RAII away...
  }
  for (auto operand : operands) {
    Type type = operand.getType();
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchInputType::get(tensorType);
    }
    body->addArgument(type);
  }
  for (auto resultType : resultTypes) {
    Type type = resultType;
    if (auto tensorType = type.dyn_cast<TensorType>()) {
      type = DispatchOutputType::get(tensorType);
    }
    body->addArgument(type);
  }
  assert(std::next(body->begin()) == body->end());
}

static ParseResult parseDispatchWorkgroupBody(OpAsmParser &parser,
                                              TypeRange operandTypes,
                                              TypeRange resultTypes,
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

  if (regionArgs.size() != operandTypes.size() + resultTypes.size()) {
    return parser.emitError(loc,
                            "region operand list required required to match "
                            "count of dispatch op operands + results");
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
    p << " : ";
    p << arg.getType();
  });
  p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

// TODO(benvanik): remove after https://bugs.llvm.org/show_bug.cgi?id=48478
// The parser/printer are modified autogenerated values to work around the bug.

static ::mlir::ParseResult parseDispatchWorkgroupsOp(
    ::mlir::OpAsmParser &parser, ::mlir::OperationState *result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4>
      workgroup_countOperands;
  ::llvm::SMLoc workgroup_countOperandsLoc;
  (void)workgroup_countOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> operandsTypes;
  ::llvm::ArrayRef<::mlir::Type> resultsTypes;
  std::unique_ptr<::mlir::Region> bodyRegion =
      std::make_unique<::mlir::Region>();
  if (parser.parseLSquare()) return ::mlir::failure();

  workgroup_countOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(workgroup_countOperands))
    return ::mlir::failure();
  if (parser.parseRSquare()) return ::mlir::failure();
  if (parser.parseLParen()) return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands)) return ::mlir::failure();
  if (parser.parseRParen()) return ::mlir::failure();
  if (parser.parseColon()) return ::mlir::failure();

  ::mlir::FunctionType operands__results_functionType;
  if (parser.parseType(operands__results_functionType))
    return ::mlir::failure();
  operandsTypes = operands__results_functionType.getInputs();
  resultsTypes = operands__results_functionType.getResults();
  if (parser.parseOptionalAttrDictWithKeyword(result->attributes))
    return ::mlir::failure();
  if (parser.parseEqual()) return ::mlir::failure();
  {
    if (parseDispatchWorkgroupBody(parser, operandsTypes, resultsTypes,
                                   *bodyRegion))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result->addTypes(resultsTypes);
  if (parser.resolveOperands(workgroup_countOperands, odsBuildableType0,
                             result->operands))
    return ::mlir::failure();
  if (parser.resolveOperands(operandsOperands, operandsTypes,
                             operandsOperandsLoc, result->operands))
    return ::mlir::failure();
  result->addRegion(std::move(bodyRegion));
  result->addAttribute(
      "operand_segment_sizes",
      parser.getBuilder().getI32VectorAttr(
          {static_cast<int32_t>(workgroup_countOperands.size()),
           static_cast<int32_t>(operandsOperands.size())}));
  return ::mlir::success();
}

static void printDispatchWorkgroupsOp(::mlir::OpAsmPrinter &p,
                                      DispatchWorkgroupsOp &op) {
  p << "flow.dispatch.workgroups";
  p << "[";
  p << op.workgroup_count();
  p << "]";
  p << ' ' << "(";
  p << op.operands();
  p << ")";
  p << ' ' << ":";
  p << ' ';
  p.printFunctionalType(op.operands().getTypes(), op.results().getTypes());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), /*elidedAttrs=*/{
                                         "operand_segment_sizes",
                                     });
  p << ' ' << "=";
  p << ' ';
  printDispatchWorkgroupBody(p, op, op.operands().getTypes(),
                             op.results().getTypes(), op.body());
}

static LogicalResult verifyDispatchWorkgroupsOp(DispatchWorkgroupsOp op) {
  if (op.workgroup_count().empty()) {
    return op.emitOpError() << "at least one workgroup dimension is required";
  }
  return success();
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
                       TypeRange results, ValueRange operands,
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
  state.addTypes(results);
  state.addOperands(operands);
  state.addAttributes(attributes);
  state.addAttribute(
      "operand_segment_sizes",
      builder.getI32VectorAttr({static_cast<int32_t>(workgroupCount.size()),
                                static_cast<int32_t>(operands.size())}));
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
  return success();
}

//===----------------------------------------------------------------------===//
// flow.ex.stream.fragment
//===----------------------------------------------------------------------===//

void ExStreamFragmentOp::build(OpBuilder &builder, OperationState &state,
                               ArrayRef<Type> resultTypes, ValueRange operands,
                               ArrayRef<NamedAttribute> attributes) {
  state.addTypes(resultTypes);
  state.addOperands(operands);
  state.addAttributes(attributes);
  state.addRegion();
}

ParseResult parseExStreamFragmentOp(OpAsmParser &parser,
                                    OperationState *result) {
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

void printExStreamFragmentOp(OpAsmPrinter &p, ExStreamFragmentOp op) {
  p << op.getOperationName();

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
    p << " -> ";
    if (op.getNumResults() > 1) p << "(";
    interleaveComma(op.getResultTypes(), p);
    if (op.getNumResults() > 1) p << ")";
  }

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{});
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
