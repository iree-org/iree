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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

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
  p.printOptionalAttrDictWithKeyword(op.getAttrs(), /*elidedAttrs=*/{
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

llvm::Optional<std::pair<DispatchRegionOp, Operation *>>
DispatchRegionOp::formFromAnchorOp(Value workload, Operation *anchorOp,
                                   OpBuilder &builder) {
  builder.setInsertionPoint(anchorOp);
  auto loc = anchorOp->getLoc();
  // Map anchor into new dispatch region.
  llvm::SmallVector<Value, 4> capturedInputs(anchorOp->getOperands());
  auto drOp = builder.create<DispatchRegionOp>(
      loc, llvm::to_vector<1>(anchorOp->getResultTypes()), workload,
      capturedInputs);
  auto *drBlock = new Block();
  drOp.body().push_back(drBlock);
  BlockAndValueMapping mapping;
  for (Value capturedInput : capturedInputs) {
    auto blockArg = drBlock->addArgument(capturedInput.getType());
    mapping.map(capturedInput, blockArg);
  }

  // Create new body.
  builder.setInsertionPointToEnd(drBlock);
  auto *newAnchorOp = builder.clone(*anchorOp, mapping);
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
  self.replaceAllUsesWith(ResultRange(newDrOp, 0, origNumResults));
  self.erase();
  self = newDrOp;

  // Add results to the terminator.
  auto terminator = block.getTerminator();
  llvm::SmallVector<Value, 4> returns(terminator->getOperands());
  returns.append(addlResults.begin(), addlResults.end());
  terminator->setOperands(returns);

  return ResultRange(self, origNumResults, addlResults.size());
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

  // Add arguments for any op arguments that need to be captured.
  for (Value newArgument : origOp->getOperands()) {
    if (mapping.contains(newArgument)) continue;
    getOperation()->insertOperands(getNumOperands(), {newArgument});
    Value newBlockArgument = block.addArgument(newArgument.getType());
    mapping.map(newArgument, newBlockArgument);
  }

  // Clone the op.
  Operation *inlinedOp = builder.clone(*origOp, mapping);

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
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{});
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
      op.getAttrs(),
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
      op.getAttrs(), /*elidedAttrs=*/{"function_ref", "sym_name"});
}

//===----------------------------------------------------------------------===//
// flow.dispatch
//===----------------------------------------------------------------------===//

static ParseResult parseDispatchOp(OpAsmParser &parser,
                                   OperationState *result) {
  auto executableLoc = parser.getNameLoc();

  // TODO(benvanik): replace with SymbolRefAttr.
  StringAttr executableAttr;
  StringAttr entryPointAttr;
  if (failed(parser.parseSymbolName(executableAttr, "executable",
                                    result->attributes)) ||
      failed(parser.parseColon()) || failed(parser.parseColon()) ||
      failed(parser.parseSymbolName(entryPointAttr, "entry_point",
                                    result->attributes))) {
    return failure();
  }
  result->attributes.set("entry_point", parser.getBuilder().getSymbolRefAttr(
                                            entryPointAttr.getValue()));
  result->attributes.set("executable", parser.getBuilder().getSymbolRefAttr(
                                           executableAttr.getValue()));

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

  SmallVector<OpAsmParser::OperandType, 4> operands;
  FunctionType entryPointType;
  if (failed(
          parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(entryPointType)) ||
      failed(
          parser.addTypesToList(entryPointType.getResults(), result->types)) ||
      failed(parser.resolveOperands(operands, entryPointType.getInputs(),
                                    executableLoc, result->operands))) {
    return failure();
  }
  return success();
}

static void printDispatchOp(OpAsmPrinter &p, DispatchOp op) {
  p << op.getOperationName() << ' ';
  // TODO(benvanik): replace with SymbolRefAttr.
  p.printSymbolName(op.executable());
  p << "::";
  p.printSymbolName(op.entry_point());
  p << "[";
  p.printOperand(op.workload());
  p << " : ";
  p.printType(op.workload().getType());
  p << "](";
  p.printOperands(op.operands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "executable",
                              "entry_point",
                          });
  p << " : ";
  p.printType(op.getEntryPointType());
}

FunctionType DispatchOp::getEntryPointType() {
  SmallVector<Type, 8> argTypes(operand_type_range{operands()});
  return FunctionType::get(argTypes, getResultTypes(), getContext());
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
  p.printOptionalAttrDict(op.getAttrs(),
                          /*elidedAttrs=*/{});
}

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowOps.cpp.inc"

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
