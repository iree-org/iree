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

#include "iree/compiler/Dialect/VM/IR/VMOps.h"

#include "iree/compiler/Dialect/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

static ParseResult parseRegionEndOp(OpAsmParser &parser,
                                    OperationState *result) {
  return parser.parseOptionalAttrDict(result->attributes);
}

static void printRegionEndOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName();
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseModuleOp(OpAsmParser &parser, OperationState *result) {
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
  ModuleOp::ensureTerminator(*body, parser.getBuilder(), result->location);
  return success();
}

static void printModuleOp(OpAsmPrinter &p, ModuleOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs=*/{mlir::SymbolTable::getSymbolAttrName()});
  p.printRegion(op.getBodyRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

void ModuleOp::build(Builder *builder, OperationState &result, StringRef name) {
  ensureTerminator(*result.addRegion(), *builder, result.location);
  result.attributes.push_back(builder->getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder->getStringAttr(name)));
}

static LogicalResult verifyModuleOp(ModuleOp op) {
  // TODO(benvanik): check export name conflicts.
  return success();
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState *result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results, impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };
  return impl::parseFunctionLikeOp(parser, *result, /*allowVariadic=*/false,
                                   buildFuncType);
}

static void printFuncOp(OpAsmPrinter &p, FuncOp &op) {
  FunctionType fnType = op.getType();
  impl::printFunctionLikeOp(p, op, fnType.getInputs(), /*isVariadic=*/false,
                            fnType.getResults());
}

void FuncOp::build(Builder *builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<NamedAttributeList> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty()) {
    return;
  }

  unsigned numInputs = type.getNumInputs();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0; i < numInputs; ++i) {
    if (auto argDict = argAttrs[i].getDictionary()) {
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
    }
  }
}

Block *FuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  entry->addArguments(getType().getInputs());
  return entry;
}

LogicalResult FuncOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

static ParseResult parseExportOp(OpAsmParser &parser, OperationState *result) {
  FlatSymbolRefAttr functionRefAttr;
  if (failed(parser.parseAttribute(functionRefAttr, "function_ref",
                                   result->attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    StringAttr exportNameAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(exportNameAttr, "export_name",
                                     result->attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    result->addAttribute("export_name", parser.getBuilder().getStringAttr(
                                            functionRefAttr.getValue()));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  return success();
}

static void printExportOp(OpAsmPrinter &p, ExportOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.function_ref());
  if (op.export_name() != op.function_ref()) {
    p << " as(\"" << op.export_name() << "\")";
  }
  p.printOptionalAttrDictWithKeyword(
      op.getAttrs(), /*elidedAttrs=*/{"function_ref", "export_name"});
}

void ExportOp::build(Builder *builder, OperationState &result,
                     FuncOp functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, builder->getSymbolRefAttr(functionRef),
        exportName.empty() ? functionRef.getName() : exportName, attrs);
}

void ExportOp::build(Builder *builder, OperationState &result,
                     FlatSymbolRefAttr functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("function_ref", functionRef);
  result.addAttribute("export_name", builder->getStringAttr(exportName));
  result.attributes.append(attrs.begin(), attrs.end());
}

static ParseResult parseImportOp(OpAsmParser &parser, OperationState *result) {
  auto builder = parser.getBuilder();
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseLParen())) {
    return parser.emitError(parser.getNameLoc()) << "invalid import name";
  }
  SmallVector<NamedAttributeList, 8> argAttrs;
  SmallVector<Type, 8> argTypes;
  while (failed(parser.parseOptionalRParen())) {
    OpAsmParser::OperandType operand;
    Type operandType;
    auto operandLoc = parser.getCurrentLocation();
    if (failed(parser.parseOperand(operand)) ||
        failed(parser.parseColonType(operandType))) {
      return parser.emitError(operandLoc) << "invalid operand";
    }
    argTypes.push_back(operandType);
    NamedAttributeList argAttrList;
    operand.name.consume_front("%");
    argAttrList.set(builder.getIdentifier("vm.name"),
                    builder.getStringAttr(operand.name));
    if (succeeded(parser.parseOptionalEllipsis())) {
      argAttrList.set(builder.getIdentifier("vm.variadic"),
                      builder.getUnitAttr());
    }
    argAttrs.push_back(argAttrList);
    if (failed(parser.parseOptionalComma())) {
      if (failed(parser.parseRParen())) {
        return parser.emitError(parser.getCurrentLocation())
               << "invalid argument list (expected rparen)";
      }
      break;
    }
  }
  SmallVector<Type, 8> resultTypes;
  if (failed(parser.parseOptionalArrowTypeList(resultTypes))) {
    return parser.emitError(parser.getCurrentLocation())
           << "invalid result type list";
  }
  for (int i = 0; i < argAttrs.size(); ++i) {
    SmallString<8> argName;
    mlir::impl::getArgAttrName(i, argName);
    result->addAttribute(argName, argAttrs[i].getDictionary());
  }
  if (failed(parser.parseOptionalAttrDictWithKeyword(result->attributes))) {
    return failure();
  }

  auto functionType =
      FunctionType::get(argTypes, resultTypes, result->getContext());
  result->addAttribute(mlir::impl::getTypeAttrName(),
                       TypeAttr::get(functionType));

  // No clue why this is required.
  result->addRegion();

  return success();
}

static void printImportOp(OpAsmPrinter &p, ImportOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.getName());
  p << "(";
  for (int i = 0; i < op.getNumFuncArguments(); ++i) {
    if (auto name = op.getArgAttrOfType<StringAttr>(i, "vm.name")) {
      p << '%' << name.getValue() << " : ";
    }
    p.printType(op.getType().getInput(i));
    if (op.getArgAttrOfType<UnitAttr>(i, "vm.variadic")) {
      p << "...";
    }
    if (i < op.getNumFuncArguments() - 1) {
      p << ", ";
    }
  }
  p << ")";
  if (op.getNumFuncResults() == 1) {
    p << " -> ";
    p.printType(op.getType().getResult(0));
  } else if (op.getNumFuncResults() > 1) {
    p << " -> (";
    interleaveComma(op.getType().getResults(), p);
    p << ")";
  }
  mlir::impl::printFunctionAttributes(p, op, op.getNumFuncArguments(),
                                      op.getNumFuncResults(),
                                      /*elided=*/
                                      {
                                          "is_variadic",
                                      });
}

void ImportOp::build(Builder *builder, OperationState &result, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<NamedAttributeList> argAttrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty()) {
    return;
  }

  unsigned numInputs = type.getNumInputs();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0; i < numInputs; ++i) {
    if (auto argDict = argAttrs[i].getDictionary()) {
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
    }
  }
}

LogicalResult ImportOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState *result) {
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

  return success();
}

static void printGlobalOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ';
  p.printSymbolName(
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue());
  if (op->getAttrOfType<UnitAttr>("is_mutable")) {
    p << " mutable";
  }
  if (auto initializer = op->getAttrOfType<FlatSymbolRefAttr>("initializer")) {
    p << " init(";
    p.printSymbolName(initializer.getValue());
    p << ')';
  }
  if (auto initialValue = op->getAttrOfType<IntegerAttr>("initial_value")) {
    p << ' ';
    p.printAttribute(initialValue);
  } else {
    p << " : ";
    p.printType(op->getAttrOfType<TypeAttr>("type").getValue());
  }
}

static LogicalResult verifyGlobalOp(Operation *op) {
  auto globalName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  auto globalType = op->getAttrOfType<TypeAttr>("type");
  auto initializerAttr = op->getAttrOfType<FlatSymbolRefAttr>("initializer");
  auto initialValueAttr = op->getAttr("initial_value");
  if (initializerAttr && initialValueAttr) {
    return op->emitOpError()
           << "globals can have either an initializer or an initial value";
  } else if (initializerAttr) {
    // Ensure initializer returns the same value as the global.
    auto initializer = op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
        initializerAttr.getValue());
    if (!initializer) {
      return op->emitOpError()
             << "initializer function " << initializerAttr << " not found";
    }
    if (initializer.getType().getNumInputs() != 0 ||
        initializer.getType().getNumResults() != 1 ||
        initializer.getType().getResult(0) != globalType.getValue()) {
      return op->emitOpError()
             << "initializer type mismatch; global " << globalName << " is "
             << globalType << " but initializer function "
             << initializer.getName() << " is " << initializer.getType();
    }
  } else if (initialValueAttr) {
    // Ensure the value is something we can convert to a const.
    if (initialValueAttr.getType() != globalType.getValue()) {
      return op->emitOpError()
             << "initial value type mismatch; global " << globalName << " is "
             << globalType << " but initial value provided is "
             << initialValueAttr.getType();
    }
  }
  return success();
}

void GlobalI32Op::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Optional<StringRef> initializer,
                        Optional<Attribute> initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder->getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder->getSymbolRefAttr(initializer.getValue()));
  } else if (initialValue.hasValue()) {
    result.addAttribute("initial_value", initialValue.getValue());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void GlobalI32Op::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable,
                        IREE::VM::FuncOp initializer,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void GlobalI32Op::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Attribute initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void GlobalI32Op::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

void GlobalRefOp::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Optional<StringRef> initializer,
                        Optional<Attribute> initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder->getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder->getSymbolRefAttr(initializer.getValue()));
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void GlobalRefOp::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable,
                        IREE::VM::FuncOp initializer,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void GlobalRefOp::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Attribute initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void GlobalRefOp::build(Builder *builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

static ParseResult parseGlobalLoadOp(OpAsmParser &parser,
                                     OperationState *result) {
  FlatSymbolRefAttr globalAttr;
  Type valueType;
  if (failed(parser.parseAttribute(globalAttr, "global", result->attributes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(valueType))) {
    return failure();
  }
  result->addTypes({valueType});
  return success();
}

static void printGlobalLoadOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ';
  p.printSymbolName(op->getAttrOfType<FlatSymbolRefAttr>("global").getValue());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"global"});
  p << " : ";
  p.printType(op->getResult(0)->getType());
}

static LogicalResult verifyGlobalLoadOp(Operation *op) {
  auto globalAttr = op->getAttrOfType<FlatSymbolRefAttr>("global");
  auto *globalOp =
      op->getParentOfType<VM::ModuleOp>().lookupSymbol(globalAttr.getValue());
  if (!globalOp) {
    return op->emitOpError() << "Undefined global: " << globalAttr;
  }
  auto globalType = globalOp->getAttrOfType<TypeAttr>("type");
  auto loadType = op->getResult(0)->getType();
  if (globalType.getValue() != loadType) {
    return op->emitOpError()
           << "Global type mismatch; global " << globalAttr << " is "
           << globalType << " but load is " << loadType;
  }
  return success();
}

static ParseResult parseGlobalStoreOp(OpAsmParser &parser,
                                      OperationState *result) {
  FlatSymbolRefAttr globalAttr;
  OpAsmParser::OperandType value;
  Type valueType;
  if (failed(parser.parseAttribute(globalAttr, "global", result->attributes)) ||
      failed(parser.parseComma()) || failed(parser.parseOperand(value)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(valueType)) ||
      failed(parser.resolveOperand(value, valueType, result->operands))) {
    return failure();
  }
  return success();
}

static void printGlobalStoreOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ';
  p.printSymbolName(op->getAttrOfType<FlatSymbolRefAttr>("global").getValue());
  p << ", ";
  p.printOperand(op->getOperand(0));
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"global"});
  p << " : ";
  p.printType(op->getOperand(0)->getType());
}

static LogicalResult verifyGlobalStoreOp(Operation *op) {
  auto globalAttr = op->getAttrOfType<FlatSymbolRefAttr>("global");
  auto *globalOp =
      op->getParentOfType<VM::ModuleOp>().lookupSymbol(globalAttr.getValue());
  if (!globalOp) {
    return op->emitOpError() << "Undefined global: " << globalAttr;
  }
  auto globalType = globalOp->getAttrOfType<TypeAttr>("type");
  auto storeType = op->getOperand(0)->getType();
  if (globalType.getValue() != storeType) {
    return op->emitOpError()
           << "Global type mismatch; global " << globalAttr << " is "
           << globalType << " but store is " << storeType;
  }
  if (!globalOp->getAttrOfType<UnitAttr>("is_mutable")) {
    return op->emitOpError() << "Global " << globalAttr
                             << " is not mutable and cannot be stored to";
  }
  return success();
}

static ParseResult parseGlobalResetRefOp(OpAsmParser &parser,
                                         OperationState *result) {
  FlatSymbolRefAttr globalAttr;
  if (failed(parser.parseAttribute(globalAttr, "global", result->attributes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes))) {
    return failure();
  }
  return success();
}

static void printGlobalResetRefOp(OpAsmPrinter &p, GlobalResetRefOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.global());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"global"});
}

static LogicalResult verifyGlobalResetRefOp(GlobalResetRefOp &op) {
  auto *globalOp = op.getParentOfType<VM::ModuleOp>().lookupSymbol(op.global());
  if (!globalOp) {
    return op.emitOpError() << "Undefined global: " << op.global();
  }
  auto globalType = globalOp->getAttrOfType<TypeAttr>("type");
  if (!globalType.getValue().isa<RefPtrType>()) {
    return op.emitOpError() << "Global type mismatch; global " << op.global()
                            << " is " << globalType << " and not a ref_ptr";
  }
  if (!globalOp->getAttrOfType<UnitAttr>("is_mutable")) {
    return op.emitOpError() << "Global " << op.global()
                            << " is not mutable and cannot be stored to";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static ParseResult parseConstI32Op(OpAsmParser &parser,
                                   OperationState *result) {
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Failed to parse optional attribute dict";
  }

  Attribute valueAttr;
  SmallVector<NamedAttribute, 1> dummyAttrs;
  if (failed(parser.parseAttribute(valueAttr, "value", dummyAttrs))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Invalid attribute encoding";
  }
  if (!ConstI32Op::isBuildableWith(valueAttr, valueAttr.getType())) {
    return parser.emitError(parser.getCurrentLocation())
           << "Incompatible type or invalid type value formatting";
  }
  valueAttr = ConstI32Op::convertConstValue(valueAttr);
  result->addAttribute("value", valueAttr);
  return parser.addTypeToList(valueAttr.getType(), result->types);
}

static void printConstI32Op(OpAsmPrinter &p, ConstI32Op &op) {
  p << op.getOperationName() << ' ';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
  p.printAttribute(op.value());
}

// static
bool ConstI32Op::isBuildableWith(Attribute value, Type type) {
  // FlatSymbolRefAttr can only be used with a function type.
  if (value.isa<FlatSymbolRefAttr>()) {
    return false;
  }
  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type) {
    return false;
  }
  // Finally, check that the attribute kind is handled.
  return value.isa<UnitAttr>() || value.isa<BoolAttr>() ||
         value.isa<IntegerAttr>() ||
         (value.isa<ElementsAttr>() && value.cast<ElementsAttr>()
                                           .getType()
                                           .getElementType()
                                           .isa<IntegerType>());
}

// static
Attribute ConstI32Op::convertConstValue(Attribute value) {
  assert(isBuildableWith(value, value.getType()));
  Builder builder(value.getContext());
  int32_t dims = 1;
  Attribute newValue;
  if (value.isa<UnitAttr>()) {
    return builder.getI32IntegerAttr(1);
  } else if (auto v = value.dyn_cast<BoolAttr>()) {
    return builder.getI32IntegerAttr(v.getValue() ? 1 : 0);
  } else if (auto v = value.dyn_cast<IntegerAttr>()) {
    return builder.getI32IntegerAttr(
        static_cast<int32_t>(v.getValue().getLimitedValue()));
  } else if (auto v = value.dyn_cast<ElementsAttr>()) {
    dims = v.getNumElements();
    ShapedType adjustedType =
        VectorType::get({dims}, builder.getIntegerType(32));
    if (auto elements = v.dyn_cast<SplatElementsAttr>()) {
      return SplatElementsAttr::get(adjustedType, elements.getSplatValue());
    } else {
      return DenseElementsAttr::get(
          adjustedType, llvm::to_vector<4>(v.getValues<Attribute>()));
    }
  }
  llvm_unreachable("unexpected attribute type");
  return Attribute();
}

void ConstI32Op::build(Builder *builder, OperationState &result,
                       Attribute value) {
  Attribute newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstI32Op::build(Builder *builder, OperationState &result,
                       int32_t value) {
  return build(builder, result, builder->getI32IntegerAttr(value));
}

static ParseResult parseConstI32ZeroOp(OpAsmParser &parser,
                                       OperationState *result) {
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Failed to parse optional attribute dict";
  }

  Type valueType;
  if (failed(parser.parseColonType(valueType))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Invalid integer type";
  }
  return parser.addTypeToList(valueType, result->types);
}

static void printConstI32ZeroOp(OpAsmPrinter &p, ConstI32ZeroOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.getResult()->getType());
}

void ConstI32ZeroOp::build(Builder *builder, OperationState &result) {
  result.addTypes(builder->getIntegerType(32));
}

static ParseResult parseConstRefZeroOp(OpAsmParser &parser,
                                       OperationState *result) {
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Failed to parse optional attribute dict";
  }

  Type objectType;
  if (failed(parser.parseColonType(objectType))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Invalid ref_ptr type";
  }
  return parser.addTypeToList(objectType, result->types);
}

static void printConstRefZeroOp(OpAsmPrinter &p, ConstRefZeroOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.getResult()->getType());
}

void ConstRefZeroOp::build(Builder *builder, OperationState &result,
                           Type objectType) {
  result.addTypes(objectType);
}

static ParseResult parseRodataOp(OpAsmParser &parser, OperationState *result) {
  StringAttr nameAttr;
  Attribute valueAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result->attributes)) ||
      failed(parser.parseAttribute(valueAttr, "value", result->attributes))) {
    return failure();
  }
  return success();
}

static void printRodataOp(OpAsmPrinter &p, RodataOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.sym_name());
  p << ' ';
  p.printAttribute(op.value());
}

void RodataOp::build(Builder *builder, OperationState &result, StringRef name,
                     ElementsAttr value, ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("sym_name", builder->getStringAttr(name));
  result.addAttribute("value", value);
  result.addAttributes(attrs);
}

static ParseResult parseConstRefRodataOp(OpAsmParser &parser,
                                         OperationState *result) {
  FlatSymbolRefAttr rodataAttr;
  Type valueType;
  if (failed(parser.parseAttribute(rodataAttr, "rodata", result->attributes)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(valueType))) {
    return failure();
  }
  result->addTypes({valueType});
  return success();
}

static void printConstRefRodataOp(OpAsmPrinter &p, ConstRefRodataOp &op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.rodata());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"rodata"});
  p << " : ";
  p.printType(op.value()->getType());
}

static LogicalResult verifyConstRefRodataOp(ConstRefRodataOp &op) {
  auto *rodataOp = op.getParentOfType<VM::ModuleOp>().lookupSymbol(op.rodata());
  if (!rodataOp) {
    return op.emitOpError() << "Undefined rodata section: " << op.rodata();
  }
  return success();
}

void ConstRefRodataOp::build(Builder *builder, OperationState &result,
                             StringRef rodataName,
                             ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("rodata", builder->getSymbolRefAttr(rodataName));
  auto type = RefPtrType::get(ByteBufferType::get(builder->getContext()));
  result.addTypes({type});
  result.addAttributes(attrs);
}

void ConstRefRodataOp::build(Builder *builder, OperationState &result,
                             RodataOp rodataOp,
                             ArrayRef<NamedAttribute> attrs) {
  build(builder, result, rodataOp.getName(), attrs);
}

//===----------------------------------------------------------------------===//
// ref_ptr operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Conditional assignment
//===----------------------------------------------------------------------===//

static ParseResult parseSelectOp(OpAsmParser &parser, OperationState *result) {
  OpAsmParser::OperandType condOperand;
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (failed(parser.parseOperand(condOperand)) ||
      failed(parser.resolveOperand(condOperand,
                                   parser.getBuilder().getIntegerType(32),
                                   result->operands)) ||
      failed(parser.parseComma()) || failed(parser.parseOperandList(ops, 2)) ||
      failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperands(ops, type, result->operands))) {
    return failure();
  }
  result->addTypes({type});
  return success();
}

static void printSelectOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1)
    << ", " << *op->getOperand(2);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// Native integer arithmetic
//===----------------------------------------------------------------------===//

static ParseResult parseUnaryArithmeticOp(OpAsmParser &parser,
                                          OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 1> ops;
  Type type;
  if (parser.parseOperandList(ops, 1) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result->operands)) {
    return failure();
  }
  result->addTypes({type});
  return success();
}

static void printUnaryArithmeticOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0)->getType();
}

static ParseResult parseBinaryArithmeticOp(OpAsmParser &parser,
                                           OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (parser.parseOperandList(ops, 2) || parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result->operands)) {
    return failure();
  }
  result->addTypes({type});
  return success();
}

static void printBinaryArithmeticOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===----------------------------------------------------------------------===//

static ParseResult parseShiftArithmeticOp(OpAsmParser &parser,
                                          OperationState *result) {
  OpAsmParser::OperandType op;
  Type type;
  IntegerAttr amount;
  if (failed(parser.parseOperand(op)) || failed(parser.parseComma()) ||
      failed(parser.parseAttribute(amount,
                                   parser.getBuilder().getIntegerType(8),
                                   "amount", result->attributes)) ||
      failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperand(op, type, result->operands))) {
    return failure();
  }
  result->addTypes({type});
  return success();
}

static void printShiftArithmeticOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0) << ", "
    << op->getAttrOfType<IntegerAttr>("amount").getInt();
  p.printOptionalAttrDict(op->getAttrs(), {"amount"});
  p << " : " << op->getResult(0)->getType();
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Native reduction (horizontal) arithmetic
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

static ParseResult parseUnaryComparisonOp(OpAsmParser &parser,
                                          OperationState *result) {
  OpAsmParser::OperandType op;
  Type type;
  if (failed(parser.parseOperand(op)) || failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperand(op, type, result->operands))) {
    return failure();
  }
  result->addTypes({IntegerType::get(32, result->getContext())});
  return success();
}

static void printUnaryComparisonOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0)->getType();
}

static ParseResult parseBinaryComparisonOp(OpAsmParser &parser,
                                           OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type type;
  if (failed(parser.parseOperandList(ops, 2)) ||
      failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperands(ops, type, result->operands))) {
    return failure();
  }
  result->addTypes({IntegerType::get(32, result->getContext())});
  return success();
}

static void printBinaryComparisonOp(OpAsmPrinter &p, Operation *op) {
  p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0)->getType();
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

static ParseResult parseBranchOp(OpAsmParser &parser, OperationState *result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result->addSuccessor(dest, destOperands);
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return failure();
  }
  return success();
}

static void printBranchOp(OpAsmPrinter &p, BranchOp &op) {
  p << op.getOperationName() << ' ';
  p.printSuccessorAndUseList(op.getOperation(), 0);
  p.printOptionalAttrDict(op.getAttrs());
}

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState *result) {
  SmallVector<Value *, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type int32Ty = parser.getBuilder().getIntegerType(32);
  if (failed(parser.parseOperand(condInfo)) || failed(parser.parseComma()) ||
      failed(parser.resolveOperand(condInfo, int32Ty, result->operands))) {
    return parser.emitError(parser.getNameLoc(),
                            "expected condition type was boolean (i32)");
  }

  // Parse the true successor.
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result->addSuccessor(dest, destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (failed(parser.parseComma()) ||
      failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result->addSuccessor(dest, destOperands);

  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return failure();
  }

  return success();
}

static void printCondBranchOp(OpAsmPrinter &p, CondBranchOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::trueIndex);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), CondBranchOp::falseIndex);
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseCallOp(OpAsmParser &parser, OperationState *result) {
  FlatSymbolRefAttr calleeAttr;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto calleeLoc = parser.getNameLoc();
  if (failed(parser.parseAttribute(calleeAttr, "callee", result->attributes)) ||
      failed(
          parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(calleeType)) ||
      failed(parser.addTypesToList(calleeType.getResults(), result->types)) ||
      failed(parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                                    result->operands))) {
    return failure();
  }
  return success();
}

static void printCallOp(OpAsmPrinter &p, CallOp &op) {
  p << op.getOperationName() << ' ' << op.getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : ";
  p.printType(FunctionType::get(llvm::to_vector<4>(op.getOperandTypes()),
                                llvm::to_vector<4>(op.getResultTypes()),
                                op.getContext()));
}

static ParseResult parseCallVariadicOp(OpAsmParser &parser,
                                       OperationState *result) {
  FlatSymbolRefAttr calleeAttr;
  FunctionType calleeType;
  auto calleeLoc = parser.getNameLoc();
  if (failed(parser.parseAttribute(calleeAttr, "callee", result->attributes)) ||
      failed(parser.parseLParen())) {
    return parser.emitError(calleeLoc) << "invalid callee symbol";
  }

  SmallVector<OpAsmParser::OperandType, 4> flatOperands;
  SmallVector<int8_t, 4> segmentSizes;
  while (failed(parser.parseOptionalRParen())) {
    if (succeeded(parser.parseOptionalLSquare())) {
      // Variadic list.
      SmallVector<OpAsmParser::OperandType, 4> segmentOperands;
      while (failed(parser.parseOptionalRSquare())) {
        OpAsmParser::OperandType segmentOperand;
        if (failed(parser.parseOperand(segmentOperand))) {
          return parser.emitError(parser.getCurrentLocation())
                 << "invalid operand";
        }
        segmentOperands.push_back(segmentOperand);
        if (failed(parser.parseOptionalComma())) {
          if (failed(parser.parseRSquare())) {
            return parser.emitError(parser.getCurrentLocation())
                   << "malformed variadic operand list";
          }
          break;
        }
      }
      segmentSizes.push_back(segmentOperands.size());
      flatOperands.append(segmentOperands.begin(), segmentOperands.end());
    } else {
      // Normal single operand.
      OpAsmParser::OperandType operand;
      if (failed(parser.parseOperand(operand))) {
        return parser.emitError(parser.getCurrentLocation())
               << "malformed non-variadic operand";
      }
      segmentSizes.push_back(-1);
      flatOperands.push_back(operand);
    }
    if (failed(parser.parseOptionalComma())) {
      if (failed(parser.parseRParen())) {
        return parser.emitError(parser.getCurrentLocation())
               << "expected closing )";
      }
      break;
    }
  }

  if (failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColon()) || failed(parser.parseLParen())) {
    return parser.emitError(parser.getCurrentLocation())
           << "malformed optional attributes list";
  }
  SmallVector<Type, 4> flatOperandTypes;
  SmallVector<Type, 4> segmentTypes;
  int segmentIndex = 0;
  while (failed(parser.parseOptionalRParen())) {
    Type operandType;
    if (failed(parser.parseType(operandType))) {
      return parser.emitError(parser.getCurrentLocation())
             << "invalid operand type";
    }
    bool isVariadic = succeeded(parser.parseOptionalEllipsis());
    if (isVariadic) {
      for (int i = 0; i < segmentSizes[segmentIndex]; ++i) {
        flatOperandTypes.push_back(operandType);
      }
    } else {
      flatOperandTypes.push_back(operandType);
    }
    segmentTypes.push_back(operandType);
    ++segmentIndex;

    if (failed(parser.parseOptionalComma())) {
      if (failed(parser.parseRParen())) {
        return parser.emitError(parser.getCurrentLocation())
               << "expected closing )";
      }
      break;
    }
  }
  if (failed(parser.resolveOperands(flatOperands, flatOperandTypes, calleeLoc,
                                    result->operands))) {
    return parser.emitError(parser.getCurrentLocation())
           << "operands do not match type list";
  }
  result->addAttribute(
      "segment_sizes",
      DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                          parser.getBuilder().getIntegerType(8)),
          segmentSizes));
  result->addAttribute("segment_types",
                       parser.getBuilder().getArrayAttr(llvm::to_vector<4>(
                           llvm::map_range(segmentTypes, [&](Type type) {
                             return TypeAttr::get(type).cast<Attribute>();
                           }))));

  if (failed(parser.parseOptionalArrowTypeList(result->types))) {
    return parser.emitError(parser.getCurrentLocation())
           << "malformed function type results";
  }

  return success();
}

static void printCallVariadicOp(OpAsmPrinter &p, CallVariadicOp &op) {
  p << op.getOperationName() << ' ' << op.getAttr("callee") << '(';
  int operand = 0;
  interleaveComma(op.segment_sizes(), p, [&](APInt segmentSize) {
    if (segmentSize.getSExtValue() == -1) {
      p.printOperand(op.getOperand(operand++));
    } else {
      p << '[';
      SmallVector<Value *, 4> segmentOperands;
      for (int i = 0; i < segmentSize.getZExtValue(); ++i) {
        segmentOperands.push_back(op.getOperand(operand++));
      }
      interleaveComma(segmentOperands, p,
                      [&](Value *operand) { p.printOperand(operand); });
      p << ']';
    }
  });
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{
                              "callee",
                              "segment_sizes",
                              "segment_types",
                          });
  p << " : (";
  interleaveComma(
      llvm::zip(op.segment_sizes(), op.segment_types()), p,
      [&](std::tuple<APInt, Attribute> segmentSizeType) {
        int segmentSize = std::get<0>(segmentSizeType).getSExtValue();
        Type segmentType =
            std::get<1>(segmentSizeType).cast<TypeAttr>().getValue();
        if (segmentSize == -1) {
          p.printType(segmentType);
        } else {
          p.printType(segmentType);
          p << "...";
        }
      });
  p << ")";
  if (op.getNumResults() == 1) {
    p << " -> ";
    p.printType(op.getResult(0)->getType());
  } else if (op.getNumResults() > 1) {
    p << " -> (";
    interleaveComma(op.getResultTypes(), p);
    p << ")";
  }
}

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result->operands));
}

static void printReturnOp(OpAsmPrinter &p, ReturnOp &op) {
  p << op.getOperationName();
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.operand_begin(), op.operand_end());
    p.printOptionalAttrDict(op.getAttrs());
    p << " : ";
    interleaveComma(op.getOperandTypes(), p);
  }
}

//===----------------------------------------------------------------------===//
// Async/fiber ops
//===----------------------------------------------------------------------===//

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState *result) {
  return parser.parseOptionalAttrDict(result->attributes);
}

static void printYieldOp(OpAsmPrinter &p, YieldOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

static ParseResult parseTraceOp(OpAsmParser &parser, OperationState *result) {
  StringAttr eventNameAttr;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> operandTypes;
  auto eventNameLoc = parser.getNameLoc();
  if (failed(parser.parseAttribute(eventNameAttr, "event_name",
                                   result->attributes)) ||
      failed(
          parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonTypeList(operandTypes)) ||
      failed(parser.resolveOperands(operands, operandTypes, eventNameLoc,
                                    result->operands))) {
    return failure();
  }
  return success();
}

static void printTraceOp(OpAsmPrinter &p, TraceOp &op) {
  p << op.getOperationName() << " " << op.getAttr("event_name") << "(";
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"event_name"});
  p << " : ";
  interleaveComma(op.getOperandTypes(), p);
}

static ParseResult parsePrintOp(OpAsmParser &parser, OperationState *result) {
  StringAttr messageAttr;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> operandTypes;
  auto messageLoc = parser.getNameLoc();
  if (failed(
          parser.parseAttribute(messageAttr, "message", result->attributes)) ||
      failed(
          parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonTypeList(operandTypes)) ||
      failed(parser.resolveOperands(operands, operandTypes, messageLoc,
                                    result->operands))) {
    return failure();
  }
  return success();
}

static void printPrintOp(OpAsmPrinter &p, PrintOp &op) {
  p << op.getOperationName() << " " << op.getAttr("message") << "(";
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"message"});
  p << " : ";
  interleaveComma(op.getOperandTypes(), p);
}

static ParseResult parseBreakOp(OpAsmParser &parser, OperationState *result) {
  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result->addSuccessor(dest, destOperands);
  return success();
}

static void printBreakOp(OpAsmPrinter &p, BreakOp &op) {
  p << op.getOperationName() << ' ';
  p.printSuccessorAndUseList(op.getOperation(), 0);
}

Block *BreakOp::getDest() { return getOperation()->getSuccessor(0); }

void BreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

static ParseResult parseCondBreakOp(OpAsmParser &parser,
                                    OperationState *result) {
  // Parse the condition.
  OpAsmParser::OperandType condInfo;
  Type int32Ty = parser.getBuilder().getIntegerType(32);
  if (failed(parser.parseOperand(condInfo)) || failed(parser.parseComma()) ||
      failed(parser.resolveOperand(condInfo, int32Ty, result->operands))) {
    return parser.emitError(parser.getNameLoc(),
                            "expected condition type was boolean (i32)");
  }

  Block *dest;
  SmallVector<Value *, 4> destOperands;
  if (failed(parser.parseSuccessorAndUseList(dest, destOperands))) {
    return failure();
  }
  result->addSuccessor(dest, destOperands);
  return success();
}

static void printCondBreakOp(OpAsmPrinter &p, CondBreakOp &op) {
  p << op.getOperationName() << ' ';
  p.printOperand(op.condition());
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), 0);
}

Block *CondBreakOp::getDest() { return getOperation()->getSuccessor(0); }

void CondBreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void CondBreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseSuccessorOperand(0, index);
}

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/VM/IR/VMOpEncoder.cpp.inc"
#include "iree/compiler/Dialect/VM/IR/VMOps.cpp.inc"

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
