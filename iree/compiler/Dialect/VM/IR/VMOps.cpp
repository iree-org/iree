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

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/STLExtras.h"
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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

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

void ModuleOp::build(OpBuilder &builder, OperationState &result,
                     StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
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

void FuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<MutableDictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
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
    if (auto argDict = argAttrs[i].getDictionary(builder.getContext())) {
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

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FuncOp functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, builder.getSymbolRefAttr(functionRef),
        exportName.empty() ? functionRef.getName() : exportName, attrs);
}

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FlatSymbolRefAttr functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("function_ref", functionRef);
  result.addAttribute("export_name", builder.getStringAttr(exportName));
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
  SmallVector<MutableDictionaryAttr, 8> argAttrs;
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
    MutableDictionaryAttr argAttrList;
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
    result->addAttribute(argName,
                         argAttrs[i].getDictionary(builder.getContext()));
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
    p << " -> (" << op.getType().getResults() << ")";
  }
  mlir::impl::printFunctionAttributes(p, op, op.getNumFuncArguments(),
                                      op.getNumFuncResults(),
                                      /*elided=*/
                                      {
                                          "is_variadic",
                                      });
}

void ImportOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<MutableDictionaryAttr> argAttrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
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
    if (auto argDict = argAttrs[i].getDictionary(builder.getContext())) {
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

  return parser.parseOptionalAttrDictWithKeyword(result->attributes);
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
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), /*elidedAttrs=*/{
                                         "sym_name",
                                         "is_mutable",
                                         "initializer",
                                         "initial_value",
                                         "type",
                                     });
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

void GlobalI32Op::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Optional<StringRef> initializer,
                        Optional<Attribute> initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder.getSymbolRefAttr(initializer.getValue()));
  } else if (initialValue.hasValue()) {
    result.addAttribute("initial_value", initialValue.getValue());
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void GlobalI32Op::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable,
                        IREE::VM::FuncOp initializer,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void GlobalI32Op::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Attribute initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void GlobalI32Op::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

void GlobalRefOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Optional<StringRef> initializer,
                        Optional<Attribute> initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (isMutable) {
    result.addAttribute("is_mutable", builder.getUnitAttr());
  }
  if (initializer.hasValue()) {
    result.addAttribute("initializer",
                        builder.getSymbolRefAttr(initializer.getValue()));
  }
  result.addAttribute("type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
}

void GlobalRefOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable,
                        IREE::VM::FuncOp initializer,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void GlobalRefOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        Attribute initialValue,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void GlobalRefOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, bool isMutable, Type type,
                        ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

static LogicalResult verifyGlobalAddressOp(GlobalAddressOp op) {
  auto *globalOp = op.getParentOfType<VM::ModuleOp>().lookupSymbol(op.global());
  if (!globalOp) {
    return op.emitOpError() << "Undefined global: " << op.global();
  }
  return success();
}

static LogicalResult verifyGlobalLoadOp(Operation *op) {
  auto globalAttr = op->getAttrOfType<FlatSymbolRefAttr>("global");
  auto *globalOp =
      op->getParentOfType<VM::ModuleOp>().lookupSymbol(globalAttr.getValue());
  if (!globalOp) {
    return op->emitOpError() << "Undefined global: " << globalAttr;
  }
  auto globalType = globalOp->getAttrOfType<TypeAttr>("type");
  auto loadType = op->getResult(0).getType();
  if (globalType.getValue() != loadType) {
    return op->emitOpError()
           << "Global type mismatch; global " << globalAttr << " is "
           << globalType << " but load is " << loadType;
  }
  return success();
}

static LogicalResult verifyGlobalStoreOp(Operation *op) {
  auto globalAttr = op->getAttrOfType<FlatSymbolRefAttr>("global");
  auto *globalOp =
      op->getParentOfType<VM::ModuleOp>().lookupSymbol(globalAttr.getValue());
  if (!globalOp) {
    return op->emitOpError() << "Undefined global: " << globalAttr;
  }
  auto globalType = globalOp->getAttrOfType<TypeAttr>("type");
  auto storeType = op->getOperand(0).getType();
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

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static ParseResult parseConstI32Op(OpAsmParser &parser,
                                   OperationState *result) {
  Attribute valueAttr;
  NamedAttrList dummyAttrs;
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
  if (failed(parser.parseOptionalAttrDict(result->attributes))) {
    return parser.emitError(parser.getCurrentLocation())
           << "Failed to parse optional attribute dict";
  }
  return parser.addTypeToList(valueAttr.getType(), result->types);
}

static void printConstI32Op(OpAsmPrinter &p, ConstI32Op &op) {
  p << op.getOperationName() << ' ';
  p.printAttribute(op.value());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
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
  return value.isa<UnitAttr>() || value.isa<IntegerAttr>() ||
         (value.isa<ElementsAttr>() && value.cast<ElementsAttr>()
                                           .getType()
                                           .getElementType()
                                           .isSignlessInteger());
}

// static
Attribute ConstI32Op::convertConstValue(Attribute value) {
  assert(isBuildableWith(value, value.getType()));
  Builder builder(value.getContext());
  int32_t dims = 1;
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

void ConstI32Op::build(OpBuilder &builder, OperationState &result,
                       Attribute value) {
  Attribute newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstI32Op::build(OpBuilder &builder, OperationState &result,
                       int32_t value) {
  return build(builder, result, builder.getI32IntegerAttr(value));
}

void ConstI32ZeroOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIntegerType(32));
}

void ConstRefZeroOp::build(OpBuilder &builder, OperationState &result,
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

void RodataOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     ElementsAttr value, ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("sym_name", builder.getStringAttr(name));
  result.addAttribute("value", value);
  result.addAttributes(attrs);
}

static LogicalResult verifyConstRefRodataOp(ConstRefRodataOp &op) {
  auto *rodataOp = op.getParentOfType<VM::ModuleOp>().lookupSymbol(op.rodata());
  if (!rodataOp) {
    return op.emitOpError() << "Undefined rodata section: " << op.rodata();
  }
  return success();
}

void ConstRefRodataOp::build(OpBuilder &builder, OperationState &result,
                             StringRef rodataName,
                             ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("rodata", builder.getSymbolRefAttr(rodataName));
  auto type = IREE::VM::RefType::get(ByteBufferType::get(builder.getContext()));
  result.addTypes({type});
  result.addAttributes(attrs);
}

void ConstRefRodataOp::build(OpBuilder &builder, OperationState &result,
                             RodataOp rodataOp,
                             ArrayRef<NamedAttribute> attrs) {
  build(builder, result, rodataOp.getName(), attrs);
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

static LogicalResult verifyListGetRefOp(ListGetRefOp &op) {
  auto listType = op.list()
                      .getType()
                      .cast<IREE::VM::RefType>()
                      .getObjectType()
                      .cast<IREE::VM::ListType>();
  auto elementType = listType.getElementType();
  auto resultType = op.result().getType();
  if (elementType.isa<IREE::VM::RefType>() !=
      resultType.isa<IREE::VM::RefType>()) {
    // Attempting to go between a primitive type and ref type.
    return op.emitError() << "cannot convert between list type " << elementType
                          << " and result type " << resultType;
  } else if (auto refType = elementType.dyn_cast<IREE::VM::RefType>()) {
    if (!refType.getObjectType().isa<IREE::VM::OpaqueType>() &&
        elementType != resultType) {
      // List has a concrete type, verify it matches.
      return op.emitError() << "list contains " << elementType
                            << " that cannot be accessed as " << resultType;
    }
  }
  return success();
}

static LogicalResult verifyListSetRefOp(ListSetRefOp &op) {
  auto listType = op.list()
                      .getType()
                      .cast<IREE::VM::RefType>()
                      .getObjectType()
                      .cast<IREE::VM::ListType>();
  auto elementType = listType.getElementType();
  auto valueType = op.value().getType();
  if (elementType.isa<IREE::VM::RefType>() !=
      valueType.isa<IREE::VM::RefType>()) {
    // Attempting to go between a primitive type and ref type.
    return op.emitError() << "cannot convert between list type " << elementType
                          << " and value type " << valueType;
  } else if (auto refType = elementType.dyn_cast<IREE::VM::RefType>()) {
    if (!refType.getObjectType().isa<IREE::VM::OpaqueType>() &&
        elementType != valueType) {
      // List has a concrete type, verify it matches.
      return op.emitError() << "list contains " << elementType
                            << " that cannot be mutated as " << valueType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Assignment
//===----------------------------------------------------------------------===//

static ParseResult parseSwitchOp(OpAsmParser &parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 4> values;
  OpAsmParser::OperandType index;
  OpAsmParser::OperandType defaultValue;
  Type type;
  if (failed(parser.parseOperand(index)) ||
      failed(parser.parseOperandList(values, OpAsmParser::Delimiter::Square)) ||
      failed(parser.parseKeyword("else")) ||
      failed(parser.parseOperand(defaultValue)) ||
      failed(parser.parseOptionalAttrDict(result->attributes)) ||
      failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperand(index,
                                   IntegerType::get(32, result->getContext()),
                                   result->operands)) ||
      failed(parser.resolveOperand(defaultValue, type, result->operands)) ||
      failed(parser.resolveOperands(values, type, result->operands)) ||
      failed(parser.addTypeToList(type, result->types))) {
    return failure();
  }
  return success();
}

template <typename T>
static void printSwitchOp(OpAsmPrinter &p, T &op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.index());
  p << "[";
  p.printOperands(op.values());
  p << "]";
  p << " else ";
  p.printOperand(op.default_value());
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.default_value().getType());
}

static ParseResult parseSwitchRefOp(OpAsmParser &parser,
                                    OperationState *result) {
  return parseSwitchOp(parser, result);
}

static void printSwitchRefOp(OpAsmPrinter &p, SwitchRefOp &op) {
  printSwitchOp(p, op);
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

Optional<MutableOperandRange> BranchOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

static ParseResult parseCallVariadicOp(OpAsmParser &parser,
                                       OperationState *result) {
  FlatSymbolRefAttr calleeAttr;
  auto calleeLoc = parser.getNameLoc();
  if (failed(parser.parseAttribute(calleeAttr, "callee", result->attributes)) ||
      failed(parser.parseLParen())) {
    return parser.emitError(calleeLoc) << "invalid callee symbol";
  }

  // Parsing here is a bit tricky as we want to be able to support things like
  // variadic lists of tuples while we don't know that the types are tuples yet.
  // We'll instead parse each segment as a flat list so `[(%a, %b), (%c, %d)]`
  // parses as `[%a, %b, %c, %d]` and then do the accounting below when parsing
  // types.
  SmallVector<OpAsmParser::OperandType, 4> flatOperands;
  SmallVector<int16_t, 4> flatSegmentSizes;
  while (failed(parser.parseOptionalRParen())) {
    if (succeeded(parser.parseOptionalLSquare())) {
      // Variadic list.
      SmallVector<OpAsmParser::OperandType, 4> flatSegmentOperands;
      while (failed(parser.parseOptionalRSquare())) {
        if (succeeded(parser.parseOptionalLParen())) {
          // List contains tuples, so track the () and parse inside of it.
          while (failed(parser.parseOptionalRParen())) {
            OpAsmParser::OperandType segmentOperand;
            if (failed(parser.parseOperand(segmentOperand))) {
              return parser.emitError(parser.getCurrentLocation())
                     << "invalid operand";
            }
            flatSegmentOperands.push_back(segmentOperand);
            if (failed(parser.parseOptionalComma())) {
              if (failed(parser.parseRParen())) {
                return parser.emitError(parser.getCurrentLocation())
                       << "malformed nested variadic tuple operand list";
              }
              break;
            }
          }
        } else {
          // Flat list of operands.
          OpAsmParser::OperandType segmentOperand;
          if (failed(parser.parseOperand(segmentOperand))) {
            return parser.emitError(parser.getCurrentLocation())
                   << "invalid operand";
          }
          flatSegmentOperands.push_back(segmentOperand);
        }
        if (failed(parser.parseOptionalComma())) {
          if (failed(parser.parseRSquare())) {
            return parser.emitError(parser.getCurrentLocation())
                   << "malformed variadic operand list";
          }
          break;
        }
      }
      flatSegmentSizes.push_back(flatSegmentOperands.size());
      flatOperands.append(flatSegmentOperands.begin(),
                          flatSegmentOperands.end());
    } else {
      // Normal single operand.
      OpAsmParser::OperandType operand;
      if (failed(parser.parseOperand(operand))) {
        return parser.emitError(parser.getCurrentLocation())
               << "malformed non-variadic operand";
      }
      flatSegmentSizes.push_back(-1);
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
      if (auto tupleType = operandType.dyn_cast<TupleType>()) {
        for (int i = 0; i < flatSegmentSizes[segmentIndex] / tupleType.size();
             ++i) {
          for (auto type : tupleType) {
            flatOperandTypes.push_back(type);
          }
        }
      } else {
        for (int i = 0; i < flatSegmentSizes[segmentIndex]; ++i) {
          flatOperandTypes.push_back(operandType);
        }
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
          VectorType::get({static_cast<int64_t>(flatSegmentSizes.size())},
                          parser.getBuilder().getIntegerType(16)),
          flatSegmentSizes));
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
  llvm::interleaveComma(
      llvm::zip(op.segment_sizes(), op.segment_types()), p,
      [&](std::tuple<APInt, Attribute> segmentSizeType) {
        int segmentSize = std::get<0>(segmentSizeType).getSExtValue();
        Type segmentType =
            std::get<1>(segmentSizeType).cast<TypeAttr>().getValue();
        if (segmentSize == -1) {
          p.printOperand(op.getOperand(operand++));
        } else {
          p << '[';
          if (auto tupleType = segmentType.dyn_cast<TupleType>()) {
            int tupleCount = segmentSize / tupleType.size();
            for (int i = 0; i < tupleCount; ++i) {
              p << '(';
              SmallVector<Value, 4> tupleOperands;
              for (int i = 0; i < tupleType.size(); ++i) {
                tupleOperands.push_back(op.getOperand(operand++));
              }
              p << tupleOperands;
              p << ')';
              if (i < tupleCount - 1) p << ", ";
            }
          } else {
            SmallVector<Value, 4> segmentOperands;
            for (int i = 0; i < segmentSize; ++i) {
              segmentOperands.push_back(op.getOperand(operand++));
            }
            p << segmentOperands;
          }
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
  llvm::interleaveComma(
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
    p << " -> " << op.getResult(0).getType();
  } else if (op.getNumResults() > 1) {
    p << " -> (" << op.getResultTypes() << ")";
  }
}

Optional<MutableOperandRange> CondBranchOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? trueDestOperandsMutable()
                            : falseDestOperandsMutable();
}

static LogicalResult verifyFailOp(FailOp op) {
  APInt status;
  if (matchPattern(op.status(), m_ConstantInt(&status))) {
    if (status == 0) {
      return op.emitOpError() << "status is 0; expected to not be OK";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Async/fiber ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

Block *BreakOp::getDest() { return getOperation()->getSuccessor(0); }

void BreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

Optional<MutableOperandRange> BreakOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
}

Block *CondBreakOp::getDest() { return getOperation()->getSuccessor(0); }

void CondBreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void CondBreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

Optional<MutableOperandRange> CondBreakOp::getMutableSuccessorOperands(
    unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOperandsMutable();
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
