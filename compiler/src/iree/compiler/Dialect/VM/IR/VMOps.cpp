// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

namespace {

template <typename NameTy>
void setResultName(OpAsmSetValueNameFn &setNameFn, Value result, NameTy name) {
  SmallString<32> osBuffer;
  llvm::raw_svector_ostream os(osBuffer);
  if (llvm::isa<VectorType>(result.getType())) {
    os << "v";
  }
  os << name;
  setNameFn(result, os.str());
}

void setResultIntegerName(OpAsmSetValueNameFn &setNameFn, Value result,
                          IntegerAttr value) {
  SmallString<32> osBuffer;
  llvm::raw_svector_ostream os(osBuffer);
  if (llvm::isa<VectorType>(result.getType())) {
    os << "v";
  }
  if (!value) {
    os << 'c';
  } else if (value.getValue() == 0) {
    os << "zero";
  } else {
    os << 'c' << value.getValue();
  }
  setNameFn(result, os.str());
}

} // namespace

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &result,
                     StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

LogicalResult ModuleOp::verify() {
  // TODO(benvanik): check export name conflicts.
  return success();
}

SmallVector<ModuleOp::Dependency> ModuleOp::getDependencies() {
  std::map<StringRef, Dependency> dependencies;
  for (auto importOp : getOps<IREE::VM::ImportOp>()) {
    auto [moduleName, importName] = importOp.getName().split(".");
    auto &dependency = dependencies[moduleName];
    dependency.name = moduleName;
    if (!importOp.getIsOptional()) {
      dependency.minimumVersion = std::max(
          dependency.minimumVersion, importOp.getMinimumVersion().value_or(0u));
      // Any required import in the module makes the entire module required.
      dependency.isOptional = false;
    }
  }
  return llvm::map_to_vector(dependencies, [](auto it) { return it.second; });
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void FuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute("function_type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty()) {
    return;
  }

  assert(type.getNumInputs() == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  function_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs,
      /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

Block *FuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  SmallVector<Location> locs(getFunctionType().getNumInputs(), getLoc());
  entry->addArguments(getFunctionType().getInputs(), locs);
  return entry;
}

LogicalResult FuncOp::verifyType() {
  auto type = getFunctionTypeAttr().getValue();
  if (!llvm::isa<FunctionType>(type))
    return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                       "' attribute of function type");
  return success();
}

void FuncOp::setReflectionAttr(StringRef name, Attribute value) {
  // TODO(benvanik): remove reflection attrs as a concept and use something more
  // MLIRish like an attribute interface/dialect interface.
  // DictionaryAttr is not very friendly for modification :/
  auto existingAttr =
      getOperation()->getAttrOfType<DictionaryAttr>("iree.reflection");
  SmallVector<NamedAttribute> attrs(existingAttr.begin(), existingAttr.end());
  bool didFind = false;
  for (size_t i = 0; i < attrs.size(); ++i) {
    if (attrs[i].getName() == name) {
      attrs[i].setValue(value);
      didFind = true;
      break;
    }
  }
  if (!didFind) {
    attrs.push_back(NamedAttribute(StringAttr::get(getContext(), name), value));
    DictionaryAttr::sortInPlace(attrs);
  }
  getOperation()->setAttr("iree.reflection",
                          DictionaryAttr::getWithSorted(getContext(), attrs));
}

ParseResult ExportOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr functionRefAttr;
  if (failed(parser.parseAttribute(functionRefAttr, "function_ref",
                                   result.attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("as"))) {
    StringAttr exportNameAttr;
    if (failed(parser.parseLParen()) ||
        failed(parser.parseAttribute(exportNameAttr, "export_name",
                                     result.attributes)) ||
        failed(parser.parseRParen())) {
      return failure();
    }
  } else {
    result.addAttribute("export_name", parser.getBuilder().getStringAttr(
                                           functionRefAttr.getValue()));
  }

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return failure();
  }

  return success();
}

void ExportOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << ' ';
  p.printSymbolName(getFunctionRef());
  if (getExportName() != getFunctionRef()) {
    p << " as(\"" << getExportName() << "\")";
  }
  p.printOptionalAttrDictWithKeyword(
      op->getAttrs(), /*elidedAttrs=*/{"function_ref", "export_name"});
}

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FuncOp functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, SymbolRefAttr::get(functionRef),
        exportName.empty() ? functionRef.getName() : exportName, attrs);
}

void ExportOp::build(OpBuilder &builder, OperationState &result,
                     FlatSymbolRefAttr functionRef, StringRef exportName,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("function_ref", functionRef);
  result.addAttribute("export_name", builder.getStringAttr(exportName));
  result.attributes.append(attrs.begin(), attrs.end());
}

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = getOperation();
  if (!symbolTable.lookupNearestSymbolFrom<IREE::VM::FuncOp>(
          op, getFunctionRefAttr())) {
    return op->emitError() << "vm.func op named '" << getFunctionRef()
                           << "' not found for export";
  }
  return success();
}

ParseResult ImportOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  StringAttr visibilityAttr;
  if (failed(parseSymbolVisibility(parser, visibilityAttr))) {
    return failure();
  }
  if (visibilityAttr) {
    result.addAttribute("sym_visibility", visibilityAttr);
  }
  if (succeeded(parser.parseOptionalKeyword("optional"))) {
    result.addAttribute("is_optional", builder.getUnitAttr());
  }
  StringAttr nameAttr;
  if (failed(parser.parseSymbolName(nameAttr,
                                    mlir::SymbolTable::getSymbolAttrName(),
                                    result.attributes)) ||
      failed(parser.parseLParen())) {
    return parser.emitError(parser.getNameLoc()) << "invalid import name";
  }
  SmallVector<DictionaryAttr, 8> argAttrs;
  SmallVector<Type, 8> argTypes;
  while (failed(parser.parseOptionalRParen())) {
    StringRef operandName;
    Type operandType;
    auto operandLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOptionalOperand(operand).has_value()) {
      if (failed(parser.parseColonType(operandType))) {
        return parser.emitError(operandLoc) << "invalid operand";
      }
      operandName = operand.name.substr(1); // consume `%`
    } else {
      if (failed(parser.parseType(operandType))) {
        return parser.emitError(operandLoc) << "invalid operand";
      }
    }
    argTypes.push_back(operandType);
    NamedAttrList argAttrList;
    if (!operandName.empty()) {
      argAttrList.set("vm.name", builder.getStringAttr(operandName));
    }
    if (succeeded(parser.parseOptionalEllipsis())) {
      argAttrList.set("vm.variadic", builder.getUnitAttr());
    }
    argAttrs.push_back(argAttrList.getDictionary(result.getContext()));
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
  function_interface_impl::addArgAndResultAttrs(
      builder, result, argAttrs,
      /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return failure();
  }

  auto functionType =
      FunctionType::get(result.getContext(), argTypes, resultTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(functionType));

  result.addRegion();

  return success();
}

void ImportOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << ' ';
  printSymbolVisibility(p, op, getSymVisibilityAttr());
  p << ' ';
  if (getIsOptional()) {
    p << "optional ";
  }
  p.printSymbolName(getName());
  p << "(";
  for (int i = 0; i < getArgumentTypes().size(); ++i) {
    if (auto name = getArgAttrOfType<StringAttr>(i, "vm.name")) {
      p << '%' << name.getValue() << " : ";
    }
    p.printType(getFunctionType().getInput(i));
    if (getArgAttrOfType<UnitAttr>(i, "vm.variadic")) {
      p << " ...";
    }
    if (i < getArgumentTypes().size() - 1) {
      p << ", ";
    }
  }
  p << ")";
  if (getResultTypes().size() == 1) {
    p << " -> ";
    p.printType(getFunctionType().getResult(0));
  } else if (getResultTypes().size() > 1) {
    p << " -> (" << getFunctionType().getResults() << ")";
  }
  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {
          getFunctionTypeAttrName(),
          getArgAttrsAttrName(),
          getResAttrsAttrName(),
          "is_variadic",
          getIsOptionalAttrName(),
          "sym_visibility",
      });
}

void ImportOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute("function_type", TypeAttr::get(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (!argAttrs.empty()) {
    assert(type.getNumInputs() == argAttrs.size() &&
           "expected as many argument attribute lists as arguments");
    function_interface_impl::addArgAndResultAttrs(
        builder, result, argAttrs,
        /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(result.name),
        getResAttrsAttrName(result.name));
  }

  result.addRegion();
}

LogicalResult ImportOp::verifyType() {
  auto type = getFunctionTypeAttr().getValue();
  if (!llvm::isa<FunctionType>(type))
    return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                       "' attribute of function type");
  return success();
}

void InitializerOp::build(OpBuilder &builder, OperationState &result,
                          ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("function_type", TypeAttr::get(FunctionType::get(
                                           builder.getContext(), {}, {})));
  result.addRegion();
  result.attributes.append(attrs.begin(), attrs.end());
}

ParseResult InitializerOp::parse(OpAsmParser &parser, OperationState &result) {
  result.addAttribute("function_type", TypeAttr::get(FunctionType::get(
                                           result.getContext(), {}, {})));
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }
  auto &body = *result.addRegion();
  if (failed(parser.parseRegion(body))) {
    return failure();
  }
  return success();
}

void InitializerOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p.printOptionalAttrDictWithKeyword(op->getAttrs(),
                                     /*elidedAttrs=*/{"function_type"});
  p << " ";
  p.printRegion(getBody());
}

Block *InitializerOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  return entry;
}

Block *InitializerOp::addBlock() {
  assert(!empty() && "function should at least have an entry block");
  push_back(new Block());
  return &back();
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

template <typename T>
static void addMemoryEffectsForGlobal(
    Operation *op, mlir::FlatSymbolRefAttr global,
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // HACK: works around the lack of symbol side effects in mlir by only saying
  // we have a side-effect if the variable we are loading is mutable.
  auto *symbolOp = SymbolTable::lookupNearestSymbolFrom(op, global);
  assert(symbolOp);
  auto globalOp = dyn_cast<T>(symbolOp);
  if (globalOp.getIsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

void GlobalLoadI32Op::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemoryEffectsForGlobal<GlobalI32Op>(*this, getGlobalAttr(), effects);
}

void GlobalLoadI32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getGlobal());
}

void GlobalLoadI64Op::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemoryEffectsForGlobal<GlobalI64Op>(*this, getGlobalAttr(), effects);
}

void GlobalLoadI64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getGlobal());
}

void GlobalLoadF32Op::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemoryEffectsForGlobal<GlobalF32Op>(*this, getGlobalAttr(), effects);
}

void GlobalLoadF32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getGlobal());
}

void GlobalLoadF64Op::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemoryEffectsForGlobal<GlobalF64Op>(*this, getGlobalAttr(), effects);
}

void GlobalLoadF64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getGlobal());
}

void GlobalLoadRefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  addMemoryEffectsForGlobal<GlobalRefOp>(*this, getGlobalAttr(), effects);
}

void GlobalLoadRefOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getGlobal());
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

template <int SZ>
static bool isConstIntegerBuildableWith(TypedAttr value, Type type) {
  // FlatSymbolRefAttr can only be used with a function type.
  if (llvm::isa<FlatSymbolRefAttr>(value)) {
    return false;
  }
  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type) {
    return false;
  }
  if (llvm::isa<UnitAttr>(value)) {
    return SZ == 32; // Conditions/bools are always i32
  } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
    return intAttr.getType().isInteger(SZ);
  } else if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(value)) {
    return elementsAttr.getShapedType().getElementType().isInteger(SZ);
  }
  return false;
}

template <int SZ>
static bool isConstFloatBuildableWith(TypedAttr value, Type type) {
  // FlatSymbolRefAttr can only be used with a function type.
  if (llvm::isa<FlatSymbolRefAttr>(value)) {
    return false;
  }
  // Otherwise, the attribute must have the same type as 'type'.
  if (value.getType() != type) {
    return false;
  }
  Type elementType;
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(value)) {
    elementType = floatAttr.getType();
  } else if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(value)) {
    elementType = elementsAttr.getShapedType().getElementType();
  }
  if (!elementType)
    return false;
  return elementType.getIntOrFloatBitWidth() == SZ;
}

template <int SZ>
static TypedAttr convertConstIntegerValue(TypedAttr value) {
  assert(isConstIntegerBuildableWith<SZ>(value, value.getType()));
  Builder builder(value.getContext());
  auto integerType = builder.getIntegerType(SZ);
  int32_t dims = 1;
  if (llvm::isa<UnitAttr>(value)) {
    return IntegerAttr::get(integerType, APInt(SZ, 1));
  } else if (auto v = llvm::dyn_cast<BoolAttr>(value)) {
    return IntegerAttr::get(integerType,
                            APInt(SZ, v.getValue() ? 1 : 0, false));
  } else if (auto v = llvm::dyn_cast<IntegerAttr>(value)) {
    return IntegerAttr::get(integerType,
                            APInt(SZ, v.getValue().getLimitedValue()));
  } else if (auto v = llvm::dyn_cast<ElementsAttr>(value)) {
    dims = v.getNumElements();
    ShapedType adjustedType = VectorType::get({dims}, integerType);
    if (auto elements = llvm::dyn_cast<SplatElementsAttr>(v)) {
      return SplatElementsAttr::get(adjustedType,
                                    elements.getSplatValue<Attribute>());
    } else {
      return DenseElementsAttr::get(adjustedType,
                                    llvm::to_vector(v.getValues<Attribute>()));
    }
  }
  assert(false && "unexpected attribute type");
  return TypedAttr();
}

static FloatType getFloatType(int bitwidth, MLIRContext *context) {
  switch (bitwidth) {
  case 16:
    return FloatType::getF16(context);
  case 32:
    return FloatType::getF32(context);
  case 64:
    return FloatType::getF64(context);
  default:
    assert(false && "unhandled floating point type");
    return {};
  }
}

template <int SZ>
static TypedAttr convertConstFloatValue(TypedAttr value) {
  assert(isConstFloatBuildableWith<SZ>(value, value.getType()));
  Builder builder(value.getContext());
  auto floatType = getFloatType(SZ, value.getContext());
  int32_t dims = 1;
  if (auto v = llvm::dyn_cast<FloatAttr>(value)) {
    return FloatAttr::get(floatType, v.getValue());
  } else if (auto v = llvm::dyn_cast<ElementsAttr>(value)) {
    dims = v.getNumElements();
    ShapedType adjustedType = VectorType::get({dims}, floatType);
    if (auto elements = llvm::dyn_cast<SplatElementsAttr>(v)) {
      return SplatElementsAttr::get(adjustedType,
                                    elements.getSplatValue<Attribute>());
    } else {
      return DenseElementsAttr::get(adjustedType,
                                    llvm::to_vector(v.getValues<Attribute>()));
    }
  }
  assert(false && "unexpected attribute type");
  return TypedAttr();
}

// static
bool ConstI32Op::isBuildableWith(TypedAttr value, Type type) {
  return isConstIntegerBuildableWith<32>(value, type);
}

// static
TypedAttr ConstI32Op::convertConstValue(TypedAttr value) {
  return convertConstIntegerValue<32>(value);
}

void ConstI32Op::build(OpBuilder &builder, OperationState &result,
                       TypedAttr value) {
  TypedAttr newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstI32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultIntegerName(setNameFn, getResult(),
                       llvm::dyn_cast<IntegerAttr>(getValue()));
}

void ConstI32Op::build(OpBuilder &builder, OperationState &result,
                       int32_t value) {
  return build(builder, result, builder.getI32IntegerAttr(value));
}

// static
bool ConstI64Op::isBuildableWith(TypedAttr value, Type type) {
  return isConstIntegerBuildableWith<64>(value, type);
}

// static
TypedAttr ConstI64Op::convertConstValue(TypedAttr value) {
  return convertConstIntegerValue<64>(value);
}

void ConstI64Op::build(OpBuilder &builder, OperationState &result,
                       TypedAttr value) {
  TypedAttr newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstI64Op::build(OpBuilder &builder, OperationState &result,
                       int64_t value) {
  return build(builder, result, builder.getI64IntegerAttr(value));
}

void ConstI64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultIntegerName(setNameFn, getResult(),
                       llvm::dyn_cast<IntegerAttr>(getValue()));
}

// static
bool ConstF32Op::isBuildableWith(TypedAttr value, Type type) {
  return isConstFloatBuildableWith<32>(value, type);
}

// static
TypedAttr ConstF32Op::convertConstValue(TypedAttr value) {
  return convertConstFloatValue<32>(value);
}

void ConstF32Op::build(OpBuilder &builder, OperationState &result,
                       TypedAttr value) {
  TypedAttr newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstF32Op::build(OpBuilder &builder, OperationState &result,
                       float value) {
  return build(builder, result, builder.getF32FloatAttr(value));
}

// static
bool ConstF64Op::isBuildableWith(TypedAttr value, Type type) {
  return isConstFloatBuildableWith<64>(value, type);
}

// static
TypedAttr ConstF64Op::convertConstValue(TypedAttr value) {
  return convertConstFloatValue<64>(value);
}

void ConstF64Op::build(OpBuilder &builder, OperationState &result,
                       TypedAttr value) {
  TypedAttr newValue = convertConstValue(value);
  result.addAttribute("value", newValue);
  result.addTypes(newValue.getType());
}

void ConstF64Op::build(OpBuilder &builder, OperationState &result,
                       double value) {
  return build(builder, result, builder.getF64FloatAttr(value));
}

void ConstI32ZeroOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIntegerType(32));
}

void ConstI32ZeroOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "zero");
}

void ConstI64ZeroOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIntegerType(64));
}

void ConstI64ZeroOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "zero");
}

void ConstF32ZeroOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getF32Type());
}

void ConstF32ZeroOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "zero");
}

void ConstF64ZeroOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getF64Type());
}

void ConstF64ZeroOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "zero");
}

void ConstRefZeroOp::build(OpBuilder &builder, OperationState &result,
                           Type objectType) {
  result.addTypes(objectType);
}

void ConstRefZeroOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "null");
}

void RodataOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     Attribute value, ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("sym_name", builder.getStringAttr(name));
  result.addAttribute("value", value);
  result.addAttributes(attrs);
}

LogicalResult ConstRefRodataOp::verify() {
  Operation *op = getOperation();
  auto *rodataOp =
      op->getParentOfType<VM::ModuleOp>().lookupSymbol(getRodata());
  if (!rodataOp) {
    return op->emitOpError() << "Undefined rodata section: " << getRodata();
  }
  return success();
}

void ConstRefRodataOp::build(OpBuilder &builder, OperationState &result,
                             StringRef rodataName,
                             ArrayRef<NamedAttribute> attrs) {
  result.addAttribute("rodata",
                      SymbolRefAttr::get(builder.getContext(), rodataName));
  auto type =
      IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext()));
  result.addTypes({type});
  result.addAttributes(attrs);
}

void ConstRefRodataOp::build(OpBuilder &builder, OperationState &result,
                             RodataOp rodataOp,
                             ArrayRef<NamedAttribute> attrs) {
  build(builder, result, rodataOp.getName(), attrs);
}

void ConstRefRodataOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), getRodata());
}

// Makes a human-readable symbol name for the given string value.
// This is not uniqued and may need uniquing before being added to the symbol
// table.
//
// For example:
//   'Some string!' -> '_utf8_some_string'
//   'I'm a really long'... -> '_utf8_im_a_really_long'
static std::string makeSafeIdentifier(StringRef unsafeIdentifier) {
  std::string result = "_utf8_";
  llvm::raw_string_ostream os(result);
  bool lastUnderscore = true;
  for (char c : unsafeIdentifier) {
    if (!llvm::isPrint(c))
      continue;
    if (llvm::isAlnum(c)) {
      os << llvm::toLower(c);
      lastUnderscore = false;
    } else if (!lastUnderscore) {
      os << "_";
      lastUnderscore = true;
    }
  }
  std::string prefix = os.str().substr(0, 32);
  if (!StringRef(prefix).endswith("_")) {
    prefix += "_";
  }
  return prefix + llvm::utohexstr(static_cast<uint64_t>(
                      llvm::hash_value(unsafeIdentifier)));
}

void RodataInlineOp::build(OpBuilder &builder, OperationState &result,
                           StringAttr value) {
  // Make an identifier-friendly version of the string so that the value is
  // more readable in IR (so "I'm some string" becomes "im_some_string", etc).
  auto safeIdentifier = makeSafeIdentifier(value.getValue());
  build(builder, result,
        IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext())),
        /*name=*/builder.getStringAttr(safeIdentifier), /*data=*/value,
        /*alignment=*/builder.getI64IntegerAttr(1),
        /*mimeType=*/nullptr);
}

void RodataTableOp::build(OpBuilder &builder, OperationState &result,
                          StringAttr name, IREE::Util::CompositeAttr value) {
  // Make an identifier-friendly version of the string so that the value is
  // more readable in IR (so "I'm some string" becomes "im_some_string", etc).
  auto safeIdentifier = makeSafeIdentifier(name.getValue());
  // Make names for the table and data based on the safe identifier.
  std::string tableName = safeIdentifier + "_table";
  std::string dataName = safeIdentifier + "_data";
  auto refType =
      IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext()));
  build(builder, result, TypeRange{refType, refType},
        /*tableName=*/builder.getStringAttr(tableName),
        /*dataName=*/builder.getStringAttr(dataName), /*dataArray=*/value,
        /*alignment=*/nullptr, /*dataAlignment=*/nullptr, /*mimeType=*/nullptr);
}

void RodataTableOp::build(OpBuilder &builder, OperationState &result,
                          IREE::Util::CompositeAttr value) {
  auto refType =
      IREE::VM::RefType::get(IREE::VM::BufferType::get(builder.getContext()));
  build(builder, result, TypeRange{refType, refType},
        /*tableName=*/nullptr, /*dataName=*/nullptr,
        /*dataArray=*/value, /*alignment=*/nullptr, /*dataAlignment=*/nullptr,
        /*mimeType=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

LogicalResult ListGetRefOp::verify() {
  Operation *op = getOperation();
  auto listType = llvm::cast<IREE::VM::ListType>(
      getList().getType().cast<IREE::VM::RefType>().getObjectType());
  auto elementType = listType.getElementType();
  auto resultType = getResult().getType();
  if (!llvm::isa<IREE::VM::OpaqueType>(elementType)) {
    if (llvm::isa<IREE::VM::RefType>(elementType) !=
        llvm::isa<IREE::VM::RefType>(resultType)) {
      // Attempting to go between a primitive type and ref type.
      return op->emitError()
             << "cannot convert between list type " << elementType
             << " and result type " << resultType;
    } else if (auto refType = llvm::dyn_cast<IREE::VM::RefType>(elementType)) {
      if (!llvm::isa<IREE::VM::OpaqueType>(refType.getObjectType()) &&
          elementType != resultType) {
        // List has a concrete type, verify it matches.
        return op->emitError() << "list contains " << elementType
                               << " that cannot be accessed as " << resultType;
      }
    }
  }
  return success();
}

LogicalResult ListSetRefOp::verify() {
  Operation *op = getOperation();
  auto listType = llvm::cast<IREE::VM::ListType>(
      getList().getType().cast<IREE::VM::RefType>().getObjectType());
  auto elementType = listType.getElementType();
  auto valueType = getValue().getType();
  if (!llvm::isa<IREE::VM::OpaqueType>(elementType)) {
    if (llvm::isa<IREE::VM::RefType>(elementType) !=
        llvm::isa<IREE::VM::RefType>(valueType)) {
      // Attempting to go between a primitive type and ref type.
      return op->emitError() << "cannot convert between list type "
                             << elementType << " and value type " << valueType;
    } else if (auto refType = llvm::dyn_cast<IREE::VM::RefType>(elementType)) {
      if (!llvm::isa<IREE::VM::OpaqueType>(refType.getObjectType()) &&
          elementType != valueType) {
        // List has a concrete type, verify it matches.
        return op->emitError() << "list contains " << elementType
                               << " that cannot be mutated as " << valueType;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Assignment
//===----------------------------------------------------------------------===//

static ParseResult parseSwitchOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> values;
  OpAsmParser::UnresolvedOperand index;
  OpAsmParser::UnresolvedOperand defaultValue;
  Type type;
  if (failed(parser.parseOperand(index)) ||
      failed(parser.parseOperandList(values, OpAsmParser::Delimiter::Square)) ||
      failed(parser.parseKeyword("else")) ||
      failed(parser.parseOperand(defaultValue)) ||
      failed(parser.parseOptionalAttrDict(result.attributes)) ||
      failed(parser.parseColonType(type)) ||
      failed(parser.resolveOperand(
          index, IntegerType::get(result.getContext(), 32), result.operands)) ||
      failed(parser.resolveOperand(defaultValue, type, result.operands)) ||
      failed(parser.resolveOperands(values, type, result.operands)) ||
      failed(parser.addTypeToList(type, result.types))) {
    return failure();
  }
  return success();
}

template <typename T>
static void printSwitchOp(OpAsmPrinter &p, T &op) {
  p << " ";
  p.printOperand(op.getIndex());
  p << "[";
  p.printOperands(op.getValues());
  p << "]";
  p << " else ";
  p.printOperand(op.getDefaultValue());
  p.printOptionalAttrDict(op->getAttrs());
  p << " : ";
  p.printType(op.getDefaultValue().getType());
}

ParseResult SwitchRefOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSwitchOp(parser, result);
}

void SwitchRefOp::print(OpAsmPrinter &p) { printSwitchOp(p, *this); }

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

void CmpEQI32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "eq");
}

void CmpEQI64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "eq");
}

void CmpNEI32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ne");
}

void CmpNEI64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ne");
}

void CmpNZI64Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "nz");
}

void CmpLTI32SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "slt");
}

void CmpLTI64SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "slt");
}

void CmpLTEI32SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "slte");
}

void CmpLTEI64SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "slte");
}

void CmpLTI32UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ult");
}

void CmpLTI64UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ult");
}

void CmpLTEI32UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ulte");
}

void CmpLTEI64UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ulte");
}

void CmpGTI32SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "sgt");
}

void CmpGTI64SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "sgt");
}

void CmpGTEI32SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "sgte");
}

void CmpGTEI64SOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "sgte");
}

void CmpGTI32UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ugt");
}

void CmpGTI64UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ugt");
}

void CmpGTEI32UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ugte");
}

void CmpGTEI64UOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "ugte");
}

void CmpNZI32Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "nz");
}

void CmpEQRefOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "req");
}

void CmpNERefOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "rne");
}

void CmpNZRefOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setResultName(setNameFn, getResult(), "rnz");
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

SuccessorOperands BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

void CallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!getOperation()->hasAttr("nosideeffects")) {
    // TODO(benvanik): actually annotate this.
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

void CallVariadicOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!getOperation()->hasAttr("nosideeffects")) {
    // TODO(benvanik): actually annotate this.
    effects.emplace_back(MemoryEffects::Read::get());
    effects.emplace_back(MemoryEffects::Write::get());
  }
}

ParseResult CallVariadicOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr calleeAttr;
  auto calleeLoc = parser.getNameLoc();
  if (failed(parser.parseAttribute(calleeAttr, "callee", result.attributes)) ||
      failed(parser.parseLParen())) {
    return parser.emitError(calleeLoc) << "invalid callee symbol";
  }

  // Parsing here is a bit tricky as we want to be able to support things like
  // variadic lists of tuples while we don't know that the types are tuples yet.
  // We'll instead parse each segment as a flat list so `[(%a, %b), (%c, %d)]`
  // parses as `[%a, %b, %c, %d]` and then do the accounting below when parsing
  // types.
  SmallVector<OpAsmParser::UnresolvedOperand> flatOperands;
  SmallVector<int16_t> flatSegmentSizes;
  while (failed(parser.parseOptionalRParen())) {
    if (succeeded(parser.parseOptionalLSquare())) {
      // Variadic list.
      SmallVector<OpAsmParser::UnresolvedOperand> flatSegmentOperands;
      while (failed(parser.parseOptionalRSquare())) {
        if (succeeded(parser.parseOptionalLParen())) {
          // List contains tuples, so track the () and parse inside of it.
          while (failed(parser.parseOptionalRParen())) {
            OpAsmParser::UnresolvedOperand segmentOperand;
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
          OpAsmParser::UnresolvedOperand segmentOperand;
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
      OpAsmParser::UnresolvedOperand operand;
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

  if (failed(parser.parseOptionalAttrDict(result.attributes)) ||
      failed(parser.parseColon()) || failed(parser.parseLParen())) {
    return parser.emitError(parser.getCurrentLocation())
           << "malformed optional attributes list";
  }
  SmallVector<Type> flatOperandTypes;
  SmallVector<Type> segmentTypes;
  SmallVector<int16_t> segmentSizes;
  int segmentIndex = 0;
  while (failed(parser.parseOptionalRParen())) {
    Type operandType;
    if (failed(parser.parseType(operandType))) {
      return parser.emitError(parser.getCurrentLocation())
             << "invalid operand type";
    }
    bool isVariadic = succeeded(parser.parseOptionalEllipsis());
    if (isVariadic) {
      int flatSegmentSize = flatSegmentSizes[segmentIndex];
      if (auto tupleType = llvm::dyn_cast<TupleType>(operandType)) {
        for (int i = 0; i < flatSegmentSize / tupleType.size(); ++i) {
          for (auto type : tupleType) {
            flatOperandTypes.push_back(type);
          }
        }
        segmentSizes.push_back(flatSegmentSize / tupleType.size());
      } else {
        for (int i = 0; i < flatSegmentSize; ++i) {
          flatOperandTypes.push_back(operandType);
        }
        segmentSizes.push_back(flatSegmentSize);
      }
    } else {
      flatOperandTypes.push_back(operandType);
      segmentSizes.push_back(-1);
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
                                    result.operands))) {
    return parser.emitError(parser.getCurrentLocation())
           << "operands do not match type list";
  }
  result.addAttribute(
      "segment_sizes",
      DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                          parser.getBuilder().getIntegerType(16)),
          segmentSizes));
  result.addAttribute("segment_types",
                      parser.getBuilder().getArrayAttr(
                          llvm::map_to_vector(segmentTypes, [&](Type type) {
                            return llvm::cast<Attribute>(TypeAttr::get(type));
                          })));

  if (failed(parser.parseOptionalArrowTypeList(result.types))) {
    return parser.emitError(parser.getCurrentLocation())
           << "malformed function type results";
  }

  return success();
}

void CallVariadicOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << ' ' << op->getAttr("callee") << '(';
  int operand = 0;
  llvm::interleaveComma(
      llvm::zip_equal(getSegmentSizes(), getSegmentTypes()), p,
      [&](std::tuple<APInt, Attribute> segmentSizeType) {
        int segmentSize = std::get<0>(segmentSizeType).getSExtValue();
        Type segmentType =
            llvm::cast<TypeAttr>(std::get<1>(segmentSizeType)).getValue();
        if (segmentSize == -1) {
          p.printOperand(getOperand(operand++));
        } else {
          p << '[';
          if (auto tupleType = llvm::dyn_cast<TupleType>(segmentType)) {
            for (size_t i = 0; i < segmentSize; ++i) {
              p << '(';
              SmallVector<Value> tupleOperands;
              for (size_t j = 0; j < tupleType.size(); ++j) {
                tupleOperands.push_back(getOperand(operand++));
              }
              p << tupleOperands;
              p << ')';
              if (i < segmentSize - 1)
                p << ", ";
            }
          } else {
            SmallVector<Value> segmentOperands;
            for (int i = 0; i < segmentSize; ++i) {
              segmentOperands.push_back(getOperand(operand++));
            }
            p << segmentOperands;
          }
          p << ']';
        }
      });
  p << ')';
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{
                              "callee",
                              "segment_sizes",
                              "segment_types",
                          });
  p << " : (";
  llvm::interleaveComma(
      llvm::zip_equal(getSegmentSizes(), getSegmentTypes()), p,
      [&](std::tuple<APInt, Attribute> segmentSizeType) {
        int segmentSize = std::get<0>(segmentSizeType).getSExtValue();
        Type segmentType =
            llvm::cast<TypeAttr>(std::get<1>(segmentSizeType)).getValue();
        if (segmentSize == -1) {
          p.printType(segmentType);
        } else {
          p.printType(segmentType);
          p << " ...";
        }
      });
  p << ")";
  if (getNumResults() == 1) {
    p << " -> " << getResult(0).getType();
  } else if (getNumResults() > 1) {
    p << " -> (" << getResultTypes() << ")";
  }
}

SuccessorOperands CondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? SuccessorOperands(getTrueDestOperandsMutable())
                            : SuccessorOperands(getFalseDestOperandsMutable());
}

static ParseResult parseBranchTableCases(
    OpAsmParser &parser, Block *&defaultDestination,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &defaultOperands,
    SmallVectorImpl<Type> &defaultOperandTypes,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  if (parser.parseKeyword("default") || parser.parseColon() ||
      parser.parseSuccessor(defaultDestination))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseOperandList(defaultOperands, OpAsmParser::Delimiter::None,
                                /*allowResultNumber=*/false) ||
        parser.parseColonTypeList(defaultOperandTypes) || parser.parseRParen())
      return failure();
  }
  while (succeeded(parser.parseOptionalComma())) {
    int64_t index = 0;
    if (failed(parser.parseInteger(index)))
      return failure();
    if (index != caseDestinations.size())
      return failure();
    Block *destination;
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    SmallVector<Type> operandTypes;
    if (failed(parser.parseColon()) ||
        failed(parser.parseSuccessor(destination)))
      return failure();
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseOperandList(operands, OpAsmParser::Delimiter::None,
                                         /*allowResultNumber=*/false)) ||
          failed(parser.parseColonTypeList(operandTypes)) ||
          failed(parser.parseRParen()))
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
  }
  return success();
}

static void printBranchTableCases(OpAsmPrinter &p, Operation *op,
                                  Block *defaultDestination,
                                  OperandRange defaultOperands,
                                  TypeRange defaultOperandTypes,
                                  SuccessorRange caseDestinations,
                                  OperandRangeRange caseOperands,
                                  const TypeRangeRange &caseOperandTypes) {
  p.increaseIndent();
  p << "  default: ";
  p.printSuccessorAndUseList(defaultDestination, defaultOperands);
  int index = 0;
  for (auto [caseDestination, caseOperands, caseOperandTypes] :
       llvm::zip_equal(caseDestinations, caseOperands, caseOperandTypes)) {
    p << ',';
    p.printNewline();
    p << (index++) << ": ";
    p.printSuccessorAndUseList(caseDestination, caseOperands);
  }
  p.decreaseIndent();
  p.printNewline();
}

SuccessorOperands BranchTableOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDefaultOperandsMutable()
                                      : getCaseOperandsMutable(index - 1));
}

Block *BranchTableOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  SuccessorRange caseDestinations = getCaseDestinations();
  if (auto valueAttr = llvm::dyn_cast_or_null<IntegerAttr>(operands.front())) {
    int64_t value = valueAttr.getValue().getSExtValue();
    if (value < 0 || value >= caseDestinations.size())
      return getDefaultDestination();
    return caseDestinations[value];
  }
  return nullptr;
}

LogicalResult verifyFailOp(Operation *op, Value statusVal) {
  APInt status;
  if (matchPattern(statusVal, m_ConstantInt(&status))) {
    if (status == 0) {
      return op->emitOpError() << "status is 0; expected to not be OK";
    }
  }
  return success();
}

ParseResult CondFailOp::parse(OpAsmParser &parser, OperationState &result) {
  // First operand is either 'condition' or 'status', both i32.
  OpAsmParser::UnresolvedOperand condition;
  if (failed(parser.parseOperand(condition))) {
    return failure();
  }

  // First try looking for an operand after a comma. If no operand, keep track
  // of the already parsed comma to avoid checking for a comma later on.
  bool trailingComma = false;
  OpAsmParser::UnresolvedOperand status = condition;
  if (succeeded(parser.parseOptionalComma()) &&
      !parser.parseOptionalOperand(status).has_value()) {
    trailingComma = true;
  }

  StringAttr messageAttr;
  if ((trailingComma || succeeded(parser.parseOptionalComma())) &&
      failed(
          parser.parseAttribute(messageAttr, "message", result.attributes))) {
    return failure();
  }

  Type operandType = IntegerType::get(result.getContext(), 32);
  if (failed(parser.resolveOperand(condition, operandType, result.operands)) ||
      failed(parser.resolveOperand(status, operandType, result.operands))) {
    return failure();
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

void CondFailOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getCondition() != getStatus()) {
    p << getCondition() << ", ";
  }
  p << getStatus();
  if (auto messageAttr = getMessage()) {
    p << ", \"" << messageAttr.value() << "\"";
  }
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"message"});
}

void ImportResolvedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  std::string name = sanitizeSymbolName(("has_" + getImport()).str());
  setResultName(setNameFn, getResult(), name);
}

//===----------------------------------------------------------------------===//
// Async/fiber ops
//===----------------------------------------------------------------------===//

void YieldOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void YieldOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

SuccessorOperands YieldOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

void BreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

SuccessorOperands BreakOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

void CondBreakOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void CondBreakOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

SuccessorOperands CondBreakOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

} // namespace mlir::iree_compiler::IREE::VM

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/VM/IR/VMOpEncoder.cpp.inc" // IWYU pragma: keep
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/VM/IR/VMOps.cpp.inc" // IWYU pragma: keep
