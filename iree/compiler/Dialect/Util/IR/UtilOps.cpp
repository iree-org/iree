// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
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
namespace Util {

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

static ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                         StringAttr &symVisibilityAttr) {
  StringRef symVisibility;
  parser.parseOptionalKeyword(&symVisibility, {"public", "private", "nested"});
  if (!symVisibility.empty()) {
    symVisibilityAttr = parser.getBuilder().getStringAttr(symVisibility);
  }
  return success();
}

static void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                  StringAttr symVisibilityAttr) {
  if (!symVisibilityAttr) {
    p << "public";
  } else {
    p << symVisibilityAttr.getValue();
  }
}

//===----------------------------------------------------------------------===//
// custom<TypeOrAttr>($type, $attr)
//===----------------------------------------------------------------------===//
// some.op custom<TypeOrAttr>($type, $attr)
// ->
// some.op : i32
// some.op = 42 : i32

static ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                                   Attribute &attr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(attr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
    typeAttr = TypeAttr::get(attr.getType());
  } else {
    Type type;
    if (failed(parser.parseColonType(type))) {
      return parser.emitError(parser.getCurrentLocation()) << "expected type";
    }
    typeAttr = TypeAttr::get(type);
  }
  return success();
}

static void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                            Attribute attr) {
  if (attr) {
    p << " = ";
    p.printAttribute(attr);
  } else {
    p << " : ";
    p.printAttribute(type);
  }
}

//===----------------------------------------------------------------------===//
// util.do_not_optimize
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
  p << "util.do_not_optimize";
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
// util.unfoldable_constant
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
  auto constOp = cast<IREE::Util::UnfoldableConstantOp>(op);
  p << "util.unfoldable_constant ";
  p.printOptionalAttrDict(constOp->getAttrs(), /*elidedAttrs=*/{"value"});

  if (constOp->getAttrs().size() > 1) p << ' ';
  p << constOp.value();

  // If the value is a symbol reference, print a trailing type.
  if (constOp.value().isa<SymbolRefAttr>()) p << " : " << constOp.getType();
}

namespace {

struct ExpandUnfoldableConstantOp
    : public OpRewritePattern<UnfoldableConstantOp> {
  using OpRewritePattern<IREE::Util::UnfoldableConstantOp>::OpRewritePattern;
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
// Globals
//===----------------------------------------------------------------------===//

// Returns true if the given |accessType| is compatible with the |globalType|.
// For example, this will return true if the global type is a tensor<?xf32>
// and the access is tensor<4xf32>.
static bool isGlobalTypeCompatible(Type globalType, Type accessType) {
  // If one is a shaped type, then they both must be and have compatible
  // shapes.
  if (globalType.isa<ShapedType>() || accessType.isa<ShapedType>()) {
    return succeeded(mlir::verifyCompatibleShape(globalType, accessType));
  }

  // TODO(benvanik): use GlobalOpInterface.

  // Otherwise, the types must be the same.
  return globalType == accessType;
}

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, Type type, Optional<StringRef> initializer,
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

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, mlir::FuncOp initializer,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, initializer.getType().getResult(0),
        initializer.getName(), llvm::None, attrs);
}

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, Type type, Attribute initialValue,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, initialValue,
        attrs);
}

void GlobalOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     bool isMutable, Type type,
                     ArrayRef<NamedAttribute> attrs) {
  build(builder, result, name, isMutable, type, llvm::None, llvm::None, attrs);
}

static LogicalResult verifyGlobalOp(GlobalOp op) {
  if (op.initializer().hasValue() && op.initial_value().hasValue()) {
    return op->emitOpError()
           << "globals can have either an initializer or an initial value";
  } else if (op.initializer().hasValue()) {
    // Ensure initializer returns the same value as the global.
    auto initializerFunc = SymbolTable::lookupNearestSymbolFrom<mlir::FuncOp>(
        op, op.initializerAttr());
    if (!initializerFunc) {
      return op.emitOpError() << "initializer function " << op.initializerAttr()
                              << " not found or wrong type";
    }
    if (initializerFunc.getType().getNumInputs() != 0 ||
        initializerFunc.getType().getNumResults() != 1 ||
        !isGlobalTypeCompatible(op.type(),
                                initializerFunc.getType().getResult(0))) {
      return op->emitOpError()
             << "initializer type mismatch; global " << op.getSymbolName()
             << " is " << op.type() << " but initializer function "
             << initializerFunc.getName() << " is "
             << initializerFunc.getType();
    }
  } else if (op.initial_value().hasValue()) {
    // Ensure the value is something we can convert to a const.
    if (!isGlobalTypeCompatible(op.type(), op.initial_valueAttr().getType())) {
      return op->emitOpError()
             << "initial value type mismatch; global " << op.getSymbolName()
             << " is " << op.type() << " but initial value provided is "
             << op.initial_valueAttr().getType();
    }
  }
  return success();
}

IREE::Util::GlobalOp GlobalAddressOp::getGlobalOp() {
  return SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      getOperation()->getParentOp(), global());
}

void GlobalAddressOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), Twine("ptr_" + global()).str());
}

static LogicalResult verifyGlobalAddressOp(GlobalAddressOp op) {
  auto globalOp = op.getGlobalOp();
  if (!globalOp) {
    return op.emitOpError() << "undefined global: " << op.global();
  }
  return success();
}

void GlobalLoadOp::build(OpBuilder &builder, OperationState &state,
                         GlobalOp globalOp, ArrayRef<NamedAttribute> attrs) {
  state.addTypes({globalOp.type()});
  state.addAttribute("global", builder.getSymbolRefAttr(globalOp));
  state.attributes.append(attrs.begin(), attrs.end());
}

IREE::Util::GlobalOp GlobalLoadOp::getGlobalOp() {
  return SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      getOperation()->getParentOp(), global());
}

bool GlobalLoadOp::isGlobalImmutable() { return !getGlobalOp().is_mutable(); }

void GlobalLoadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), global());
}

void GlobalLoadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // HACK: works around the lack of symbol side effects in mlir by only saying
  // we have a side-effect if the variable we are loading is mutable.
  auto globalOp =
      SymbolTable::lookupNearestSymbolFrom<GlobalOp>(*this, global());
  assert(globalOp);
  if (globalOp.is_mutable()) {
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

static LogicalResult verifyGlobalLoadOp(GlobalLoadOp op) {
  auto globalOp = op.getGlobalOp();
  if (!globalOp) {
    return op->emitOpError() << "undefined global: " << op.global();
  }
  auto loadType = op->getResult(0).getType();
  if (!isGlobalTypeCompatible(globalOp.type(), loadType)) {
    return op->emitOpError()
           << "global type mismatch; global " << op.global() << " is "
           << globalOp.type() << " but load is " << loadType;
  }
  return success();
}

static LogicalResult verifyGlobalLoadIndirectOp(GlobalLoadIndirectOp &op) {
  auto globalType =
      op.global().getType().cast<IREE::Util::PtrType>().getTargetType();
  auto loadType = op.result().getType();
  if (!isGlobalTypeCompatible(globalType, loadType)) {
    return op.emitOpError() << "global type mismatch; global pointer is "
                            << globalType << " but load is " << loadType;
  }
  return success();
}

IREE::Util::GlobalOp GlobalStoreOp::getGlobalOp() {
  return SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      getOperation()->getParentOp(), global());
}

static LogicalResult verifyGlobalStoreOp(GlobalStoreOp op) {
  auto globalOp = op.getGlobalOp();
  if (!globalOp) {
    return op->emitOpError() << "undefined global: " << op.global();
  }
  auto storeType = op->getOperand(0).getType();
  if (globalOp.type() != storeType) {
    return op->emitOpError()
           << "global type mismatch; global " << op.global() << " is "
           << globalOp.type() << " but store is " << storeType;
  }
  if (!globalOp.isMutable()) {
    return op->emitOpError() << "global " << op.global()
                             << " is not mutable and cannot be stored to";
  }
  return success();
}

static LogicalResult verifyGlobalStoreIndirectOp(GlobalStoreIndirectOp &op) {
  auto globalType =
      op.global().getType().cast<IREE::Util::PtrType>().getTargetType();
  auto storeType = op.value().getType();
  if (!isGlobalTypeCompatible(globalType, storeType)) {
    return op.emitOpError() << "global type mismatch; global pointer is "
                            << globalType << " but store is " << storeType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

static ParseResult parseListTypeGet(OpAsmParser &parser, Type &listType,
                                    Type &elementType) {
  if (failed(parser.parseType(listType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected !util.list<T> type");
  }
  auto listElementType = listType.cast<ListType>().getElementType();
  if (succeeded(parser.parseOptionalArrow())) {
    // Use overridden type - required for variants only.
    if (failed(parser.parseType(elementType))) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "expected an element type when specifying list access types");
    }
    if (!ListType::canImplicitlyCast(listElementType, elementType)) {
      return parser.emitError(
          parser.getCurrentLocation(),
          "list access types must match the same base type as the list element "
          "type (when not variant)");
    }
  } else {
    // Use list element type as the result element type.
    elementType = listElementType;
  }
  return success();
}

static void printListTypeGet(OpAsmPrinter &printer, Operation *, Type listType,
                             Type elementType) {
  printer.printType(listType);
  auto listElementType = listType.cast<ListType>().getElementType();
  if (listElementType != elementType) {
    printer.printArrowTypeList(ArrayRef<Type>{elementType});
  }
}

static ParseResult parseListTypeSet(OpAsmParser &parser, Type &listType,
                                    Type &elementType) {
  Type leadingType;
  if (failed(parser.parseType(leadingType))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected element type or !util.list<T> type");
  }
  if (succeeded(parser.parseOptionalArrow())) {
    elementType = leadingType;
    if (failed(parser.parseType(listType)) || !listType.isa<ListType>()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
  } else {
    if (!leadingType.isa<ListType>()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected an !util.list<T> type");
    }
    listType = leadingType;
    elementType = listType.cast<ListType>().getElementType();
  }
  return success();
}

static void printListTypeSet(OpAsmPrinter &printer, Operation *, Type listType,
                             Type elementType) {
  auto listElementType = listType.cast<ListType>().getElementType();
  if (listElementType != elementType) {
    printer.printType(elementType);
    printer.printArrowTypeList(ArrayRef<Type>{listType});
  } else {
    printer.printType(listType);
  }
}

static LogicalResult verifyListGetOp(ListGetOp &op) {
  auto listType = op.list().getType().cast<IREE::Util::ListType>();
  auto elementType = listType.getElementType();
  auto resultType = op.result().getType();
  if (!ListType::canImplicitlyCast(elementType, resultType)) {
    return op.emitError() << "list contains " << elementType
                          << " and cannot be accessed as " << resultType;
  }
  return success();
}

static LogicalResult verifyListSetOp(ListSetOp &op) {
  auto listType = op.list().getType().cast<IREE::Util::ListType>();
  auto elementType = listType.getElementType();
  auto valueType = op.value().getType();
  if (!ListType::canImplicitlyCast(valueType, elementType)) {
    return op.emitError() << "list contains " << elementType
                          << " and cannot be mutated as " << valueType;
  }
  return success();
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilOps.cpp.inc"
