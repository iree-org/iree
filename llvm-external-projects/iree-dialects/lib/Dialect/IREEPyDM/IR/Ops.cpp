// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"

#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::iree_pydm;

using PyBoolType = mlir::iree_pydm::BoolType;
using PyConstantOp = mlir::iree_pydm::ConstantOp;
using PyIntegerType = mlir::iree_pydm::IntegerType;
using PyRealType = mlir::iree_pydm::RealType;
using PyCallOp = mlir::iree_pydm::CallOp;
using PyFuncOp = mlir::iree_pydm::FuncOp;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static Value getNumericZeroConstant(Location loc, Type numericType,
                                    OpBuilder &builder) {
  return TypeSwitch<Type, Value>(numericType)
      .Case([&](PyBoolType t) -> Value {
        return builder.create<PyConstantOp>(loc, t, builder.getBoolAttr(false));
      })
      .Case([&](PyIntegerType t) -> Value {
        return builder.create<PyConstantOp>(loc, t,
                                            builder.getI64IntegerAttr(0));
      })
      .Case([&](PyRealType t) -> Value {
        return builder.create<PyConstantOp>(loc, t,
                                            builder.getF64FloatAttr(0.0));
      });
}

static Value getBoolConstant(Location loc, bool pred, OpBuilder &builder) {
  return builder.create<PyConstantOp>(loc, builder.getType<BoolType>(),
                                      builder.getBoolAttr(pred));
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

OpFoldResult PyConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult NoneOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return UnitAttr::get(getContext());
}

OpFoldResult SuccessOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return UnitAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

void AllocFreeVarOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), name());
}

//===----------------------------------------------------------------------===//
// ApplyCompareOp
//===----------------------------------------------------------------------===//

namespace {

/// Matches an `apply_compare` op where both operands are defined by
/// `box` ops that have the same operand type. Replaces the operands with the
/// operands of the `box`.
struct UnboxApplyCompareOperands : public OpRewritePattern<ApplyCompareOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ApplyCompareOp op,
                                PatternRewriter &rewriter) const override {
    auto boxLeft = op.left().getDefiningOp<BoxOp>();
    auto boxRight = op.right().getDefiningOp<BoxOp>();
    if (!boxLeft || !boxRight) return failure();
    if (boxLeft.primitive().getType() != boxRight.primitive().getType())
      return failure();
    rewriter.replaceOpWithNewOp<ApplyCompareOp>(
        op, rewriter.getType<BoolType>(), op.dunder_nameAttr(),
        boxLeft.primitive(), boxRight.primitive());
    return success();
  }
};

}  // namespace

void ApplyCompareOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<UnboxApplyCompareOperands>(context);
}

//===----------------------------------------------------------------------===//
// AsBoolOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldAsBoolFromBool : public OpRewritePattern<AsBoolOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsBoolOp op,
                                PatternRewriter &rewriter) const override {
    if (op.value().getType().isa<BoolType>()) {
      rewriter.replaceOp(op, op.value());
      return success();
    }
    return failure();
  }
};

struct FoldAsBoolFromNumeric : public OpRewritePattern<AsBoolOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsBoolOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptType = op.value().getType().dyn_cast<PythonTypeInterface>();
    if (!ptType) return failure();
    if (!ptType.getNumericPromotionOrder()) return failure();

    auto boolType = rewriter.getType<BoolType>();
    Value zeroValue =
        getNumericZeroConstant(loc, op.value().getType(), rewriter);
    Value trueValue = getBoolConstant(loc, true, rewriter);
    Value falseValue = getBoolConstant(loc, false, rewriter);
    Value cmpResult = rewriter.create<ApplyCompareOp>(
        loc, boolType, rewriter.getStringAttr("eq"), op.value(), zeroValue);
    rewriter.replaceOpWithNewOp<SelectOp>(op, boolType, cmpResult, falseValue,
                                          trueValue);
    return success();
  }
};

}  // namespace

void AsBoolOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<FoldAsBoolFromBool, FoldAsBoolFromNumeric>(context);
}

OpFoldResult AsBoolOp::fold(ArrayRef<Attribute> operands) {
  Builder b(getContext());
  // Fold NoneType to False.
  if (value().getType().isa<NoneType>()) {
    return b.getBoolAttr(false);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// BoolToPredOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolToPredOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};
  // Since both BoolType and I1 share the attribute form (an IntegerAttr of I1),
  // we can just return it.
  return operands[0];
}

//===----------------------------------------------------------------------===//
// BoxOp and UnboxOp
//===----------------------------------------------------------------------===//

LogicalResult BoxOp::canonicalize(BoxOp op, PatternRewriter &rewriter) {
  // Sometimes boxes are emitted when the input is an object. Just remove.
  if (op.primitive().getType().isa<ObjectType>()) {
    rewriter.replaceOp(op, op.primitive());
    return success();
  }

  return failure();
}

LogicalResult UnboxOp::canonicalize(UnboxOp unboxOp,
                                    PatternRewriter &rewriter) {
  auto loc = unboxOp.getLoc();

  // Handle the case of an immediate BoxOp producer.
  if (auto boxProducer =
          dyn_cast_or_null<BoxOp>(unboxOp.object().getDefiningOp())) {
    // If the producer is boxing to the same type we are unboxing, then
    // just elide everything.
    if (boxProducer.primitive().getType() == unboxOp.primitive().getType()) {
      auto successValue = rewriter.create<SuccessOp>(
          loc, rewriter.getType<ExceptionResultType>());
      rewriter.replaceOp(unboxOp, {successValue, boxProducer.primitive()});
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DynamicBinaryPromoteOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBinaryPromoteOp::canonicalize(DynamicBinaryPromoteOp op,
                                                   PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto leftType = op.left().getType();
  auto rightType = op.right().getType();
  auto leftResultType = op.getResultTypes()[0];
  auto rightResultType = op.getResultTypes()[1];
  auto leftPt = leftType.dyn_cast<PythonTypeInterface>();
  auto rightPt = rightType.dyn_cast<PythonTypeInterface>();
  if (!leftPt || !rightPt) return failure();

  Optional<int> leftOrder = leftPt.getNumericPromotionOrder();
  Optional<int> rightOrder = rightPt.getNumericPromotionOrder();
  Value newLeft = op.left();
  Value newRight = op.right();

  // Simple case: same types pass through.
  if (leftType == rightType) {
    // Nothing - pass-through rewrite.
  } else if (leftOrder && rightOrder) {
    // Both numeric.
    if (*leftOrder > *rightOrder) {
      newRight = rewriter.create<PromoteNumericOp>(loc, leftType, newRight);
    }
    if (*rightOrder > *leftOrder) {
      newLeft = rewriter.create<PromoteNumericOp>(loc, rightType, newLeft);
    }
  } else {
    return failure();
  }

  // Need to box back to the original type (which will always be a generic
  // object).
  newLeft = rewriter.create<BoxOp>(loc, leftResultType, newLeft);
  newRight = rewriter.create<BoxOp>(loc, rightResultType, newRight);

  rewriter.replaceOp(op, {newLeft, newRight});
  return success();
}

//===----------------------------------------------------------------------===//
// FunctionalIfOp
//===----------------------------------------------------------------------===//

::llvm::StringRef FunctionalIfOp::getDefaultDialect() { return "iree_pydm"; }

static LogicalResult verify(FunctionalIfOp op) {
  if (op.getNumResults() != 0 && op.elseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

  return RegionBranchOpInterface::verifyTypes(op);
}

static ParseResult parseFunctionalIfOp(OpAsmParser &parser,
                                       OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  Type conditionType = builder.getType<PyBoolType>();
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, conditionType, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types)) return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  // IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    // IfOp::ensureTerminator(*elseRegion, parser.getBuilder(),
    // result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

static void print(OpAsmPrinter &p, FunctionalIfOp op) {
  bool printBlockTerminators = false;

  p << " " << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict(op->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void FunctionalIfOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->elseRegion();
  if (elseRegion->empty()) elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  if (auto condAttr = operands.front().dyn_cast_or_null<BoolAttr>()) {
    bool condition = condAttr.getValue();
    // Add the successor regions using the condition.
    regions.push_back(RegionSuccessor(condition ? &thenRegion() : elseRegion));
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&thenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion) regions.push_back(RegionSuccessor(elseRegion));
  }
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

::llvm::StringRef PyFuncOp::getDefaultDialect() { return "iree_pydm"; }

LogicalResult PyFuncOp::verifyType() {
  // TODO: Enforce arg/result invariants.
  return success();
}

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(PyFuncOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult verify(PyFuncOp op) {
  // TODO: Enforce invariants.
  return success();
}

//===----------------------------------------------------------------------===//
// PatternMatchCallOp
//===----------------------------------------------------------------------===//

LogicalResult PatternMatchCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto verifySymbols = [&](ArrayAttr symbols) -> LogicalResult {
    for (auto symbolAttr : symbols) {
      auto symbol = symbolAttr.cast<FlatSymbolRefAttr>();
      PyFuncOp fn =
          symbolTable.lookupNearestSymbolFrom<PyFuncOp>(*this, symbol);
      if (!fn)
        return emitOpError() << "'" << symbol.getValue()
                             << "' does not reference a valid function";
    }
    return success();
  };
  auto genericsAttr = (*this)->getAttrOfType<ArrayAttr>("generic_match");
  if (!genericsAttr)
    return emitOpError(
        "requires a 'generic_match' array of symbol reference attributes");
  if (failed(verifySymbols(genericsAttr))) return failure();

  auto specificsAttr = (*this)->getAttrOfType<ArrayAttr>("specific_match");
  if (!specificsAttr)
    return emitOpError(
        "requires a 'specific_match' array of symbol reference attributes");
  if (failed(verifySymbols(specificsAttr))) return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// PromoteNumericOp
//===----------------------------------------------------------------------===//

OpFoldResult PromoteNumericOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};

  Builder b(getContext());
  Attribute fromAttr = operands[0];
  return TypeSwitch<Type, OpFoldResult>(getResult().getType())
      .Case([&](PyIntegerType toType) -> OpFoldResult {
        return TypeSwitch<Attribute, OpFoldResult>(fromAttr)
            .Case([&](BoolAttr fromBool) -> OpFoldResult {
              return b.getI64IntegerAttr(fromBool.getValue() ? 1 : 0);
            })
            .Default([](Attribute) -> OpFoldResult { return {}; });
      })
      .Case([&](PyRealType toType) -> OpFoldResult {
        return TypeSwitch<Attribute, OpFoldResult>(fromAttr)
            .Case([&](BoolAttr fromBool) -> OpFoldResult {
              return b.getF64FloatAttr(fromBool.getValue() ? 1.0 : 0.0);
            })
            .Case([&](IntegerAttr fromInteger) -> OpFoldResult {
              APInt value = fromInteger.getValue();
              return b.getF64FloatAttr(value.getSExtValue());
            })
            .Default([](Attribute) -> OpFoldResult { return {}; });
      })
      .Default([](Type) -> OpFoldResult { return {}; });
}

LogicalResult PromoteNumericOp::canonicalize(PromoteNumericOp op,
                                             PatternRewriter &rewriter) {
  if (op.input().getType() == op.getResult().getType()) {
    rewriter.replaceOp(op, op.input());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// RaiseOnFailureOp
//===----------------------------------------------------------------------===//

LogicalResult iree_pydm::RaiseOnFailureOp::fold(
    ArrayRef<Attribute> operands, SmallVectorImpl<OpFoldResult> &results) {
  assert(operands.size() == 1 && "expected one fold operand");
  // Unit exception result is success. Just elide.
  if (operands[0] && operands[0].isa<UnitAttr>()) {
    erase();
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0]) return {};

  BoolAttr boolAttr = operands[0].cast<BoolAttr>();
  if (boolAttr.getValue())
    return true_value();
  else
    return false_value();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult PyCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  PyFuncOp fn = symbolTable.lookupNearestSymbolFrom<PyFuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != fnType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }
  }

  return success();
}

FunctionType PyCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// DynamicCallOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  Operation *fn = symbolTable.lookupNearestSymbolFrom(*this, fnAttr);
  if (!fn || !isa<PyFuncOp>(fn))
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";
  return success();
}

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.cpp.inc"
