// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"

#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

using llvm::dbgs;

using PyBoolType = PYDM::BoolType;
using PyConstantOp = PYDM::ConstantOp;
using PyIntegerType = PYDM::IntegerType;
using PyRealType = PYDM::RealType;
using PyCallOp = PYDM::CallOp;
using PyFuncOp = PYDM::FuncOp;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {

/// Generic pattern to unbox any operands that are a specific object
/// type (i.e. object<integer>).
struct UnboxOperands : public RewritePattern {
  UnboxOperands(StringRef rootName, MLIRContext *context,
                Optional<llvm::SmallSet<int, 4>> operandIndices = None)
      : RewritePattern(rootName, 1, context), operandIndices(operandIndices) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    bool changed = false;
    SmallVector<Value> operands(op->getOperands());
    auto excResultType = rewriter.getType<ExceptionResultType>();
    for (int operandIndex = 0, e = operands.size(); operandIndex < e;
         ++operandIndex) {
      Value &operand = operands[operandIndex];
      if (operandIndices && !operandIndices->contains(operandIndex))
        continue;
      if (auto objectType = operand.getType().dyn_cast<ObjectType>()) {
        Type primitiveType = objectType.getPrimitiveType();
        if (primitiveType) {
          // Unbox.
          auto unboxOp = rewriter.create<UnboxOp>(
              loc, TypeRange{excResultType, primitiveType}, operand);
          operand = unboxOp.primitive();
          changed = true;
        }
      }
    }

    if (changed) {
      rewriter.updateRootInPlace(op, [&]() { op->setOperands(operands); });
      return success();
    }

    return failure();
  }
  Optional<llvm::SmallSet<int, 4>> operandIndices;
};

} // namespace

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
// ApplyBinaryOp
//===----------------------------------------------------------------------===//

namespace {
struct ApplyBinaryToSequenceClone : public OpRewritePattern<ApplyBinaryOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ApplyBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (op.dunder_name() != "mul")
      return failure();
    Value listOperand;
    Value countOperand;
    if (isBuiltinSequence(op.left()) && isInteger(op.right())) {
      listOperand = op.left();
      countOperand = op.right();
    } else if (isInteger(op.left()) && isBuiltinSequence(op.right())) {
      countOperand = op.left();
      listOperand = op.right();
    } else {
      return failure();
    }
    Type resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<SequenceCloneOp>(op, resultType, listOperand,
                                                 countOperand);
    return success();
  }

  static bool isBuiltinSequence(Value operand) {
    return operand.getType().isa<PYDM::ListType, PYDM::TupleType>();
  }
  static bool isInteger(Value operand) {
    return operand.getType().isa<PYDM::IntegerType>();
  }
};
} // namespace

void ApplyBinaryOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<UnboxOperands>(getOperationName(), context);
  patterns.add<ApplyBinaryToSequenceClone>(context);
}

bool ApplyBinaryOp::refineResultTypes() {
  auto leftType = left().getType();
  auto rightType = right().getType();
  auto applyUpdates = [&](Type newResultType) -> bool {
    if (newResultType != getResult().getType()) {
      getResult().setType(newResultType);
      return true;
    }
    return false;
  };

  // Both numeric types. It is only dynamically legal for statically known
  // numeric types to be the same, in which case the result type must be the
  // same as well.
  auto ptLeft = leftType.dyn_cast<PythonTypeInterface>();
  auto ptRight = rightType.dyn_cast<PythonTypeInterface>();
  if (ptLeft && ptRight && ptLeft.getNumericPromotionOrder() &&
      ptRight.getNumericPromotionOrder()) {
    if (leftType == rightType) {
      return applyUpdates(leftType);
    }
  }

  // (list, integer) or (integer, list) refine to the list type.
  if (dunder_name() == "mul") {
    auto leftList = leftType.dyn_cast<ListType>();
    auto rightList = rightType.dyn_cast<ListType>();
    auto leftInteger = leftType.dyn_cast<IntegerType>();
    auto rightInteger = rightType.dyn_cast<IntegerType>();
    if (leftList && rightInteger) {
      return applyUpdates(leftList);
    } else if (leftInteger && rightList) {
      return applyUpdates(rightList);
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// ApplyCompareOp
//===----------------------------------------------------------------------===//

void ApplyCompareOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<UnboxOperands>(getOperationName(), context);
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
    if (!ptType)
      return failure();
    if (!ptType.getNumericPromotionOrder())
      return failure();

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

} // namespace

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
// AssignSubscriptOp
//===----------------------------------------------------------------------===//

void AssignSubscriptOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  llvm::SmallSet<int, 4> unboxIndices;
  unboxIndices.insert(0);
  unboxIndices.insert(1);
  patterns.add<UnboxOperands>(getOperationName(), context, unboxIndices);
}

//===----------------------------------------------------------------------===//
// BoolToPredOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolToPredOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0])
    return {};
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

  // Box to an appropriate type and static info cast.
  ObjectType objectType = rewriter.getType<ObjectType>(nullptr);
  if (op.object().getType() == objectType &&
      !op.primitive().getType().isa<ObjectType>()) {
    auto refinedBox = rewriter.create<BoxOp>(
        op.getLoc(),
        rewriter.getType<ObjectType>(
            op.primitive().getType().cast<PrimitiveType>()),
        op.primitive());
    rewriter.replaceOpWithNewOp<StaticInfoCastOp>(op, op.object().getType(),
                                                  refinedBox);
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

namespace {

/// Resolves a DynamicBinaryPromote over numeric operands to either elide
/// or insert specific PromoteNumeric ops.
struct ResolveNumericDynamicBinaryPromote
    : public OpRewritePattern<DynamicBinaryPromoteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBinaryPromoteOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto leftType = op.left().getType();
    auto rightType = op.right().getType();
    auto leftResultType = op.getResultTypes()[0];
    auto rightResultType = op.getResultTypes()[1];
    auto leftPt = leftType.dyn_cast<PythonTypeInterface>();
    auto rightPt = rightType.dyn_cast<PythonTypeInterface>();
    if (!leftPt || !rightPt)
      return failure();

    Optional<int> leftOrder = leftPt.getNumericPromotionOrder();
    Optional<int> rightOrder = rightPt.getNumericPromotionOrder();
    Value newLeft = op.left();
    Value newRight = op.right();

    if (leftOrder && rightOrder) {
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
};

/// If we statically determine one of the arguments to be a concrete, non
/// numeric type, then the op has no meaning and is elided.
struct ElideNonNumericDynamicBinaryPromote
    : public OpRewritePattern<DynamicBinaryPromoteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBinaryPromoteOp op,
                                PatternRewriter &rewriter) const override {
    if ((!isConcreteNonNumericType(op.left().getType()) &&
         !isConcreteNonNumericType(op.right().getType())))
      return failure();

    // Since DynamicBinaryPromote already returns object, and we only match
    // non-object operands, box them back.
    auto loc = op.getLoc();
    auto leftResultType = op.getResultTypes()[0];
    auto rightResultType = op.getResultTypes()[1];
    Value newLeft = rewriter.create<BoxOp>(loc, leftResultType, op.left());
    Value newRight = rewriter.create<BoxOp>(loc, rightResultType, op.right());
    rewriter.replaceOp(op, {newLeft, newRight});
    return success();
  }

  static bool isConcreteNonNumericType(Type t) {
    if (t.isa<ObjectType>())
      return false;
    auto pt = t.dyn_cast<PythonTypeInterface>();
    if (!pt || pt.getNumericPromotionOrder())
      return false;
    return true;
  }
};

} // namespace

void DynamicBinaryPromoteOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ResolveNumericDynamicBinaryPromote>(context);
  patterns.add<UnboxOperands>(getOperationName(), context);
  patterns.add<ElideNonNumericDynamicBinaryPromote>(context);
}

//===----------------------------------------------------------------------===//
// FunctionalIfOp
//===----------------------------------------------------------------------===//

::llvm::StringRef FunctionalIfOp::getDefaultDialect() { return "iree_pydm"; }

LogicalResult FunctionalIfOp::verify() {
  if (getNumResults() != 0 && elseRegion().empty())
    return emitOpError("must have an else block if defining values");
  return success();
}

ParseResult FunctionalIfOp::parse(OpAsmParser &parser, OperationState &result) {
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
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
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
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void FunctionalIfOp::print(OpAsmPrinter &p) {
  FunctionalIfOp op = *this;
  bool printBlockTerminators = false;

  p << " " << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << " ";
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
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
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  if (auto condAttr = operands.front().dyn_cast_or_null<BoolAttr>()) {
    bool condition = condAttr.getValue();
    // Add the successor regions using the condition.
    regions.push_back(RegionSuccessor(condition ? &thenRegion() : elseRegion));
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&thenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion)
      regions.push_back(RegionSuccessor(elseRegion));
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

ParseResult PyFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void PyFuncOp::print(OpAsmPrinter &p) {
  FunctionType fnType = getType();
  function_interface_impl::printFunctionOp(
      p, *this, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

//===----------------------------------------------------------------------===//
// MakeListOp
//===----------------------------------------------------------------------===//

LogicalResult MakeListOp::verify() {
  auto listType = list().getType().cast<ListType>();
  switch (listType.getStorageClass()) {
  case CollectionStorageClass::Boxed:
    for (auto element : elements()) {
      if (!element.getType().isa<ObjectType>()) {
        return emitOpError() << "making a list with boxed storage class "
                                "must have object elements. Got: "
                             << element.getType();
      }
    }
    break;
  case CollectionStorageClass::Unboxed:
    for (auto element : elements()) {
      if (element.getType().isa<ObjectType>()) {
        return emitOpError() << "making a list with unboxed storage class "
                                "must not have object elements. Got: "
                             << element.getType();
      }
    }
    break;
  case CollectionStorageClass::Empty:
    if (!elements().empty()) {
      return emitOpError()
             << "making a list with empty storage class must have zero "
                "elements";
    }
    break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

void NegOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<UnboxOperands>(getOperationName(), context);
}

bool NegOp::refineResultTypes() {
  if (value().getType() != getResult().getType()) {
    getResult().setType(value().getType());
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// PatternMatchCallOp
//===----------------------------------------------------------------------===//

LogicalResult
PatternMatchCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
  if (failed(verifySymbols(genericsAttr)))
    return failure();

  auto specificsAttr = (*this)->getAttrOfType<ArrayAttr>("specific_match");
  if (!specificsAttr)
    return emitOpError(
        "requires a 'specific_match' array of symbol reference attributes");
  if (failed(verifySymbols(specificsAttr)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// PromoteNumericOp
//===----------------------------------------------------------------------===//

OpFoldResult PromoteNumericOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0])
    return {};

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

LogicalResult PYDM::RaiseOnFailureOp::canonicalize(RaiseOnFailureOp op,
                                                   PatternRewriter &rewriter) {
  if (op.exc_result().getDefiningOp<SuccessOp>()) {
    op.getOperation()->erase();
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[0])
    return {};

  BoolAttr boolAttr = operands[0].cast<BoolAttr>();
  if (boolAttr.getValue())
    return true_value();
  else
    return false_value();
}

//===----------------------------------------------------------------------===//
// SequenceCloneOp
//===----------------------------------------------------------------------===//

bool SequenceCloneOp::refineResultTypes() {
  if (sequence().getType() != getResult().getType()) {
    getResult().setType(sequence().getType());
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// SubscriptOp
//===----------------------------------------------------------------------===//

void SubscriptOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<UnboxOperands>(getOperationName(), context);
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

LogicalResult
DynamicCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.cpp.inc"
