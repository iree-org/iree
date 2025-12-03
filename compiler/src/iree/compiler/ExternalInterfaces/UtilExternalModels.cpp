// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/UtilExternalModels.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// InferIntDivisibilityOpInterface
//===----------------------------------------------------------------------===//

static IREE::Util::ConstantIntDivisibility
getDivisibilityOfOperand(Value v,
                         IREE::Util::IntegerDivisibility divisibility) {
  if (!divisibility.isUninitialized()) {
    return divisibility.getValue();
  }
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal))) {
    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    return IREE::Util::ConstantIntDivisibility(udiv, sdiv);
  }
  return IREE::Util::ConstantIntDivisibility(1, 1);
}

/// Visits affine expressions and recursively calculates the divisibilities of
/// each subexpression. The final divisibilities of the expression and its
/// subexpressions will be stored in the map for which a reference is provided
/// to the AffineExprDivisibilityFinder (i.e., `divisibilityMap`).
class AffineExprDivisibilityFinder
    : public AffineExprVisitor<AffineExprDivisibilityFinder,
                               IREE::Util::ConstantIntDivisibility> {
public:
  using ExprDivisibilityMap =
      llvm::DenseMap<AffineExpr, IREE::Util::ConstantIntDivisibility>;
  AffineExprDivisibilityFinder(ExprDivisibilityMap &divisibilityMap)
      : divisibilityMap(divisibilityMap) {}

  /// Constant expressions are trivial, since they are always static.
  IREE::Util::ConstantIntDivisibility
  visitConstantExpr(AffineConstantExpr expr) {
    uint64_t constValue = std::abs(expr.getValue());
    return IREE::Util::ConstantIntDivisibility(constValue, constValue);
  }

  /// Dim expressions cannot be analyzed further, so return the divisibility
  /// in `divisibilityMap` if it has been populated by the caller, or fallback
  /// to the minimum divisibility.
  IREE::Util::ConstantIntDivisibility visitDimExpr(AffineDimExpr expr) {
    if (divisibilityMap.contains(expr)) {
      return divisibilityMap[expr];
    }
    return IREE::Util::IntegerDivisibility::getMinDivisibility().getValue();
  }

  /// Symbol expressions cannot be analyzed further, so return the divisibility
  /// in `divisibilityMap` if it has been populated by the caller, or fallback
  /// to the minimum divisibility.
  IREE::Util::ConstantIntDivisibility visitSymbolExpr(AffineSymbolExpr expr) {
    if (divisibilityMap.contains(expr)) {
      return divisibilityMap[expr];
    }
    return IREE::Util::IntegerDivisibility::getMinDivisibility().getValue();
  }

  /// The divisibility of an addition is the GCD of its constituents'
  /// divisibilities. This callback is used for subtraction as well.
  IREE::Util::ConstantIntDivisibility visitAddExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr)) {
      return divisibilityMap[expr];
    }
    IREE::Util::ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    IREE::Util::ConstantIntDivisibility rhsDiv = visit(expr.getRHS());
    return lhsDiv.getUnion(rhsDiv);
  }

  /// The divisibility of a multiplication is the product of its constituents'
  /// divisibilities.
  IREE::Util::ConstantIntDivisibility visitMulExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr)) {
      return divisibilityMap[expr];
    }
    IREE::Util::ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    IREE::Util::ConstantIntDivisibility rhsDiv = visit(expr.getRHS());
    return IREE::Util::ConstantIntDivisibility(lhsDiv.udiv() * rhsDiv.udiv(),
                                               lhsDiv.sdiv() * rhsDiv.sdiv());
  }

  IREE::Util::ConstantIntDivisibility
  visitFloorDivExpr(AffineBinaryOpExpr expr) {
    return visitDivExpr(expr);
  }

  IREE::Util::ConstantIntDivisibility
  visitCeilDivExpr(AffineBinaryOpExpr expr) {
    return visitDivExpr(expr);
  }

  /// Mod expressions could be inferred to be zero in some cases, but for now
  /// just return the minimum divisibility.
  /// TODO(Max191): Handle evenly divisible cases, and ensure that the zero
  /// divisibility propagates properly through parent expressions.
  IREE::Util::ConstantIntDivisibility visitModExpr(AffineBinaryOpExpr expr) {
    return visitInvalidExpr(expr);
  }

private:
  IREE::Util::ConstantIntDivisibility
  visitInvalidExpr(AffineBinaryOpExpr expr) {
    return IREE::Util::IntegerDivisibility::getMinDivisibility().getValue();
  }

  /// Helper shared by ceildiv and floordiv implementations. The divisibility of
  /// a division is simply the quotient of its constituents' divisibilities as
  /// long as the division has no remainder. If there is a remainder, then the
  /// divisibility cannot be easily inferred, so we fallback to the minimum
  /// divisibility.
  IREE::Util::ConstantIntDivisibility visitDivExpr(AffineBinaryOpExpr expr) {
    if (divisibilityMap.contains(expr)) {
      return divisibilityMap[expr];
    }
    IREE::Util::ConstantIntDivisibility lhsDiv = visit(expr.getLHS());
    auto constRhs = dyn_cast<AffineConstantExpr>(expr.getRHS());
    if (!constRhs) {
      return IREE::Util::ConstantIntDivisibility(1, 1);
    }
    int64_t constValue = constRhs.getValue();
    uint64_t divUDiv = lhsDiv.udiv() % static_cast<uint64_t>(constValue) == 0
                           ? lhsDiv.udiv() / static_cast<uint64_t>(constValue)
                           : 1;
    uint64_t divSDiv = lhsDiv.sdiv() % std::abs(constValue) == 0
                           ? lhsDiv.sdiv() / std::abs(constValue)
                           : 1;
    return IREE::Util::ConstantIntDivisibility(divUDiv, divSDiv);
  }

  ExprDivisibilityMap &divisibilityMap;
};

/// Returns the divisibilities of each AffineMap result based on the
/// divisibilities of its dims and symbols. The `dimAndSymbolDivisibilities`
/// should contain the divisibilities of the dims, followed by the
/// divisibilities of the symbols in ascending order by their positions.
static SmallVector<IREE::Util::ConstantIntDivisibility> getResultDivisibilities(
    AffineMap map,
    ArrayRef<IREE::Util::ConstantIntDivisibility> dimAndSymbolDivisibilities) {
  // Seed the AffineExprDivisibilityFinder with the dimAndSymbolDivisibilities.
  llvm::DenseMap<AffineExpr, IREE::Util::ConstantIntDivisibility>
      exprDivisibilityMap;
  SmallVector<AffineExpr> inputExprs;
  inputExprs.append(llvm::map_to_vector(
      llvm::seq<int64_t>(map.getNumDims()),
      [&](int64_t dim) { return getAffineDimExpr(dim, map.getContext()); }));
  inputExprs.append(llvm::map_to_vector(
      llvm::seq<int64_t>(map.getNumSymbols()),
      [&](int64_t sym) { return getAffineSymbolExpr(sym, map.getContext()); }));
  for (auto [expr, divisibility] :
       llvm::zip_equal(inputExprs, dimAndSymbolDivisibilities)) {
    exprDivisibilityMap[expr] = divisibility;
  }
  AffineExprDivisibilityFinder divisibilityFinder(exprDivisibilityMap);

  // Walk each result expression and compute their divisibilities.
  SmallVector<IREE::Util::ConstantIntDivisibility> resultDivisibilities;
  for (AffineExpr resultExpr : map.getResults()) {
    resultDivisibilities.push_back(divisibilityFinder.visit(resultExpr));
  }
  return resultDivisibilities;
}

struct AffineApplyInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          AffineApplyInferIntDivisibilityOpInterface, affine::AffineApplyOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto affineApplyOp = cast<affine::AffineApplyOp>(op);
    SmallVector<IREE::Util::ConstantIntDivisibility> operandDivisibilities;
    for (auto [operand, divisibility] :
         llvm::zip(affineApplyOp.getOperands(), argDivs)) {
      operandDivisibilities.push_back(
          getDivisibilityOfOperand(operand, divisibility));
    }

    SmallVector<IREE::Util::ConstantIntDivisibility> resultDivisibilities =
        getResultDivisibilities(affineApplyOp.getMap(), operandDivisibilities);
    for (auto [result, divisibility] :
         llvm::zip_equal(affineApplyOp->getResults(), resultDivisibilities)) {
      setResultDivs(result, divisibility);
    }
  }
};

/// Infer the result divisibility of an affine.min or affine.max operation
/// based on its operand divisibilities. The result divisibility is the GCD
/// of the divisibilities of each of the affine map results, because the result
/// of the affine.min/max op could be any of these results.
template <typename MinOrMaxTy>
static void inferAffineMinOrMaxResultDivisibility(
    MinOrMaxTy minOrMaxOp, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
    IREE::Util::SetIntDivisibilityFn setResultDivs) {
  static_assert(
      llvm::is_one_of<MinOrMaxTy, affine::AffineMinOp,
                      affine::AffineMaxOp>::value,
      "MinOrMaxTy must be affine::AffineMinOp or affine::AffineMaxOp");
  SmallVector<IREE::Util::ConstantIntDivisibility> operandDivisibilities;
  for (auto [operand, divisibility] :
       llvm::zip(minOrMaxOp.getOperands(), argDivs)) {
    operandDivisibilities.push_back(
        getDivisibilityOfOperand(operand, divisibility));
  }

  SmallVector<IREE::Util::ConstantIntDivisibility> resultDivisibilities =
      getResultDivisibilities(minOrMaxOp.getMap(), operandDivisibilities);

  IREE::Util::ConstantIntDivisibility resultDivisibility =
      resultDivisibilities.pop_back_val();
  for (auto divisibility : resultDivisibilities) {
    resultDivisibility = resultDivisibility.getUnion(divisibility);
  }
  setResultDivs(minOrMaxOp.getResult(), resultDivisibility);
}

struct AffineMinInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          AffineMinInferIntDivisibilityOpInterface, affine::AffineMinOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto affineMinOp = cast<affine::AffineMinOp>(op);
    inferAffineMinOrMaxResultDivisibility(affineMinOp, argDivs, setResultDivs);
  }
};

struct AffineMaxInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          AffineMaxInferIntDivisibilityOpInterface, affine::AffineMaxOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto affineMaxOp = cast<affine::AffineMaxOp>(op);
    inferAffineMinOrMaxResultDivisibility(affineMaxOp, argDivs, setResultDivs);
  }
};

struct ArithConstantInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithConstantInferIntDivisibilityOpInterface, arith::ConstantOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto constOp = cast<arith::ConstantOp>(op);
    auto constAttr = dyn_cast_if_present<IntegerAttr>(constOp.getValue());
    if (constAttr) {
      const APInt &value = constAttr.getValue();
      uint64_t udiv = value.getZExtValue();
      uint64_t sdiv = std::abs(value.getSExtValue());
      setResultDivs(constOp.getResult(),
                    IREE::Util::ConstantIntDivisibility(udiv, sdiv));
    }
  }
};

struct ArithMulIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithMulIInferIntDivisibilityOpInterface, arith::MulIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto mulOp = cast<arith::MulIOp>(op);

    auto lhsDivisibility = getDivisibilityOfOperand(mulOp.getLhs(), argDivs[0]);
    auto rhsDivisibility = getDivisibilityOfOperand(mulOp.getRhs(), argDivs[1]);

    uint64_t mulUDiv = lhsDivisibility.udiv() * rhsDivisibility.udiv();
    uint64_t mulSDiv = lhsDivisibility.sdiv() * rhsDivisibility.sdiv();

    setResultDivs(mulOp.getResult(),
                  IREE::Util::ConstantIntDivisibility(mulUDiv, mulSDiv));
  }
};

struct ArithDivUIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithDivUIInferIntDivisibilityOpInterface, arith::DivUIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto divOp = cast<arith::DivUIOp>(op);

    APInt intVal;
    if (!matchPattern(divOp.getRhs(), m_ConstantInt(&intVal))) {
      return;
    }

    auto lhsDivisibility = getDivisibilityOfOperand(divOp.getLhs(), argDivs[0]);

    uint64_t divUDiv = lhsDivisibility.udiv() % intVal.getZExtValue() == 0
                           ? lhsDivisibility.udiv() / intVal.getZExtValue()
                           : 1;
    uint64_t divSDiv =
        lhsDivisibility.sdiv() % std::abs(intVal.getSExtValue()) == 0
            ? lhsDivisibility.sdiv() / std::abs(intVal.getSExtValue())
            : 1;

    setResultDivs(divOp, IREE::Util::ConstantIntDivisibility(divUDiv, divSDiv));
  }
};

//===----------------------------------------------------------------------===//
// ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

/// For some reason, this interface has to be done as an external model.
struct UtilAssumeIntValueBoundsOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          UtilAssumeIntValueBoundsOpInterface, IREE::Util::AssumeIntOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto assumeOp = cast<IREE::Util::AssumeIntOp>(op);
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "value is a result of this index op");
    auto [min, max] =
        assumeOp.getUnionedUnsignedRange(result.getResultNumber());

    std::optional<int64_t> udiv =
        assumeOp.getUnionedUnsignedDivisor(result.getResultNumber());

    if (min) {
      cstr.bound(result) >= *min;
    }
    if (max) {
      cstr.bound(result) <= *max;
    }
    if (udiv) {
      // To represent the divisibility guarantee, emit a bound clamping the
      // value to the udiv value. i.e.
      //
      // v == floordiv(v, udiv) * udiv
      //
      // Mod/divide folders can cleanup such terms with the appropriate bounds
      // query.
      AffineExpr expr =
          cstr.getExpr(assumeOp.getOperand(result.getResultNumber()));
      AffineExpr udivCst =
          getAffineConstantExpr(udiv.value(), op->getContext());
      AffineExpr clampExpr = expr.floorDiv(udivCst) * udivCst;
      cstr.bound(result) == clampExpr;
    }
  }
};

//===----------------------------------------------------------------------===//
// GlobalOpInterface
//===----------------------------------------------------------------------===//

struct GlobalOpInterfaceExternalModel
    : public IREE::Util::GlobalOpInterface::ExternalModel<
          GlobalOpInterfaceExternalModel, ml_program::GlobalOp> {
  Attribute getGlobalInitialValue(Operation *op) const {
    return cast<ml_program::GlobalOp>(op).getValueAttr();
  }
  void setGlobalInitialValue(Operation *op, Attribute value) const {
    if (value) {
      cast<ml_program::GlobalOp>(op).setValueAttr(value);
    } else {
      cast<ml_program::GlobalOp>(op).removeValueAttr();
    }
  }

  IREE::Util::InliningPolicyAttrInterface
  getGlobalInliningPolicy(Operation *op) const {
    if (op->hasAttr("noinline"))
      return IREE::Util::InlineNeverAttr::get(op->getContext());
    return {};
  }
  void
  setGlobalInliningPolicy(Operation *op,
                          IREE::Util::InliningPolicyAttrInterface value) const {
    if (isa_and_nonnull<IREE::Util::InlineNeverAttr>(value)) {
      op->setAttr("noinline", UnitAttr::get(op->getContext()));
    } else {
      op->removeAttr("noinline");
    }
  }

  IREE::Util::GlobalLoadOpInterface createLoadOp(Operation *op, Location loc,
                                                 OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    if (globalOp.getIsMutable()) {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          ml_program::GlobalLoadOp::create(builder, loc, globalOp.getType(),
                                           FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    } else {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          ml_program::GlobalLoadConstOp::create(
              builder, loc, globalOp.getType(),
              FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    }
  }

  IREE::Util::GlobalStoreOpInterface createStoreOp(Operation *op, Location loc,
                                                   Value value,
                                                   OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    return cast<IREE::Util::GlobalStoreOpInterface>(
        ml_program::GlobalStoreOp ::create(
            builder, loc, FlatSymbolRefAttr::get(globalOp), value)
            .getOperation());
  }
};

//===----------------------------------------------------------------------===//
// NumericCastOpInterface
//===----------------------------------------------------------------------===//

// Since all details of the interface are provided via default implementations,
// we can just have one templated external model to apply per op, vs one
// explicit model per op.
struct GenericNumericCastExternalModel {
  template <typename OpTy>
  struct ExternalModel
      : public IREE::Util::NumericCastOpInterface::ExternalModel<
            ExternalModel<OpTy>, OpTy> {};

  template <typename OpTy>
  static void add(MLIRContext *context) {
    OpTy::template attachInterface<ExternalModel<OpTy>>(*context);
  }

  template <typename OpTy1, typename OpTy2, typename... More>
  static void add(MLIRContext *context) {
    add<OpTy1>(context);
    add<OpTy2, More...>(context);
  }
};

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

struct InsertSliceOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          InsertSliceOpTiedOpInterface, tensor::InsertSliceOp> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        insertSliceOp.getDest());
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    return {1}; // dest
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    return {1}; // dest
  }
};

template <typename OpTy>
struct LinalgOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          LinalgOpTiedOpInterface<OpTy>, OpTy> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        linalgOp.getDpsInits()[resultIndex]);
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return {linalgOp.getDpsInitsMutable()[resultIndex].getOperandNumber()};
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    SmallVector<int64_t> result;
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      result.push_back(*getTiedResultOperandIndex(op, i));
    return result;
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `TiedOpInterface` with each of them.
template <typename... Ops>
struct LinalgOpTiedOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (void)std::initializer_list<int>{
        0,
        (Ops::template attachInterface<LinalgOpTiedOpInterface<Ops>>(*context),
         0)...};
  }
};

//===----------------------------------------------------------------------===//
// HoistableOpInterface
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct UnhoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          UnhoistableOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return false; }
  bool isHoistableLeafOp(Operation *) const { return false; }
};

template <typename OpTy>
struct HoistableNonLeafOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableNonLeafOpInterface<OpTy>, OpTy> {
  bool isHoistableLeafOp(Operation *) const { return false; }
};

// The default interface is always hoistable. This acts as an override
// for other default hoistability checks as the interface is checked
// first.
template <typename OpTy>
struct AlwaysHoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          AlwaysHoistableOpInterface<OpTy>, OpTy> {};

template <typename OpTy>
struct HoistableLinalgOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableLinalgOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return true; }

  // Determines if a linalg op is a hoistable leaf, based on heuristics.
  bool isHoistableLeafOp(Operation *op) const {
    // Don't hoist bit extend ops because fusing them with their
    // consumers prevents materializing the high bit-width tensor and they
    // preform very little real computation.
    if (IREE::LinalgExt::isBitExtendOp(op)) {
      return false;
    }

    // Hoist all non-generic linalg ops except for fill ops which should be
    // fused with their consumers.
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp) {
      return !isa<linalg::FillOp>(op);
    }

    // Don't hoist ops with no tensor inputs. They are likely to be fill-like
    // or sequences (from `linalg.index`) which can be fused with their
    // consumers.
    if (IREE::LinalgExt::hasOnlyScalarInputs(genericOp)) {
      return false;
    }

    // Don't hoist broadcast-like ops because fusing them makes the new
    // op cheaper.
    if (linalg::isaBroadcastOpInterface(genericOp).has_value()) {
      return false;
    }

    // Hoist all other ops.
    return true;
  }
  bool isAtomicallyHoistableOp(Operation *) const { return true; }
  bool isOperandHoistable(Operation *, OpOperand *) const { return true; }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated Hoistable___OpInterface.
template <typename... Ops>
struct UnhoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<UnhoistableOpInterface<Ops>>(*context), ...);
  }
};

template <typename... Ops>
struct HoistableNonLeafOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableNonLeafOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct AlwaysHoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<AlwaysHoistableOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct HoistableLinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableLinalgOpInterface<Ops>>(*context),
     ...);
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<ml_program::MLProgramDialect>();
  registry.insert<tensor::TensorDialect>();

  registry.addExtension(
      +[](MLIRContext *context, ml_program::MLProgramDialect *dialect) {
        ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(
            *context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            arith::ArithDialect *dialect) {
    GenericNumericCastExternalModel::add<
        arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
        arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp, arith::TruncFOp,
        arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp>(context);
    arith::ConstantOp::attachInterface<
        ArithConstantInferIntDivisibilityOpInterface>(*context);
    arith::MulIOp::attachInterface<ArithMulIInferIntDivisibilityOpInterface>(
        *context);
    arith::DivUIOp::attachInterface<ArithDivUIInferIntDivisibilityOpInterface>(
        *context);
  });

  registry.addExtension(
      +[](MLIRContext *context, affine::AffineDialect *dialect) {
        affine::AffineApplyOp::attachInterface<
            AffineApplyInferIntDivisibilityOpInterface>(*context);
        affine::AffineMinOp::attachInterface<
            AffineMinInferIntDivisibilityOpInterface>(*context);
        affine::AffineMaxOp::attachInterface<
            AffineMaxInferIntDivisibilityOpInterface>(*context);
      });

  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        tensor::InsertSliceOp::attachInterface<InsertSliceOpTiedOpInterface>(
            *context);
      });

  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `TiedOpInterface` to all ops
        // one-by-one.
        LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
        >::registerOpInterface(context);
  });

  // Hoistable Op Interface registration.

  // Register hoistable op interfaces for Encoding ops.
  registry.addExtension(
      +[](MLIRContext *context, IREE::Encoding::IREEEncodingDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::Encoding::SetEncodingOp>::registerOpInterface(context);
      });

  // Register hoistable op interfaces for Flow ops.
  registry.addExtension(
      +[](MLIRContext *context, IREE::Flow::FlowDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::Flow::DispatchWorkgroupCountOp>::registerOpInterface(context);

        AlwaysHoistableOpInterfaceHelper<
            IREE::Flow::TensorEncodeOp>::registerOpInterface(context);
      });

  // Register hoistable op interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Structured op implementations and a handful of pure ops are included.
        // Notably: IndexOp is not included because it establishes a hidden
        // dependency to the iterator and is non-const.

        // Register all LinalgOps ops. `LinalgOp` is an interface and it is
        // not possible to attach an external interface to an existing
        // interface. Therefore, attach the `HoistableLinalgOpInterface` to all
        // ops one-by-one.
        HoistableLinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
        UnhoistableOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
            >::registerOpInterface(context);

        AlwaysHoistableOpInterfaceHelper<
            linalg::PackOp, linalg::UnPackOp>::registerOpInterface(context);
      });
  // Register hoistable op interfaces for tensor ops.
  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        // Never hoist empty and other pure metadata ops as a leaf. It's fine to
        // hoist them as a part of a larger constant tree that does actual work.
        HoistableNonLeafOpInterfaceHelper<
            tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ExtractSliceOp>::registerOpInterface(context);
        // Cases of trivial pack/unpack should be handled as canonicalizations
        // before we get here, thus we're safe to always hoist.
        AlwaysHoistableOpInterfaceHelper<tensor::PadOp>::registerOpInterface(
            context);
      });
  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        IREE::Util::AssumeIntOp::attachInterface<
            UtilAssumeIntValueBoundsOpInterface>(*context);
      });
}

} // namespace mlir::iree_compiler
