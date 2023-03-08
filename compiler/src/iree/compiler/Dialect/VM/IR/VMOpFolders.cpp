// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <type_traits>

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//
// TODO(benvanik): share these among dialects?

namespace {

/// Creates a constant zero attribute matching the given type.
Attribute zeroOfType(Type type) {
  return Builder(type.getContext()).getZeroAttr(type);
}

/// Creates a constant one attribute matching the given type.
Attribute oneOfType(Type type) {
  Builder builder(type.getContext());
  if (type.isa<FloatType>()) {
    return builder.getFloatAttr(type, 1.0);
  } else if (auto integerTy = type.dyn_cast<IntegerType>()) {
    return builder.getIntegerAttr(integerTy, APInt(integerTy.getWidth(), 1));
  } else if (type.isa<RankedTensorType, VectorType>()) {
    auto vtType = type.cast<ShapedType>();
    auto element = oneOfType(vtType.getElementType());
    if (!element) return {};
    return DenseElementsAttr::get(vtType, element);
  }
  return {};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

namespace {

// Deletes empty vm.initializer ops.
struct DropEmptyInitializerOp : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBody().getBlocks().size() != 1) return failure();
    auto &block = op.getBody().front();
    if (block.empty() || isa<ReturnOp>(block.front())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Inlines constant stores from initializers into the global initializer.
// This is not strictly required but can help our initialization code perform
// more efficient initialization of large numbers of primitive values.
struct InlineConstGlobalInitializer : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> deadOps;
    op.walk([&](Operation *op) {
      if (!isGlobalStoreOp(op)) return;
      auto value = op->getOperand(0);
      Attribute valueAttr;
      if (!matchPattern(value, m_Constant(&valueAttr))) return;
      auto globalRefAttr = op->getAttrOfType<SymbolRefAttr>("global");
      assert(globalRefAttr);
      auto globalOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
              op, globalRefAttr);
      rewriter.updateRootInPlace(
          globalOp, [&]() { globalOp.setGlobalInitialValue(valueAttr); });
      deadOps.push_back(op);
    });
    if (deadOps.empty()) return failure();
    for (auto deadOp : deadOps) rewriter.eraseOp(deadOp);
    return success();
  }

  bool isGlobalStoreOp(Operation *op) const {
    // TODO(benvanik): trait/interface to make this more generic?
    return isa<IREE::VM::GlobalStoreI32Op>(op) ||
           isa<IREE::VM::GlobalStoreI64Op>(op) ||
           isa<IREE::VM::GlobalStoreF32Op>(op) ||
           isa<IREE::VM::GlobalStoreF64Op>(op) ||
           isa<IREE::VM::GlobalStoreRefOp>(op);
  }
};

}  // namespace

void InitializerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<DropEmptyInitializerOp, InlineConstGlobalInitializer>(context);
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

namespace {

/// Drops initial_values from globals where the value is 0, as by default all
/// globals are zero-initialized upon module load.
template <typename T>
struct DropDefaultConstGlobalOpInitializer : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInitialValue().has_value()) return failure();
    if (auto value =
            op.getInitialValueAttr().template dyn_cast<IntegerAttr>()) {
      if (value.getValue() != 0) return failure();
    } else if (auto value =
                   op.getInitialValueAttr().template dyn_cast<FloatAttr>()) {
      if (value.getValue().isNonZero()) return failure();
    }
    auto visibility = op.getVisibility();
    auto newOp = rewriter.replaceOpWithNewOp<T>(
        op, op.getSymName(), op.getIsMutable(), op.getType(),
        llvm::to_vector<4>(op->getDialectAttrs()));
    newOp.setVisibility(visibility);
    return success();
  }
};

}  // namespace

void GlobalI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<DropDefaultConstGlobalOpInitializer<GlobalI32Op>>(context);
}

void GlobalI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<DropDefaultConstGlobalOpInitializer<GlobalI64Op>>(context);
}

void GlobalF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<DropDefaultConstGlobalOpInitializer<GlobalF32Op>>(context);
}

void GlobalF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<DropDefaultConstGlobalOpInitializer<GlobalF64Op>>(context);
}

void GlobalRefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void GlobalLoadI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {}

void GlobalLoadI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {}

void GlobalLoadF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {}

void GlobalLoadF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {}

namespace {

template <typename INDIRECT, typename DIRECT>
struct PropagateGlobalLoadAddress : public OpRewritePattern<INDIRECT> {
  using OpRewritePattern<INDIRECT>::OpRewritePattern;
  LogicalResult matchAndRewrite(INDIRECT op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<IREE::Util::GlobalAddressOpInterface>(
            op.getGlobal().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<DIRECT>(op, op.getValue().getType(),
                                          addressOp.getGlobalAttr());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalLoadIndirectI32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectI32Op, GlobalLoadI32Op>>(
      context);
}

void GlobalLoadIndirectI64Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectI64Op, GlobalLoadI64Op>>(
      context);
}

void GlobalLoadIndirectF32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectF32Op, GlobalLoadF32Op>>(
      context);
}

void GlobalLoadIndirectF64Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectF64Op, GlobalLoadF64Op>>(
      context);
}

void GlobalLoadIndirectRefOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectRefOp, GlobalLoadRefOp>>(
      context);
}

namespace {

template <typename INDIRECT, typename DIRECT>
struct PropagateGlobalStoreAddress : public OpRewritePattern<INDIRECT> {
  using OpRewritePattern<INDIRECT>::OpRewritePattern;
  LogicalResult matchAndRewrite(INDIRECT op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<IREE::Util::GlobalAddressOpInterface>(
            op.getGlobal().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<DIRECT>(op, op.getValue(),
                                          addressOp.getGlobalAttr());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreIndirectI32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectI32Op, GlobalStoreI32Op>>(
      context);
}

void GlobalStoreIndirectI64Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectI64Op, GlobalStoreI64Op>>(
      context);
}

void GlobalStoreIndirectF32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectF32Op, GlobalStoreF32Op>>(
      context);
}

void GlobalStoreIndirectF64Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectF64Op, GlobalStoreF64Op>>(
      context);
}

void GlobalStoreIndirectRefOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectRefOp, GlobalStoreRefOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

namespace {

template <typename AttrT, typename GeneralOp, typename ZeroOp>
struct FoldZeroConstPrimitive final : public OpRewritePattern<GeneralOp> {
  using OpRewritePattern<GeneralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GeneralOp constOp,
                                PatternRewriter &rewriter) const override {
    if ((std::is_same<AttrT, IntegerAttr>::value &&
         matchPattern(constOp.getResult(), m_Zero())) ||
        (std::is_same<AttrT, FloatAttr>::value &&
         matchPattern(constOp.getResult(), m_AnyZeroFloat()))) {
      rewriter.replaceOpWithNewOp<ZeroOp>(constOp);
      return success();
    }
    return failure();
  }
};

}  // namespace

OpFoldResult ConstI32Op::fold(FoldAdaptor operands) { return getValue(); }

void ConstI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results
      .insert<FoldZeroConstPrimitive<IntegerAttr, ConstI32Op, ConstI32ZeroOp>>(
          context);
}

OpFoldResult ConstI64Op::fold(FoldAdaptor operands) { return getValue(); }

void ConstI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results
      .insert<FoldZeroConstPrimitive<IntegerAttr, ConstI64Op, ConstI64ZeroOp>>(
          context);
}

OpFoldResult ConstF32Op::fold(FoldAdaptor operands) { return getValue(); }

void ConstF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<FloatAttr, ConstF32Op, ConstF32ZeroOp>>(
      context);
}

OpFoldResult ConstF64Op::fold(FoldAdaptor operands) { return getValue(); }

void ConstF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<FloatAttr, ConstF64Op, ConstF64ZeroOp>>(
      context);
}

OpFoldResult ConstI32ZeroOp::fold(FoldAdaptor operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstI64ZeroOp::fold(FoldAdaptor operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstF32ZeroOp::fold(FoldAdaptor operands) {
  return FloatAttr::get(getResult().getType(), 0.0f);
}

OpFoldResult ConstF64ZeroOp::fold(FoldAdaptor operands) {
  return FloatAttr::get(getResult().getType(), 0.0);
}

OpFoldResult ConstRefZeroOp::fold(FoldAdaptor operands) {
  // TODO(benvanik): relace unit attr with a proper null ref attr.
  return UnitAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// vm.ref operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Conditional assignment
//===----------------------------------------------------------------------===//

template <typename T>
static OpFoldResult foldSelectOp(T op) {
  if (matchPattern(op.getCondition(), m_Zero())) {
    // 0 ? x : y = y
    return op.getFalseValue();
  } else if (matchPattern(op.getCondition(), m_NonZero())) {
    // !0 ? x : y = x
    return op.getTrueValue();
  } else if (op.getTrueValue() == op.getFalseValue()) {
    // c ? x : x = x
    return op.getTrueValue();
  }
  return {};
}

OpFoldResult SelectI32Op::fold(FoldAdaptor operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectI64Op::fold(FoldAdaptor operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectF32Op::fold(FoldAdaptor operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectF64Op::fold(FoldAdaptor operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectRefOp::fold(FoldAdaptor operands) {
  return foldSelectOp(*this);
}

template <typename T>
static OpFoldResult foldSwitchOp(T op) {
  APInt indexValue;
  if (matchPattern(op.getIndex(), m_ConstantInt(&indexValue))) {
    // Index is constant and we can resolve immediately.
    int64_t index = indexValue.getSExtValue();
    if (index < 0 || index >= op.getValues().size()) {
      return op.getDefaultValue();
    }
    return op.getValues()[index];
  }

  bool allValuesMatch = true;
  for (auto value : op.getValues()) {
    if (value != op.getDefaultValue()) {
      allValuesMatch = false;
      break;
    }
  }
  if (allValuesMatch) {
    // All values (and the default) are the same so just return it regardless of
    // the provided index.
    return op.getDefaultValue();
  }

  return {};
}

OpFoldResult SwitchI32Op::fold(FoldAdaptor operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchI64Op::fold(FoldAdaptor operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchF32Op::fold(FoldAdaptor operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchF64Op::fold(FoldAdaptor operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchRefOp::fold(FoldAdaptor operands) {
  return foldSwitchOp(*this);
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<APInt(ElementValueT)>>
static Attribute constFoldUnaryOp(Attribute rawOperand,
                                  const CalculationT &calculate) {
  if (auto operand = rawOperand.dyn_cast_or_null<AttrElementT>()) {
    return AttrElementT::get(operand.getType(), calculate(operand.getValue()));
  } else if (auto operand = rawOperand.dyn_cast_or_null<SplatElementsAttr>()) {
    auto elementResult = constFoldUnaryOp<AttrElementT>(
        {operand.getSplatValue<Attribute>()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(operand.getType(), elementResult);
  } else if (auto operand = rawOperand.dyn_cast_or_null<ElementsAttr>()) {
    return operand.cast<DenseIntOrFPElementsAttr>().mapValues(
        operand.getType().getElementType(),
        llvm::function_ref<ElementValueT(const ElementValueT &)>(
            [&](const ElementValueT &value) { return calculate(value); }));
  }
  return {};
}

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
static Attribute constFoldFloatUnaryOp(
    Attribute rawOperand, const std::function<APFloat(APFloat)> &calculate) {
  if (auto operand = rawOperand.dyn_cast_or_null<FloatAttr>()) {
    return FloatAttr::get(operand.getType(), calculate(operand.getValue()));
  } else if (auto operand = rawOperand.dyn_cast_or_null<SplatElementsAttr>()) {
    auto elementResult =
        constFoldFloatUnaryOp({operand.getSplatValue<Attribute>()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(operand.getType(), elementResult);
  } else if (auto operand = rawOperand.dyn_cast_or_null<ElementsAttr>()) {
    return operand.cast<DenseIntOrFPElementsAttr>().mapValues(
        operand.getType().getElementType(),
        llvm::function_ref<APInt(const APFloat &)>([&](const APFloat &value) {
          return calculate(value).bitcastToAPInt();
        }));
  }
  return {};
}

/// Performs const folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
/// Note: return type will match the operand types.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldBinaryOp(Attribute rawLhs, Attribute rawRhs,
                                   const CalculationT &calculate) {
  if (auto lhs = rawLhs.dyn_cast_or_null<AttrElementT>()) {
    auto rhs = rawRhs.dyn_cast_or_null<AttrElementT>();
    if (!rhs) return {};
    return AttrElementT::get(lhs.getType(),
                             calculate(lhs.getValue(), rhs.getValue()));
  } else if (auto lhs = rawLhs.dyn_cast_or_null<SplatElementsAttr>()) {
    // TODO(benvanik): handle splat/otherwise.
    auto rhs = rawRhs.dyn_cast_or_null<SplatElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto elementResult = constFoldBinaryOp<AttrElementT>(
        lhs.getSplatValue<Attribute>(), rhs.getSplatValue<Attribute>(),
        calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(lhs.getType(), elementResult);
  } else if (auto lhs = rawLhs.dyn_cast_or_null<ElementsAttr>()) {
    auto rhs = rawRhs.dyn_cast_or_null<ElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto lhsIt = lhs.getValues<AttrElementT>().begin();
    auto rhsIt = rhs.getValues<AttrElementT>().begin();
    SmallVector<Attribute, 4> resultAttrs(lhs.getNumElements());
    for (int64_t i = 0; i < lhs.getNumElements(); ++i) {
      resultAttrs[i] =
          constFoldBinaryOp<AttrElementT>(*lhsIt, *rhsIt, calculate);
      if (!resultAttrs[i]) return {};
      ++lhsIt;
      ++rhsIt;
    }
    return DenseElementsAttr::get(lhs.getType(), resultAttrs);
  }
  return {};
}

/// Performs const folding `calculate` with element-wise behavior on the three
/// attributes in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldTernaryOp(Attribute rawA, Attribute rawB,
                                    Attribute rawC,
                                    const CalculationT &calculate) {
  if (auto a = rawA.dyn_cast_or_null<AttrElementT>()) {
    auto b = rawB.dyn_cast_or_null<AttrElementT>();
    auto c = rawC.dyn_cast_or_null<AttrElementT>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    return AttrElementT::get(
        a.getType(), calculate(a.getValue(), b.getValue(), c.getValue()));
  } else if (auto a = rawA.dyn_cast_or_null<SplatElementsAttr>()) {
    // TODO(benvanik): handle splat/otherwise.
    auto b = rawB.dyn_cast_or_null<SplatElementsAttr>();
    auto c = rawC.dyn_cast_or_null<SplatElementsAttr>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    auto elementResult = constFoldTernaryOp<AttrElementT>(
        a.getSplatValue<Attribute>(), b.getSplatValue<Attribute>(),
        c.getSplatValue<Attribute>(), calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(a.getType(), elementResult);
  } else if (auto a = rawA.dyn_cast_or_null<ElementsAttr>()) {
    auto b = rawB.dyn_cast_or_null<ElementsAttr>();
    auto c = rawC.dyn_cast_or_null<ElementsAttr>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    auto aIt = a.getValues<AttrElementT>().begin();
    auto bIt = b.getValues<AttrElementT>().begin();
    auto cIt = c.getValues<AttrElementT>().begin();
    SmallVector<Attribute, 4> resultAttrs(a.getNumElements());
    for (int64_t i = 0; i < a.getNumElements(); ++i) {
      resultAttrs[i] =
          constFoldTernaryOp<AttrElementT>(*aIt, *bIt, *cIt, calculate);
      if (!resultAttrs[i]) return {};
      ++aIt;
      ++bIt;
      ++cIt;
    }
    return DenseElementsAttr::get(a.getType(), resultAttrs);
  }
  return {};
}

// %0 = vm.mul.f32 %a, %b : f32
// %1 = vm.add.f32 %0, %c : f32
// ->
// %1 = vm.fma.f32 %a, %b, %c : f32
template <class MulOp, class AddOp, class FMAOp>
struct FuseFMAOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    auto fuse = [&](MulOp mulOp, Value a, Value b, Value c) {
      if (!mulOp->hasOneUse() ||
          mulOp->isUsedOutsideOfBlock(mulOp->getBlock())) {
        return failure();
      }
      rewriter.replaceOp(
          addOp,
          rewriter
              .create<FMAOp>(rewriter.getFusedLoc({a.getLoc(), c.getLoc()}),
                             a.getType(), a, b, c)
              .getResult());
      return success();
    };
    if (auto mulOp = dyn_cast_or_null<MulOp>(addOp.getLhs().getDefiningOp())) {
      return fuse(mulOp, mulOp.getLhs(), mulOp.getRhs(), addOp.getRhs());
    } else if (auto mulOp =
                   dyn_cast_or_null<MulOp>(addOp.getRhs().getDefiningOp())) {
      return fuse(mulOp, mulOp.getLhs(), mulOp.getRhs(), addOp.getLhs());
    }
    return failure();
  }
};

template <class AttrElementT, typename ADD, typename SUB,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldAddOp(ADD op, Attribute lhs, Attribute rhs) {
  if ((std::is_same<AttrElementT, IntegerAttr>::value &&
       matchPattern(op.getRhs(), m_Zero())) ||
      (std::is_same<AttrElementT, FloatAttr>::value &&
       matchPattern(op.getRhs(), m_AnyZeroFloat()))) {
    // x + 0 = x or 0 + y = y (commutative)
    return op.getLhs();
  }
  if (auto subOp = dyn_cast_or_null<SUB>(op.getLhs().getDefiningOp())) {
    if (subOp.getLhs() == op.getRhs()) return subOp.getRhs();
    if (subOp.getRhs() == op.getRhs()) return subOp.getLhs();
  } else if (auto subOp = dyn_cast_or_null<SUB>(op.getRhs().getDefiningOp())) {
    if (subOp.getLhs() == op.getLhs()) return subOp.getRhs();
    if (subOp.getRhs() == op.getLhs()) return subOp.getLhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      lhs, rhs,
      [](const ElementValueT &a, const ElementValueT &b) { return a + b; });
}

OpFoldResult AddI32Op::fold(FoldAdaptor operands) {
  return foldAddOp<IntegerAttr, AddI32Op, SubI32Op>(*this, operands.getLhs(),
                                                    operands.getRhs());
}

void AddI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FuseFMAOp<MulI32Op, AddI32Op, FMAI32Op>>(context);
}

OpFoldResult AddI64Op::fold(FoldAdaptor operands) {
  return foldAddOp<IntegerAttr, AddI64Op, SubI64Op>(*this, operands.getLhs(),
                                                    operands.getRhs());
}

void AddI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FuseFMAOp<MulI64Op, AddI64Op, FMAI64Op>>(context);
}

template <class AttrElementT, typename SUB, typename ADD,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldSubOp(SUB op, Attribute lhs, Attribute rhs) {
  if ((std::is_same<AttrElementT, IntegerAttr>::value &&
       matchPattern(op.getRhs(), m_Zero())) ||
      (std::is_same<AttrElementT, FloatAttr>::value &&
       matchPattern(op.getRhs(), m_AnyZeroFloat()))) {
    // x - 0 = x
    return op.getLhs();
  }
  if (auto addOp = dyn_cast_or_null<ADD>(op.getLhs().getDefiningOp())) {
    if (addOp.getLhs() == op.getRhs()) return addOp.getRhs();
    if (addOp.getRhs() == op.getRhs()) return addOp.getLhs();
  } else if (auto addOp = dyn_cast_or_null<ADD>(op.getRhs().getDefiningOp())) {
    if (addOp.getLhs() == op.getLhs()) return addOp.getRhs();
    if (addOp.getRhs() == op.getLhs()) return addOp.getLhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      lhs, rhs,
      [](const ElementValueT &a, const ElementValueT &b) { return a - b; });
}

OpFoldResult SubI32Op::fold(FoldAdaptor operands) {
  return foldSubOp<IntegerAttr, SubI32Op, AddI32Op>(*this, operands.getLhs(),
                                                    operands.getRhs());
}

OpFoldResult SubI64Op::fold(FoldAdaptor operands) {
  return foldSubOp<IntegerAttr, SubI64Op, AddI64Op>(*this, operands.getLhs(),
                                                    operands.getRhs());
}

template <class AttrElementT, typename T,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldMulOp(T op, Attribute lhs, Attribute rhs) {
  if ((std::is_same<AttrElementT, IntegerAttr>::value &&
       matchPattern(op.getRhs(), m_Zero())) ||
      (std::is_same<AttrElementT, FloatAttr>::value &&
       matchPattern(op.getRhs(), m_AnyZeroFloat()))) {
    // x * 0 = 0 or 0 * y = 0 (commutative)
    return zeroOfType(op.getType());
  } else if ((std::is_same<AttrElementT, IntegerAttr>::value &&
              matchPattern(op.getRhs(), m_One())) ||
             (std::is_same<AttrElementT, FloatAttr>::value &&
              matchPattern(op.getRhs(), m_OneFloat()))) {
    // x * 1 = x or 1 * y = y (commutative)
    return op.getLhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      lhs, rhs,
      [](const ElementValueT &a, const ElementValueT &b) { return a * b; });
}

template <class AttrElementT, typename T, typename CONST_OP,
          class ElementValueT = typename AttrElementT::ValueType>
struct FoldConstantMulOperand : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    AttrElementT c1, c2;
    if (!matchPattern(op.getRhs(), m_Constant(&c1))) return failure();
    if (auto mulOp = dyn_cast_or_null<T>(op.getLhs().getDefiningOp())) {
      if (matchPattern(mulOp.getRhs(), m_Constant(&c2))) {
        auto c = rewriter.createOrFold<CONST_OP>(
            rewriter.getFusedLoc({mulOp.getLoc(), op.getLoc()}),
            constFoldBinaryOp<AttrElementT>(
                c1, c2, [](const ElementValueT &a, const ElementValueT &b) {
                  return a * b;
                }));
        rewriter.replaceOpWithNewOp<T>(op, op.getType(), mulOp.getLhs(), c);
        return success();
      }
    }
    return failure();
  }
};

OpFoldResult MulI32Op::fold(FoldAdaptor operands) {
  return foldMulOp<IntegerAttr>(*this, operands.getLhs(), operands.getRhs());
}

void MulI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<IntegerAttr, MulI32Op, ConstI32Op>>(
      context);
}

OpFoldResult MulI64Op::fold(FoldAdaptor operands) {
  return foldMulOp<IntegerAttr>(*this, operands.getLhs(), operands.getRhs());
}

void MulI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<IntegerAttr, MulI64Op, ConstI64Op>>(
      context);
}

template <typename DivOpT, typename MulOpT>
static OpFoldResult foldDivSOp(DivOpT op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getRhs(), m_One())) {
    // x / 1 = x
    return op.getLhs();
  } else if (auto mulOp =
                 dyn_cast_or_null<MulOpT>(op.getLhs().getDefiningOp())) {
    // Only applies to signed divides (matches LLVM behavior).
    if (mulOp.getRhs() == op.getRhs()) {
      // c = mul a, b
      // d = div c, b
      // ->
      // d = a
      return mulOp.getLhs();
    }
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a.sdiv(b); });
}

OpFoldResult DivI32SOp::fold(FoldAdaptor operands) {
  return foldDivSOp<DivI32SOp, MulI32Op>(*this, operands.getLhs(),
                                         operands.getRhs());
}

OpFoldResult DivI64SOp::fold(FoldAdaptor operands) {
  return foldDivSOp<DivI64SOp, MulI64Op>(*this, operands.getLhs(),
                                         operands.getRhs());
}

template <typename DivOpT>
static OpFoldResult foldDivUOp(DivOpT op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getRhs(), m_One())) {
    // x / 1 = x
    return op.getLhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a.udiv(b); });
}

OpFoldResult DivI32UOp::fold(FoldAdaptor operands) {
  return foldDivUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult DivI64UOp::fold(FoldAdaptor operands) {
  return foldDivUOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldRemSOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_Zero()) ||
             matchPattern(op.getRhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a.srem(b); });
}

OpFoldResult RemI32SOp::fold(FoldAdaptor operands) {
  return foldRemSOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult RemI64SOp::fold(FoldAdaptor operands) {
  return foldRemSOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldRemUOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_Zero()) ||
             matchPattern(op.getRhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a.urem(b); });
}

OpFoldResult RemI32UOp::fold(FoldAdaptor operands) {
  return foldRemUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult RemI64UOp::fold(FoldAdaptor operands) {
  return foldRemUOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldFMAOp(T op, Attribute a, Attribute b, Attribute c) {
  // a * b + c
  if (matchPattern(op.getA(), m_Zero()) || matchPattern(op.getB(), m_Zero())) {
    return op.getC();
  }
  return constFoldTernaryOp<IntegerAttr>(
      a, b, c, [](const APInt &a, const APInt &b, const APInt &c) {
        return APInt(a.getBitWidth(),
                     a.getSExtValue() * b.getSExtValue() + c.getSExtValue());
      });
}

template <typename FMAOp, typename MulOp, typename AddOp>
struct CanonicalizeFMA final : public OpRewritePattern<FMAOp> {
  using OpRewritePattern<FMAOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FMAOp fmaOp,
                                PatternRewriter &rewriter) const override {
    // a * b + c
    if (matchPattern(fmaOp.getA(), m_One())) {
      // 1 * b + c = b + c
      rewriter.replaceOpWithNewOp<AddOp>(fmaOp, fmaOp.getType(), fmaOp.getB(),
                                         fmaOp.getC());
      return success();
    } else if (matchPattern(fmaOp.getB(), m_One())) {
      // a * 1 + c = a + c
      rewriter.replaceOpWithNewOp<AddOp>(fmaOp, fmaOp.getType(), fmaOp.getA(),
                                         fmaOp.getC());
      return success();
    } else if (matchPattern(fmaOp.getC(), m_Zero())) {
      // a * b + 0 = a * b
      rewriter.replaceOpWithNewOp<MulOp>(fmaOp, fmaOp.getType(), fmaOp.getA(),
                                         fmaOp.getB());
      return success();
    }
    return failure();
  }
};

OpFoldResult FMAI32Op::fold(FoldAdaptor operands) {
  return foldFMAOp(*this, operands.getA(), operands.getB(), operands.getC());
}

OpFoldResult FMAI64Op::fold(FoldAdaptor operands) {
  return foldFMAOp(*this, operands.getA(), operands.getB(), operands.getC());
}

void FMAI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAI32Op, MulI32Op, AddI32Op>>(context);
}

void FMAI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAI64Op, MulI64Op, AddI64Op>>(context);
}

OpFoldResult AbsI32Op::fold(FoldAdaptor operands) {
  return constFoldUnaryOp<IntegerAttr>(operands.getOperand(),
                                       [](const APInt &a) { return a.abs(); });
}

OpFoldResult AbsI64Op::fold(FoldAdaptor operands) {
  return constFoldUnaryOp<IntegerAttr>(operands.getOperand(),
                                       [](const APInt &a) { return a.abs(); });
}

OpFoldResult MinI32SOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::smin(lhs, rhs);
                                        });
}

OpFoldResult MinI64SOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::smin(lhs, rhs);
                                        });
}

OpFoldResult MinI32UOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::umin(lhs, rhs);
                                        });
}

OpFoldResult MinI64UOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::umin(lhs, rhs);
                                        });
}

OpFoldResult MaxI32SOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::smax(lhs, rhs);
                                        });
}

OpFoldResult MaxI64SOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::smax(lhs, rhs);
                                        });
}

OpFoldResult MaxI32UOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::umax(lhs, rhs);
                                        });
}

OpFoldResult MaxI64UOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(operands.getLhs(), operands.getRhs(),
                                        [](const APInt &lhs, const APInt &rhs) {
                                          return llvm::APIntOps::umax(lhs, rhs);
                                        });
}

//===----------------------------------------------------------------------===//
// Floating-point arithmetic
//===----------------------------------------------------------------------===//

OpFoldResult AddF32Op::fold(FoldAdaptor operands) {
  return foldAddOp<FloatAttr, AddF32Op, SubF32Op>(*this, operands.getLhs(),
                                                  operands.getRhs());
}

void AddF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<FloatAttr, MulF32Op, ConstF32Op>>(
      context);
  results.insert<FuseFMAOp<MulF32Op, AddF32Op, FMAF32Op>>(context);
}

OpFoldResult AddF64Op::fold(FoldAdaptor operands) {
  return foldAddOp<FloatAttr, AddF64Op, SubF64Op>(*this, operands.getLhs(),
                                                  operands.getRhs());
}

void AddF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FuseFMAOp<MulF64Op, AddF64Op, FMAF64Op>>(context);
}

OpFoldResult SubF32Op::fold(FoldAdaptor operands) {
  return foldSubOp<FloatAttr, SubF32Op, AddF32Op>(*this, operands.getLhs(),
                                                  operands.getRhs());
}

OpFoldResult SubF64Op::fold(FoldAdaptor operands) {
  return foldSubOp<FloatAttr, SubF64Op, AddF64Op>(*this, operands.getLhs(),
                                                  operands.getRhs());
}

OpFoldResult MulF32Op::fold(FoldAdaptor operands) {
  return foldMulOp<FloatAttr>(*this, operands.getLhs(), operands.getRhs());
}

void MulF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<FloatAttr, MulF32Op, ConstF32Op>>(
      context);
}

OpFoldResult MulF64Op::fold(FoldAdaptor operands) {
  return foldMulOp<FloatAttr>(*this, operands.getLhs(), operands.getRhs());
}

void MulF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<FloatAttr, MulF64Op, ConstF64Op>>(
      context);
}

template <typename T>
static OpFoldResult foldDivFOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getRhs(), m_OneFloat())) {
    // x / 1 = x
    return op.getLhs();
  }
  return constFoldBinaryOp<FloatAttr>(
      lhs, rhs, [](const APFloat &a, const APFloat &b) {
        APFloat c = a;
        c.divide(b, APFloat::rmNearestTiesToAway);
        return c;
      });
}

OpFoldResult DivF32Op::fold(FoldAdaptor operands) {
  return foldDivFOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult DivF64Op::fold(FoldAdaptor operands) {
  return foldDivFOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldRemFOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
             matchPattern(op.getRhs(), m_OneFloat())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(lhs, rhs,
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat c = a;
                                        c.remainder(b);
                                        return c;
                                      });
}

OpFoldResult RemF32Op::fold(FoldAdaptor operands) {
  return foldRemFOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult RemF64Op::fold(FoldAdaptor operands) {
  return foldRemFOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldFMAFOp(T op, Attribute a, Attribute b, Attribute c) {
  // a * b + c
  if (matchPattern(op.getA(), m_AnyZeroFloat()) ||
      matchPattern(op.getB(), m_AnyZeroFloat())) {
    return op.getC();
  }
  return constFoldTernaryOp<FloatAttr>(
      a, b, c, [](const APFloat &a, const APFloat &b, const APFloat &c) {
        APFloat d = a;
        d.fusedMultiplyAdd(b, c, APFloat::rmNearestTiesToAway);
        return d;
      });
}

OpFoldResult FMAF32Op::fold(FoldAdaptor operands) {
  return foldFMAFOp(*this, operands.getA(), operands.getB(), operands.getC());
}

OpFoldResult FMAF64Op::fold(FoldAdaptor operands) {
  return foldFMAFOp(*this, operands.getA(), operands.getB(), operands.getC());
}

void FMAF32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAF32Op, MulF32Op, AddF32Op>>(context);
}

void FMAF64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAF64Op, MulF64Op, AddF64Op>>(context);
}

OpFoldResult AbsF32Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.clearSign();
    return b;
  });
}

OpFoldResult AbsF64Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.clearSign();
    return b;
  });
}

OpFoldResult NegF32Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.changeSign();
    return b;
  });
}

OpFoldResult NegF64Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.changeSign();
    return b;
  });
}

OpFoldResult CeilF32Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardPositive);
    return b;
  });
}

OpFoldResult CeilF64Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardPositive);
    return b;
  });
}

OpFoldResult FloorF32Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardNegative);
    return b;
  });
}

OpFoldResult FloorF64Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardNegative);
    return b;
  });
}

OpFoldResult MinF32Op::fold(FoldAdaptor operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands.getLhs(), operands.getRhs(),
      [](const APFloat &a, const APFloat &b) { return llvm::minnum(a, b); });
}

OpFoldResult MinF64Op::fold(FoldAdaptor operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands.getLhs(), operands.getRhs(),
      [](const APFloat &a, const APFloat &b) { return llvm::minnum(a, b); });
}

OpFoldResult MaxF32Op::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<FloatAttr>(
      operands.getLhs(), operands.getRhs(),
      [](const APFloat &a, const APFloat &b) { return llvm::maxnum(a, b); });
}

OpFoldResult MaxF64Op::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) return getLhs();
  return constFoldBinaryOp<FloatAttr>(
      operands.getLhs(), operands.getRhs(),
      [](const APFloat &a, const APFloat &b) { return llvm::maxnum(a, b); });
}

//===----------------------------------------------------------------------===//
// Floating-point math
//===----------------------------------------------------------------------===//

OpFoldResult SqrtF32Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    return APFloat(sqrtf(a.convertToFloat()));
  });
}

OpFoldResult SqrtF64Op::fold(FoldAdaptor operands) {
  return constFoldFloatUnaryOp(operands.getOperand(), [](const APFloat &a) {
    return APFloat(sqrt(a.convertToDouble()));
  });
}

//===----------------------------------------------------------------------===//
// Integer bit manipulation
//===----------------------------------------------------------------------===//

template <typename T>
static OpFoldResult foldNotOp(T op, Attribute operand) {
  return constFoldUnaryOp<IntegerAttr>(operand, [](APInt a) {
    a.flipAllBits();
    return a;
  });
}

OpFoldResult NotI32Op::fold(FoldAdaptor operands) {
  return foldNotOp(*this, operands.getOperand());
}

OpFoldResult NotI64Op::fold(FoldAdaptor operands) {
  return foldNotOp(*this, operands.getOperand());
}

template <typename T>
static OpFoldResult foldAndOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x & 0 = 0 or 0 & y = 0 (commutative)
    return zeroOfType(op.getType());
  } else if (op.getLhs() == op.getRhs()) {
    // x & x = x
    return op.getLhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a & b; });
}

OpFoldResult AndI32Op::fold(FoldAdaptor operands) {
  return foldAndOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult AndI64Op::fold(FoldAdaptor operands) {
  return foldAndOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldOrOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x | 0 = x or 0 | y = y (commutative)
    return op.getLhs();
  } else if (op.getLhs() == op.getRhs()) {
    // x | x = x
    return op.getLhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a | b; });
}

OpFoldResult OrI32Op::fold(FoldAdaptor operands) {
  return foldOrOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult OrI64Op::fold(FoldAdaptor operands) {
  return foldOrOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldXorOp(T op, Attribute lhs, Attribute rhs) {
  if (matchPattern(op.getRhs(), m_Zero())) {
    // x ^ 0 = x or 0 ^ y = y (commutative)
    return op.getLhs();
  } else if (op.getLhs() == op.getRhs()) {
    // x ^ x = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      lhs, rhs, [](const APInt &a, const APInt &b) { return a ^ b; });
}

OpFoldResult XorI32Op::fold(FoldAdaptor operands) {
  return foldXorOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult XorI64Op::fold(FoldAdaptor operands) {
  return foldXorOp(*this, operands.getLhs(), operands.getRhs());
}

template <typename T>
static OpFoldResult foldCtlzOp(T op, Attribute operand) {
  return constFoldUnaryOp<IntegerAttr>(operand, [](APInt a) {
    return APInt(a.getBitWidth(), a.countLeadingZeros());
  });
}

OpFoldResult CtlzI32Op::fold(FoldAdaptor operands) {
  return foldCtlzOp(*this, operands.getOperand());
}

OpFoldResult CtlzI64Op::fold(FoldAdaptor operands) {
  return foldCtlzOp(*this, operands.getOperand());
}

//===----------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===----------------------------------------------------------------------===//

template <typename T>
static OpFoldResult foldShlOp(T op, Attribute operand, Attribute amount) {
  if (matchPattern(op.getOperand(), m_Zero())) {
    // 0 << y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getAmount(), m_Zero())) {
    // x << 0 = x
    return op.getOperand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operand, amount,
      [&](const APInt &a, const APInt &b) { return a.shl(b); });
}

OpFoldResult ShlI32Op::fold(FoldAdaptor operands) {
  return foldShlOp(*this, operands.getOperand(), operands.getAmount());
}

OpFoldResult ShlI64Op::fold(FoldAdaptor operands) {
  return foldShlOp(*this, operands.getOperand(), operands.getAmount());
}

template <typename T>
static OpFoldResult foldShrSOp(T op, Attribute operand, Attribute amount) {
  if (matchPattern(op.getOperand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getAmount(), m_Zero())) {
    // x >> 0 = x
    return op.getOperand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operand, amount,
      [&](const APInt &a, const APInt &b) { return a.ashr(b); });
}

OpFoldResult ShrI32SOp::fold(FoldAdaptor operands) {
  return foldShrSOp(*this, operands.getOperand(), operands.getAmount());
}

OpFoldResult ShrI64SOp::fold(FoldAdaptor operands) {
  return foldShrSOp(*this, operands.getOperand(), operands.getAmount());
}

template <typename T>
static OpFoldResult foldShrUOp(T op, Attribute operand, Attribute amount) {
  if (matchPattern(op.getOperand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.getAmount(), m_Zero())) {
    // x >> 0 = x
    return op.getOperand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operand, amount,
      [&](const APInt &a, const APInt &b) { return a.lshr(b); });
}

OpFoldResult ShrI32UOp::fold(FoldAdaptor operands) {
  return foldShrUOp(*this, operands.getOperand(), operands.getAmount());
}

OpFoldResult ShrI64UOp::fold(FoldAdaptor operands) {
  return foldShrUOp(*this, operands.getOperand(), operands.getAmount());
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<ElementValueT(ElementValueT)>>
static Attribute constFoldConversionOp(Type resultType, Attribute rawOperand,
                                       const CalculationT &calculate) {
  if (auto operand = rawOperand.dyn_cast_or_null<AttrElementT>()) {
    return AttrElementT::get(resultType, calculate(operand.getValue()));
  }
  return {};
}

OpFoldResult TruncI32I8Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI32I16Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I8Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI64I16Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I32Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(32); });
}

OpFoldResult TruncF64F32Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<FloatAttr>(
      FloatType::getF32(getContext()), operands.getOperand(),
      [&](const APFloat &a) { return APFloat(a.convertToFloat()); });
}

OpFoldResult ExtI8I32SOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).sext(32); });
}

OpFoldResult ExtI8I32UOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult ExtI16I32SOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).sext(32); });
}

OpFoldResult ExtI16I32UOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult ExtI8I64SOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).sext(64); });
}

OpFoldResult ExtI8I64UOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(8).zext(64); });
}

OpFoldResult ExtI16I64SOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).sext(64); });
}

OpFoldResult ExtI16I64UOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.trunc(16).zext(64); });
}

OpFoldResult ExtI32I64SOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.sext(64); });
}

OpFoldResult ExtI32I64UOp::fold(FoldAdaptor operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands.getOperand(),
      [&](const APInt &a) { return a.zext(64); });
}

OpFoldResult ExtF32F64Op::fold(FoldAdaptor operands) {
  return constFoldConversionOp<FloatAttr>(
      FloatType::getF64(getContext()), operands.getOperand(),
      [&](const APFloat &a) { return APFloat(a.convertToDouble()); });
}

namespace {

template <typename SRC_OP, typename OP_A, int SZ_T, typename OP_B>
struct PseudoIntegerConversionToSplitConversionOp
    : public OpRewritePattern<SRC_OP> {
  using OpRewritePattern<SRC_OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(SRC_OP op,
                                PatternRewriter &rewriter) const override {
    auto tmp = rewriter.createOrFold<OP_A>(
        op.getLoc(), rewriter.getIntegerType(SZ_T), op.getOperand());
    rewriter.replaceOpWithNewOp<OP_B>(op, op.getResult().getType(), tmp);
    return success();
  }
};

}  // namespace

void TruncI64I8Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      TruncI64I8Op, TruncI64I32Op, 32, TruncI32I8Op>>(context);
}

void TruncI64I16Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      TruncI64I16Op, TruncI64I32Op, 32, TruncI32I16Op>>(context);
}

void ExtI8I64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI8I64SOp, ExtI8I32SOp, 32, ExtI32I64SOp>>(context);
}

void ExtI8I64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI8I64UOp, ExtI8I32UOp, 32, ExtI32I64UOp>>(context);
}

void ExtI16I64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI16I64SOp, ExtI16I32SOp, 32, ExtI32I64SOp>>(context);
}

void ExtI16I64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI16I64UOp, ExtI16I32UOp, 32, ExtI32I64UOp>>(context);
}

template <
    class SrcAttrElementT, class DstAttrElementT,
    class SrcElementValueT = typename SrcAttrElementT::ValueType,
    class DstElementValueT = typename DstAttrElementT::ValueType,
    class CalculationT = std::function<DstElementValueT(SrcElementValueT)>>
static Attribute constFoldCastOp(Type resultType, Attribute rawOperand,
                                 const CalculationT &calculate) {
  if (auto operand = rawOperand.dyn_cast_or_null<SrcAttrElementT>()) {
    return DstAttrElementT::get(resultType, calculate(operand.getValue()));
  }
  return {};
}

OpFoldResult CastSI32F32Op::fold(FoldAdaptor operands) {
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      Float32Type::get(getContext()), operands.getOperand(),
      [&](const APInt &a) {
        APFloat b = APFloat(0.0f);
        b.convertFromAPInt(a, /*IsSigned=*/true, APFloat::rmNearestTiesToAway);
        return b;
      });
}

OpFoldResult CastUI32F32Op::fold(FoldAdaptor operands) {
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      Float32Type::get(getContext()), operands.getOperand(),
      [&](const APInt &a) {
        APFloat b = APFloat(0.0f);
        b.convertFromAPInt(a, /*IsSigned=*/false, APFloat::rmNearestTiesToAway);
        return b;
      });
}

OpFoldResult CastF32SI32Op::fold(FoldAdaptor operands) {
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APFloat &a) {
        bool isExact = false;
        llvm::APSInt b(/*BitWidth=*/32, /*isUnsigned=*/false);
        a.convertToInteger(b, APFloat::rmNearestTiesToAway, &isExact);
        return b;
      });
}

OpFoldResult CastF32UI32Op::fold(FoldAdaptor operands) {
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      IntegerType::get(getContext(), 32), operands.getOperand(),
      [&](const APFloat &a) {
        bool isExact = false;
        llvm::APSInt b(/*BitWidth=*/32, /*isUnsigned=*/false);
        a.convertToInteger(b, APFloat::rmNearestTiesToAway, &isExact);
        b.setIsUnsigned(true);
        return b;
      });
}

//===----------------------------------------------------------------------===//
// Native reduction (horizontal) arithmetic
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

namespace {

/// Performs const folding `calculate` on the given `operands` and returns the
/// result as an i32 bool if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<APInt(ElementValueT)>>
static Attribute constFoldUnaryCmpOp(Attribute rawOperand,
                                     const CalculationT &calculate) {
  if (auto operand = rawOperand.dyn_cast_or_null<AttrElementT>()) {
    auto boolType = IntegerType::get(operand.getContext(), 32);
    return IntegerAttr::get(boolType, calculate(operand.getValue()));
  } else if (auto operand = rawOperand.dyn_cast_or_null<ElementsAttr>()) {
    auto boolType = IntegerType::get(operand.getContext(), 32);
    return operand.cast<DenseIntOrFPElementsAttr>().mapValues(
        boolType,
        llvm::function_ref<APInt(const ElementValueT &)>(
            [&](const ElementValueT &value) { return calculate(value); }));
  }
  return {};
}

/// Performs const folding `calculate` with on the two given operand attributes
/// in `operands` and returns the result if possible as an I32 bool.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldBinaryCmpOp(Attribute rawLhs, Attribute rawRhs,
                                      const CalculationT &calculate) {
  if (auto lhs = rawLhs.dyn_cast_or_null<AttrElementT>()) {
    auto rhs = rawRhs.dyn_cast_or_null<AttrElementT>();
    if (!rhs) return {};
    auto boolType = IntegerType::get(lhs.getContext(), 32);
    return AttrElementT::get(boolType,
                             calculate(lhs.getValue(), rhs.getValue()));
  }
  return {};
}

/// Swaps the cmp op with its inverse if the result is inverted.
template <typename OP, typename INV>
struct SwapInvertedCmpOps : public OpRewritePattern<OP> {
  using OpRewritePattern<OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    // We generate xor(cmp(...), 1) to flip conditions, so look for that pattern
    // so that we can do the swap here and remove the xor.
    if (!op.getResult().hasOneUse()) {
      // Can't change if there are multiple users.
      return failure();
    }
    if (auto xorOp = dyn_cast_or_null<XorI32Op>(*op.getResult().user_begin())) {
      Attribute rhs;
      if (xorOp.getLhs() == op.getResult() &&
          matchPattern(xorOp.getRhs(), m_Constant(&rhs)) &&
          rhs.cast<IntegerAttr>().getInt() == 1) {
        auto invValue = rewriter.createOrFold<INV>(
            op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs());
        rewriter.replaceOp(op, {invValue});
        rewriter.replaceOp(xorOp, {invValue});
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpEQOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x == x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.eq(b); });
}

OpFoldResult CmpEQI32Op::fold(FoldAdaptor operands) {
  return foldCmpEQOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpEQI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQI32Op, CmpNEI32Op>>(context);
}

OpFoldResult CmpEQI64Op::fold(FoldAdaptor operands) {
  return foldCmpEQOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpEQI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQI64Op, CmpNEI64Op>>(context);
}

template <typename T>
static OpFoldResult foldCmpNEOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x != x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.ne(b); });
}

OpFoldResult CmpNEI32Op::fold(FoldAdaptor operands) {
  return foldCmpNEOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpNEI64Op::fold(FoldAdaptor operands) {
  return foldCmpNEOp(*this, operands.getLhs(), operands.getRhs());
}

namespace {

/// Changes a cmp.ne.i32 check against 0 to a cmp.nz.i32.
template <typename NE_OP, typename NZ_OP>
struct CmpNEZeroToCmpNZ : public OpRewritePattern<NE_OP> {
  using OpRewritePattern<NE_OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(NE_OP op,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(op.getRhs(), m_Zero())) {
      rewriter.replaceOpWithNewOp<NZ_OP>(op, op.getType(), op.getLhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpNEI32Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEI32Op, CmpEQI32Op>,
                 CmpNEZeroToCmpNZ<CmpNEI32Op, CmpNZI32Op>>(context);
}

void CmpNEI64Op::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEI64Op, CmpEQI64Op>,
                 CmpNEZeroToCmpNZ<CmpNEI64Op, CmpNZI64Op>>(context);
}

template <typename T>
static OpFoldResult foldCmpLTSOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.slt(b); });
}

OpFoldResult CmpLTI32SOp::fold(FoldAdaptor operands) {
  return foldCmpLTSOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTI64SOp::fold(FoldAdaptor operands) {
  return foldCmpLTSOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpLTI32SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void CmpLTI64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

template <typename T>
static OpFoldResult foldCmpLTUOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.ult(b); });
}

OpFoldResult CmpLTI32UOp::fold(FoldAdaptor operands) {
  return foldCmpLTUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTI64UOp::fold(FoldAdaptor operands) {
  return foldCmpLTUOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpLTI32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void CmpLTI64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

namespace {

/// Rewrites a vm.cmp.lte.* pseudo op to a vm.cmp.lt.* op.
template <typename T, typename U>
struct RewritePseudoCmpLTEToLT : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // !(lhs > rhs)
    auto condValue = rewriter.createOrFold<U>(op.getLoc(), op.getType(),
                                              op.getRhs(), op.getLhs());
    rewriter.replaceOpWithNewOp<XorI32Op>(
        op, op.getType(), condValue,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpLTESOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.sle(b); });
}

OpFoldResult CmpLTEI32SOp::fold(FoldAdaptor operands) {
  return foldCmpLTESOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTEI64SOp::fold(FoldAdaptor operands) {
  return foldCmpLTESOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpLTEI32SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32SOp, CmpGTI32SOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI32SOp, CmpLTI32SOp>>(context);
}

void CmpLTEI64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI64SOp, CmpGTI64SOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpLTEUOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.ule(b); });
}

OpFoldResult CmpLTEI32UOp::fold(FoldAdaptor operands) {
  return foldCmpLTEUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTEI64UOp::fold(FoldAdaptor operands) {
  return foldCmpLTEUOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpLTEI32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32UOp, CmpGTI32UOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI32UOp, CmpLTI32UOp>>(context);
}

void CmpLTEI64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI64UOp, CmpGTI64UOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI64UOp, CmpLTI64UOp>>(context);
}

namespace {

/// Rewrites a vm.cmp.gt.* pseudo op to a vm.cmp.lt.* op.
template <typename T, typename U>
struct RewritePseudoCmpGTToLT : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // rhs < lhs
    rewriter.replaceOpWithNewOp<U>(op, op.getType(), op.getRhs(), op.getLhs());
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpGTSOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.sgt(b); });
}

OpFoldResult CmpGTI32SOp::fold(FoldAdaptor operands) {
  return foldCmpGTSOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTI64SOp::fold(FoldAdaptor operands) {
  return foldCmpGTSOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTI32SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32SOp, CmpLTEI32SOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI32SOp, CmpLTI32SOp>>(context);
}

void CmpGTI64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI64SOp, CmpLTEI64SOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpGTUOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.ugt(b); });
}

OpFoldResult CmpGTI32UOp::fold(FoldAdaptor operands) {
  return foldCmpGTUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTI64UOp::fold(FoldAdaptor operands) {
  return foldCmpGTUOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTI32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32UOp, CmpLTEI32UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI32UOp, CmpLTI32UOp>>(context);
}

void CmpGTI64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI64UOp, CmpLTEI64UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI64UOp, CmpLTI64UOp>>(context);
}

namespace {

/// Rewrites a vm.cmp.gte.* pseudo op to a vm.cmp.lt.* op.
template <typename T, typename U>
struct RewritePseudoCmpGTEToLT : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // !(lhs < rhs)
    auto condValue = rewriter.createOrFold<U>(op.getLoc(), op.getType(),
                                              op.getLhs(), op.getRhs());
    rewriter.replaceOpWithNewOp<XorI32Op>(
        op, op.getType(), condValue,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpGTESOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.sge(b); });
}

OpFoldResult CmpGTEI32SOp::fold(FoldAdaptor operands) {
  return foldCmpGTESOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTEI64SOp::fold(FoldAdaptor operands) {
  return foldCmpGTESOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTEI32SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32SOp, CmpLTI32SOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI32SOp, CmpLTI32SOp>>(context);
}

void CmpGTEI64SOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI64SOp, CmpLTI64SOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpGTEUOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpOp<IntegerAttr>(
      lhs, rhs, [&](const APInt &a, const APInt &b) { return a.uge(b); });
}

OpFoldResult CmpGTEI32UOp::fold(FoldAdaptor operands) {
  return foldCmpGTEUOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTEI64UOp::fold(FoldAdaptor operands) {
  return foldCmpGTEUOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTEI32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32UOp, CmpLTI32UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI32UOp, CmpLTI32UOp>>(context);
}

void CmpGTEI64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI64UOp, CmpLTI64UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI64UOp, CmpLTI64UOp>>(context);
}

OpFoldResult CmpNZI32Op::fold(FoldAdaptor operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands.getOperand(),
      [&](const APInt &a) { return APInt(32, a.getBoolValue()); });
}

OpFoldResult CmpNZI64Op::fold(FoldAdaptor operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands.getOperand(),
      [&](const APInt &a) { return APInt(64, a.getBoolValue()); });
}

//===----------------------------------------------------------------------===//
// Floating-point comparison
//===----------------------------------------------------------------------===//

/// Performs const folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the integer result of the comparison
/// if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldBinaryCmpFOp(Attribute rawLhs, Attribute rawRhs,
                                       const CalculationT &calculate) {
  if (auto lhs = rawLhs.dyn_cast_or_null<AttrElementT>()) {
    auto rhs = rawRhs.dyn_cast_or_null<AttrElementT>();
    if (!rhs) return {};
    return IntegerAttr::get(IntegerType::get(lhs.getContext(), 32),
                            calculate(lhs.getValue(), rhs.getValue()));
  } else if (auto lhs = rawLhs.dyn_cast_or_null<SplatElementsAttr>()) {
    // TODO(benvanik): handle splat/otherwise.
    auto rhs = rawRhs.dyn_cast_or_null<SplatElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto elementResult = constFoldBinaryCmpFOp<AttrElementT>(
        lhs.getSplatValue<Attribute>(), rhs.getSplatValue<Attribute>(),
        calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(IntegerType::get(lhs.getContext(), 32),
                                  elementResult);
  } else if (auto lhs = rawLhs.dyn_cast_or_null<ElementsAttr>()) {
    auto rhs = rawRhs.dyn_cast_or_null<ElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto lhsIt = lhs.getValues<AttrElementT>().begin();
    auto rhsIt = rhs.getValues<AttrElementT>().begin();
    SmallVector<Attribute, 4> resultAttrs(lhs.getNumElements());
    for (int64_t i = 0; i < lhs.getNumElements(); ++i) {
      resultAttrs[i] =
          constFoldBinaryCmpFOp<AttrElementT>(*lhsIt, *rhsIt, calculate);
      if (!resultAttrs[i]) return {};
      ++lhsIt;
      ++rhsIt;
    }
    return DenseElementsAttr::get(IntegerType::get(lhs.getContext(), 32),
                                  resultAttrs);
  }
  return {};
}

enum CmpFOrdering {
  ORDERED = 0,
  UNORDERED = 1,
};

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpEQFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x == x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpEqual;
        } else {
          return result == APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpEQF32OOp::fold(FoldAdaptor operands) {
  return foldCmpEQFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpEQF64OOp::fold(FoldAdaptor operands) {
  return foldCmpEQFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpEQF32UOp::fold(FoldAdaptor operands) {
  return foldCmpEQFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpEQF64UOp::fold(FoldAdaptor operands) {
  return foldCmpEQFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

void CmpEQF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF32OOp, CmpNEF32OOp>>(context);
}

void CmpEQF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF64OOp, CmpNEF64OOp>>(context);
}

void CmpEQF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF32UOp, CmpNEF32UOp>>(context);
}

void CmpEQF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF64UOp, CmpNEF64UOp>>(context);
}

namespace {

/// Rewrites a vm.cmp.f*.near pseudo op to a ULP-based comparison.
template <typename T, typename ConstFOp, typename ConstIOp, typename CmpGTEFOp,
          typename CmpEQFOp, typename CmpLTIOp, typename BitcastFToIOp,
          typename SubIOp, typename AbsIOp>
struct RewritePseudoCmpNear : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // Units in the Last Place (ULP) comparison algorithm from this reference:
    // https://www.gamedeveloper.com/programming/in-depth-comparing-floating-point-numbers-2012-edition
    // See also the C++ implementation in the constant folder below.

    auto loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();

    auto *originalBlock = rewriter.getInsertionBlock();
    auto *continuationBlock = rewriter.splitBlock(
        originalBlock, op.getOperation()->getNextNode()->getIterator());
    auto comparisonResult = continuationBlock->addArgument(i32Type, loc);

    // Compute `sign(lhs) != sign(rhs)` with `(lhs > 0) != (rhs > 0)`.
    // TODO(scotttodd): replace with pseudo op for vm.sign.f32
    //     * extract high bit: bitcastf32i32(lhs) >> 31
    //     * xor high bits of lhs and rhs
    auto zero = rewriter.createOrFold<ConstFOp>(loc, 0);
    auto lhsPositive =
        rewriter.createOrFold<CmpGTEFOp>(loc, i32Type, op.getLhs(), zero);
    auto rhsPositive =
        rewriter.createOrFold<CmpGTEFOp>(loc, i32Type, op.getRhs(), zero);
    auto signsNotEqual = rewriter.createOrFold<IREE::VM::CmpNEI32Op>(
        loc, i32Type, lhsPositive, rhsPositive);

    // If signs differ, perform a direct comparison of `lhs == rhs`.
    auto *directComparisonBlock = rewriter.createBlock(continuationBlock);
    auto exactEqual =
        rewriter.createOrFold<CmpEQFOp>(loc, i32Type, op.getLhs(), op.getRhs());
    rewriter.createOrFold<IREE::VM::BranchOp>(loc, continuationBlock,
                                              exactEqual);

    // ...else, perform a full ULP-based comparison.
    auto *ulpComparisonBlock = rewriter.createBlock(continuationBlock);
    auto lhsInt =
        rewriter.createOrFold<BitcastFToIOp>(loc, i32Type, op.getLhs());
    auto rhsInt =
        rewriter.createOrFold<BitcastFToIOp>(loc, i32Type, op.getRhs());
    auto signedUlpsDiff =
        rewriter.createOrFold<SubIOp>(loc, i32Type, lhsInt, rhsInt);
    auto absUlpsDiff =
        rewriter.createOrFold<AbsIOp>(loc, i32Type, signedUlpsDiff);
    // The constant chosen here is arbitrary. Higher values increase the
    // distance between arguments that is tolerated.
    auto maxUlpsDiff = rewriter.createOrFold<ConstIOp>(loc, 100);
    auto ulpCompare =
        rewriter.createOrFold<CmpLTIOp>(loc, i32Type, absUlpsDiff, maxUlpsDiff);
    rewriter.createOrFold<IREE::VM::BranchOp>(loc, continuationBlock,
                                              ulpCompare);

    // Go back up and insert the branch between comparison cases.
    rewriter.setInsertionPointAfter(signsNotEqual.getDefiningOp());
    rewriter.createOrFold<IREE::VM::CondBranchOp>(
        loc, signsNotEqual, directComparisonBlock, ulpComparisonBlock);

    rewriter.replaceOp(op, {comparisonResult});
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpEQNearOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x ~ x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        // See the corresponding rewrite pattern above for references used here.
        if (a.isNegative() != b.isNegative()) {
          return a.compare(b) == APFloat::cmpEqual;
        } else {
          auto lhsInt = a.bitcastToAPInt();
          auto rhsInt = b.bitcastToAPInt();
          auto signedUlpsDiff = lhsInt - rhsInt;
          auto absUlpsDiff = signedUlpsDiff.abs();
          return absUlpsDiff.slt(100);
        }
      });
}

OpFoldResult CmpEQF32NearOp::fold(FoldAdaptor operands) {
  return foldCmpEQNearOp(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpEQF64NearOp::fold(FoldAdaptor operands) {
  return foldCmpEQNearOp(*this, operands.getLhs(), operands.getRhs());
}

void CmpEQF32NearOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<RewritePseudoCmpNear<CmpEQF32NearOp, ConstF32Op, ConstI32Op,
                                      CmpGTEF32OOp, CmpEQF32OOp, CmpLTI32SOp,
                                      BitcastF32I32Op, SubI32Op, AbsI32Op>>(
      context);
}

void CmpEQF64NearOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<RewritePseudoCmpNear<CmpEQF64NearOp, ConstF64Op, ConstI64Op,
                                      CmpGTEF64OOp, CmpEQF64OOp, CmpLTI64SOp,
                                      BitcastF64I64Op, SubI64Op, AbsI64Op>>(
      context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpNEFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x != x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result != APFloat::cmpEqual;
        } else {
          return result != APFloat::cmpEqual && result != APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpNEF32OOp::fold(FoldAdaptor operands) {
  return foldCmpNEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpNEF64OOp::fold(FoldAdaptor operands) {
  return foldCmpNEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpNEF32UOp::fold(FoldAdaptor operands) {
  return foldCmpNEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpNEF64UOp::fold(FoldAdaptor operands) {
  return foldCmpNEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

void CmpNEF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF32OOp, CmpEQF32OOp>>(context);
}

void CmpNEF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF64OOp, CmpEQF64OOp>>(context);
}

void CmpNEF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF32UOp, CmpEQF32UOp>>(context);
}

void CmpNEF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF64UOp, CmpEQF64UOp>>(context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpLTFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpLessThan;
        } else {
          return result == APFloat::cmpLessThan ||
                 result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpLTF32OOp::fold(FoldAdaptor operands) {
  return foldCmpLTFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTF64OOp::fold(FoldAdaptor operands) {
  return foldCmpLTFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTF32UOp::fold(FoldAdaptor operands) {
  return foldCmpLTFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTF64UOp::fold(FoldAdaptor operands) {
  return foldCmpLTFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

void CmpLTF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void CmpLTF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void CmpLTF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

void CmpLTF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpLTEFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpLessThan || result == APFloat::cmpEqual;
        } else {
          return result == APFloat::cmpLessThan ||
                 result == APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpLTEF32OOp::fold(FoldAdaptor operands) {
  return foldCmpLTEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTEF64OOp::fold(FoldAdaptor operands) {
  return foldCmpLTEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTEF32UOp::fold(FoldAdaptor operands) {
  return foldCmpLTEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpLTEF64UOp::fold(FoldAdaptor operands) {
  return foldCmpLTEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpGTFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpGreaterThan;
        } else {
          return result == APFloat::cmpGreaterThan ||
                 result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpGTF32OOp::fold(FoldAdaptor operands) {
  return foldCmpGTFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTF64OOp::fold(FoldAdaptor operands) {
  return foldCmpGTFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTF32UOp::fold(FoldAdaptor operands) {
  return foldCmpGTFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTF64UOp::fold(FoldAdaptor operands) {
  return foldCmpGTFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF32OOp, CmpLTEF32OOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF32OOp, CmpLTF32OOp>>(context);
}

void CmpGTF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF64OOp, CmpLTEF64OOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF64OOp, CmpLTF64OOp>>(context);
}

void CmpGTF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF32UOp, CmpLTEF32UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF32UOp, CmpLTF32UOp>>(context);
}

void CmpGTF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF64UOp, CmpLTEF64UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF64UOp, CmpLTF64UOp>>(context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpGTEFOp(T op, Attribute lhs, Attribute rhs) {
  if (op.getLhs() == op.getRhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryCmpFOp<FloatAttr>(
      lhs, rhs, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpGreaterThan ||
                 result == APFloat::cmpEqual;
        } else {
          return result == APFloat::cmpGreaterThan ||
                 result == APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpGTEF32OOp::fold(FoldAdaptor operands) {
  return foldCmpGTEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTEF64OOp::fold(FoldAdaptor operands) {
  return foldCmpGTEFOp<ORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTEF32UOp::fold(FoldAdaptor operands) {
  return foldCmpGTEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

OpFoldResult CmpGTEF64UOp::fold(FoldAdaptor operands) {
  return foldCmpGTEFOp<UNORDERED>(*this, operands.getLhs(), operands.getRhs());
}

void CmpGTEF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF32OOp, CmpLTF32OOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF32OOp, CmpLTF32OOp>>(context);
}

void CmpGTEF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF64OOp, CmpLTF64OOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF64OOp, CmpLTF64OOp>>(context);
}

void CmpGTEF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF32UOp, CmpLTF32UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF32UOp, CmpLTF32UOp>>(context);
}

void CmpGTEF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF64UOp, CmpLTF64UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF64UOp, CmpLTF64UOp>>(context);
}

namespace {

/// Rewrites a vm.cmp.nz.* pseudo op to a vm.cmp.ne.* op with a constant 0.
template <typename T, typename U, typename CZ>
struct RewritePseudoCmpNZToNE : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<U>(op, op.getType(), op.getOperand(),
                                   rewriter.create<CZ>(op.getLoc()));
    return success();
  }
};

}  // namespace

OpFoldResult CmpNZF32OOp::fold(FoldAdaptor operands) {
  return constFoldUnaryCmpOp<FloatAttr>(
      operands.getOperand(),
      [&](const APFloat &a) { return APInt(32, a.isNonZero()); });
}

OpFoldResult CmpNZF64OOp::fold(FoldAdaptor operands) {
  return constFoldUnaryCmpOp<FloatAttr>(
      operands.getOperand(),
      [&](const APFloat &a) { return APInt(32, a.isNonZero()); });
}

OpFoldResult CmpNZF32UOp::fold(FoldAdaptor operands) {
  return constFoldUnaryCmpOp<FloatAttr>(
      operands.getOperand(),
      [&](const APFloat &a) { return APInt(32, a.isNonZero() || a.isNaN()); });
}

OpFoldResult CmpNZF64UOp::fold(FoldAdaptor operands) {
  return constFoldUnaryCmpOp<FloatAttr>(
      operands.getOperand(),
      [&](const APFloat &a) { return APInt(32, a.isNonZero() || a.isNaN()); });
}

void CmpNZF32OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF32OOp, CmpNEF32OOp, ConstF32ZeroOp>>(
          context);
}

void CmpNZF64OOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF64OOp, CmpNEF64OOp, ConstF64ZeroOp>>(
          context);
}

void CmpNZF32UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF32UOp, CmpNEF32UOp, ConstF32ZeroOp>>(
          context);
}

void CmpNZF64UOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF64UOp, CmpNEF64UOp, ConstF64ZeroOp>>(
          context);
}

OpFoldResult CmpNaNF32Op::fold(FoldAdaptor operands) {
  if (auto operand = operands.getOperand().dyn_cast_or_null<FloatAttr>()) {
    return operand.getValue().isNaN() ? oneOfType(getType())
                                      : zeroOfType(getType());
  }
  return {};
}

OpFoldResult CmpNaNF64Op::fold(FoldAdaptor operands) {
  if (auto operand = operands.getOperand().dyn_cast_or_null<FloatAttr>()) {
    return operand.getValue().isNaN() ? oneOfType(getType())
                                      : zeroOfType(getType());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// vm.ref comparison
//===----------------------------------------------------------------------===//

OpFoldResult CmpEQRefOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) {
    // x == x = true
    return oneOfType(getType());
  } else if (operands.getLhs() && operands.getRhs() &&
             operands.getLhs() == operands.getRhs()) {
    // Constant null == null = true
    return oneOfType(getType());
  }
  return {};
}

namespace {

/// Changes a cmp.eq.ref check against null to a cmp.nz.ref and inverted cond.
struct NullCheckCmpEQRefToCmpNZRef : public OpRewritePattern<CmpEQRefOp> {
  using OpRewritePattern<CmpEQRefOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CmpEQRefOp op,
                                PatternRewriter &rewriter) const override {
    Attribute rhs;
    if (matchPattern(op.getRhs(), m_Constant(&rhs))) {
      auto cmpNz =
          rewriter.create<CmpNZRefOp>(op.getLoc(), op.getType(), op.getLhs());
      rewriter.replaceOpWithNewOp<XorI32Op>(
          op, op.getType(), cmpNz,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpEQRefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<NullCheckCmpEQRefToCmpNZRef>(context);
}

OpFoldResult CmpNERefOp::fold(FoldAdaptor operands) {
  if (getLhs() == getRhs()) {
    // x != x = false
    return zeroOfType(getType());
  }
  return {};
}

namespace {

/// Changes a cmp.ne.ref check against null to a cmp.nz.ref.
struct NullCheckCmpNERefToCmpNZRef : public OpRewritePattern<CmpNERefOp> {
  using OpRewritePattern<CmpNERefOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CmpNERefOp op,
                                PatternRewriter &rewriter) const override {
    Attribute rhs;
    if (matchPattern(op.getRhs(), m_Constant(&rhs))) {
      rewriter.replaceOpWithNewOp<CmpNZRefOp>(op, op.getType(), op.getLhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpNERefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<NullCheckCmpNERefToCmpNZRef>(context);
}

OpFoldResult CmpNZRefOp::fold(FoldAdaptor operands) {
  Attribute operandValue;
  if (matchPattern(getOperand(), m_Constant(&operandValue))) {
    // x == null
    return zeroOfType(getType());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

/// Given a successor, try to collapse it to a new destination if it only
/// contains a passthrough unconditional branch. If the successor is
/// collapsable, `successor` and `successorOperands` are updated to reference
/// the new destination and values. `argStorage` is an optional storage to use
/// if operands to the collapsed successor need to be remapped.
static LogicalResult collapseBranch(Block *&successor,
                                    ValueRange &successorOperands,
                                    SmallVectorImpl<Value> &argStorage) {
  // Check that the successor does not have multiple predecessors.
  if (std::distance(successor->pred_begin(), successor->pred_end()) > 1) {
    return failure();
  }
  // Check that the successor only contains a unconditional branch.
  if (std::next(successor->begin()) != successor->end()) return failure();
  // Check that the terminator is an unconditional branch.
  BranchOp successorBranch = dyn_cast<BranchOp>(successor->getTerminator());
  if (!successorBranch) return failure();
  // Check that the arguments are only used within the terminator.
  for (BlockArgument arg : successor->getArguments()) {
    for (Operation *user : arg.getUsers())
      if (user != successorBranch) return failure();
  }
  // Don't try to collapse branches to infinite loops.
  Block *successorDest = successorBranch.getDest();
  if (successorDest == successor) return failure();

  // Update the operands to the successor. If the branch parent has no
  // arguments, we can use the branch operands directly.
  OperandRange operands = successorBranch.getOperands();
  if (successor->args_empty()) {
    successor = successorDest;
    successorOperands = operands;
    return success();
  }

  // Otherwise, we need to remap any argument operands.
  for (Value operand : operands) {
    BlockArgument argOperand = operand.dyn_cast<BlockArgument>();
    if (argOperand && argOperand.getOwner() == successor)
      argStorage.push_back(successorOperands[argOperand.getArgNumber()]);
    else
      argStorage.push_back(operand);
  }
  successor = successorDest;
  successorOperands = argStorage;
  return success();
}

namespace {

/// Simplify a branch to a block that has a single predecessor. This effectively
/// merges the two blocks.
///
/// (same logic as for std.br)
struct SimplifyBrToBlockWithSinglePred : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    // Check that the successor block has a single predecessor.
    Block *succ = op.getDest();
    Block *opParent = op.getOperation()->getBlock();
    if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors())) {
      return failure();
    }

    // Merge the successor into the current block and erase the branch.
    rewriter.mergeBlocks(succ, opParent, op.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

///   br ^bb1
/// ^bb1
///   br ^bbN(...)
///
///  -> br ^bbN(...)
///
/// (same logic as for std.br)
struct SimplifyPassThroughBr : public OpRewritePattern<BranchOp> {
  using OpRewritePattern<BranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BranchOp op,
                                PatternRewriter &rewriter) const override {
    Block *dest = op.getDest();
    ValueRange destOperands = op.getOperands();
    SmallVector<Value, 4> destOperandStorage;

    // Try to collapse the successor if it points somewhere other than this
    // block.
    if (dest == op.getOperation()->getBlock() ||
        failed(collapseBranch(dest, destOperands, destOperandStorage))) {
      return failure();
    }

    // Create a new branch with the collapsed successor.
    rewriter.replaceOpWithNewOp<BranchOp>(op, dest, destOperands);
    return success();
  }
};

}  // namespace

void BranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<SimplifyBrToBlockWithSinglePred, SimplifyPassThroughBr>(
      context);
}

namespace {

/// Simplifies a cond_br with a constant condition to an unconditional branch.
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CondBranchOp op,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(op.getCondition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getTrueDest(),
                                            op.getTrueOperands());
      return success();
    } else if (matchPattern(op.getCondition(), m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getFalseDest(),
                                            op.getFalseOperands());
      return success();
    }
    return failure();
  }
};

/// Simplifies a cond_br with both targets (including operands) being equal to
/// an unconditional branch.
struct SimplifySameTargetCondBranchOp : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CondBranchOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getTrueDest() != op.getFalseDest()) {
      // Targets differ so we need to be a cond branch.
      return failure();
    }

    // If all operands match between the targets then we can become a normal
    // branch to the shared target.
    auto trueOperands = llvm::to_vector<4>(op.getTrueOperands());
    auto falseOperands = llvm::to_vector<4>(op.getFalseOperands());
    if (trueOperands == falseOperands) {
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getTrueDest(), trueOperands);
      return success();
    }

    return failure();
  }
};

/// Swaps the cond_br true and false targets if the condition is inverted.
struct SwapInvertedCondBranchOpTargets : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CondBranchOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): figure out something more reliable when the xor may be
    // used on a non-binary value.
    // We generate xor(cmp(...), 1) to flip conditions, so look for that pattern
    // so that we can do the swap here and remove the xor.
    // auto condValue = op.getCondition();
    // if (auto xorOp = dyn_cast_or_null<XorI32Op>(condValue.getDefiningOp())) {
    //   Attribute rhs;
    //   if (matchPattern(xorOp.getRhs(), m_Constant(&rhs)) &&
    //       rhs.cast<IntegerAttr>().getInt() == 1) {
    //     rewriter.replaceOpWithNewOp<CondBranchOp>(
    //         op, xorOp.getLhs(), op.getFalseDest(), op.getFalseOperands(),
    //         op.getTrueDest(), op.getTrueOperands());
    //     return success();
    //   }
    // }
    return failure();
  }
};

}  // namespace

void CondBranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SimplifyConstCondBranchPred, SimplifySameTargetCondBranchOp,
                 SwapInvertedCondBranchOpTargets>(context);
}

namespace {

/// Removes vm.call ops to functions that are marked as having no side-effects
/// if the results are unused.
template <typename T>
struct EraseUnusedCallOp : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // First check if the call is unused - this ensures we only do the symbol
    // lookup if we are actually going to use it.
    for (auto result : op.getResults()) {
      if (!result.use_empty()) {
        return failure();
      }
    }

    auto *calleeOp = SymbolTable::lookupSymbolIn(
        op->template getParentOfType<ModuleOp>(), op.getCallee());

    bool hasNoSideEffects = false;
    if (calleeOp->getAttr("nosideeffects")) {
      hasNoSideEffects = true;
    } else if (auto import = dyn_cast<ImportInterface>(calleeOp)) {
      hasNoSideEffects = !import.hasSideEffects();
    }
    if (!hasNoSideEffects) {
      // Op has side-effects (or may have them); can't remove.
      return failure();
    }

    // Erase op as it is unused.
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void CallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<EraseUnusedCallOp<CallOp>>(context);
}

namespace {

/// Converts a vm.call.variadic to a non-variadic function to a normal vm.call.
struct ConvertNonVariadicToCallOp : public OpRewritePattern<CallVariadicOp> {
  using OpRewritePattern<CallVariadicOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CallVariadicOp op,
                                PatternRewriter &rewriter) const override {
    // If any segment size is != -1 (which indicates variadic) we bail.
    for (const auto &segmentSize : op.getSegmentSizes()) {
      if (segmentSize.getSExtValue() != -1) {
        return failure();
      }
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, op.getCallee(),
                                        llvm::to_vector<4>(op.getResultTypes()),
                                        op.getOperands());
    return success();
  }
};

}  // namespace

void CallVariadicOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<EraseUnusedCallOp<CallVariadicOp>, ConvertNonVariadicToCallOp>(
      context);
}

namespace {

/// Rewrites a cond_fail op to a cond_branch to a fail op.
struct RewriteCondFailToBranchFail : public OpRewritePattern<CondFailOp> {
  using OpRewritePattern<CondFailOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CondFailOp op,
                                PatternRewriter &rewriter) const override {
    auto *block = rewriter.getInsertionBlock();

    // Create the block with the vm.fail in it.
    // This is what we will branch to if the condition is true at runtime.
    auto *failBlock =
        rewriter.createBlock(block, {op.getStatus().getType()}, {op.getLoc()});
    block->moveBefore(failBlock);
    rewriter.setInsertionPointToStart(failBlock);
    rewriter.create<FailOp>(op.getLoc(), failBlock->getArgument(0),
                            op.getMessage().value_or(""));

    // Replace the original cond_fail with our cond_branch, splitting the block
    // and continuing if the condition is not taken.
    auto *continueBlock = rewriter.splitBlock(
        block, op.getOperation()->getNextNode()->getIterator());
    rewriter.setInsertionPointToEnd(block);
    rewriter.replaceOpWithNewOp<CondBranchOp>(op, op.getCondition(), failBlock,
                                              ValueRange{op.getStatus()},
                                              continueBlock, ValueRange{});

    return success();
  }
};

}  // namespace

void CondFailOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<RewriteCondFailToBranchFail>(context);
}

namespace {

/// Rewrites a check op to a cmp and a cond_fail.
template <typename CheckOp, typename CmpI32Op, typename CmpI64Op,
          typename CmpF32Op, typename CmpF64Op, typename CmpRefOp>
struct RewriteCheckToCondFail : public OpRewritePattern<CheckOp> {
  using OpRewritePattern<CheckOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CheckOp op,
                                PatternRewriter &rewriter) const override {
    Type condType = rewriter.getI32Type();
    Value condValue;
    Type operandType = op.getOperation()->getOperand(0).getType();
    if (operandType.template isa<RefType>()) {
      condValue = rewriter.template createOrFold<CmpRefOp>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else if (operandType.isInteger(32)) {
      condValue = rewriter.template createOrFold<CmpI32Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else if (operandType.isInteger(64)) {
      condValue = rewriter.template createOrFold<CmpI64Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else if (operandType.isF32()) {
      condValue = rewriter.template createOrFold<CmpF32Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else if (operandType.isF64()) {
      condValue = rewriter.template createOrFold<CmpF64Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else {
      return failure();
    }
    condValue = rewriter.createOrFold<XorI32Op>(
        op.getLoc(), condType, condValue,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
    auto statusCode = rewriter.createOrFold<ConstI32Op>(
        op.getLoc(), /*IREE_STATUS_FAILED_PRECONDITION=*/9);
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(op, condValue, statusCode,
                                                      op.getMessageAttr());
    return success();
  }
};

}  // namespace

void CheckEQOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckEQOp, CmpEQI32Op, CmpEQI64Op,
                                        CmpEQF32OOp, CmpEQF64OOp, CmpEQRefOp>>(
      context);
}

void CheckNEOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNEOp, CmpNEI32Op, CmpNEI64Op,
                                        CmpNEF32OOp, CmpNEF64OOp, CmpNERefOp>>(
      context);
}

void CheckNZOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNZOp, CmpNZI32Op, CmpNZI64Op,
                                        CmpNZF32OOp, CmpNZF64OOp, CmpNZRefOp>>(
      context);
}

void CheckNearlyEQOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<
      RewriteCheckToCondFail<CheckNearlyEQOp, CmpEQI32Op, CmpEQI64Op,
                             CmpEQF32NearOp, CmpEQF64NearOp, CmpEQRefOp>>(
      context);
}

namespace {

// Folds vm.import.resolved ops referencing required imports.
struct RequiredImportResolver : public OpRewritePattern<ImportResolvedOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ImportResolvedOp op,
                                PatternRewriter &rewriter) const override {
    auto importOp = SymbolTable::lookupNearestSymbolFrom<IREE::VM::ImportOp>(
        op, op.getImportAttr());
    if (!importOp || importOp.getIsOptional()) return failure();
    rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(op, 1);
    return success();
  }
};

}  // namespace

void ImportResolvedOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<RequiredImportResolver>(context);
}

//===----------------------------------------------------------------------===//
// Async/fiber ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

namespace {

template <typename T>
struct RemoveDisabledDebugOp : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): if debug disabled then replace inputs -> outputs.
    return failure();
  }
};

template <typename T>
struct RemoveDisabledDebugAsyncOp : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): if debug disabled then replace with a branch to dest.
    return failure();
  }
};

struct SimplifyConstCondBreakPred : public OpRewritePattern<CondBreakOp> {
  using OpRewritePattern<CondBreakOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CondBreakOp op,
                                PatternRewriter &rewriter) const override {
    IntegerAttr condValue;
    if (!matchPattern(op.getCondition(), m_Constant(&condValue))) {
      return failure();
    }

    if (condValue.getValue() != 0) {
      // True - always break (to the same destination).
      rewriter.replaceOpWithNewOp<BreakOp>(op, op.getDest(),
                                           op.getDestOperands());
    } else {
      // False - skip the break.
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDest(),
                                            op.getDestOperands());
    }
    return success();
  }
};

}  // namespace

void TraceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugOp<TraceOp>>(context);
}

void PrintOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugOp<PrintOp>>(context);
}

void BreakOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugAsyncOp<BreakOp>>(context);
}

void CondBreakOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<RemoveDisabledDebugAsyncOp<CondBreakOp>,
                 SimplifyConstCondBreakPred>(context);
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
