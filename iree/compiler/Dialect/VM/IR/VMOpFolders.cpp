// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

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

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

namespace {

/// Converts global initializer functions that evaluate to a constant to a
/// specified initial value.
template <typename T>
struct InlineConstGlobalOpInitializer : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (!op.initializer()) return failure();
    auto initializer = dyn_cast_or_null<FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.initializer().getValue()));
    if (!initializer) return failure();
    if (initializer.getBlocks().size() == 1 &&
        initializer.getBlocks().front().getOperations().size() == 2 &&
        isa<ReturnOp>(initializer.getBlocks().front().getOperations().back())) {
      auto &primaryOp = initializer.getBlocks().front().getOperations().front();
      Attribute constResult;
      if (matchPattern(primaryOp.getResult(0), m_Constant(&constResult))) {
        rewriter.replaceOpWithNewOp<T>(op, op.sym_name(), op.is_mutable(),
                                       op.type(), constResult);
        return success();
      }
    }
    return failure();
  }
};

/// Drops initial_values from globals where the value is 0, as by default all
/// globals are zero-initialized upon module load.
template <typename T>
struct DropDefaultConstGlobalOpInitializer : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (!op.initial_value().hasValue()) return failure();
    if (auto value = op.initial_valueAttr().template dyn_cast<IntegerAttr>()) {
      if (value.getValue() != 0) return failure();
    } else if (auto value =
                   op.initial_valueAttr().template dyn_cast<FloatAttr>()) {
      if (value.getValue().isNonZero()) return failure();
    }
    rewriter.replaceOpWithNewOp<T>(op, op.sym_name(), op.is_mutable(),
                                   op.type(),
                                   llvm::to_vector<4>(op->getDialectAttrs()));
    return success();
  }
};

}  // namespace

void GlobalI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalI32Op>,
                 DropDefaultConstGlobalOpInitializer<GlobalI32Op>>(context);
}

void GlobalI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalI64Op>,
                 DropDefaultConstGlobalOpInitializer<GlobalI64Op>>(context);
}

void GlobalF32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalF32Op>,
                 DropDefaultConstGlobalOpInitializer<GlobalF32Op>>(context);
}

void GlobalF64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalF64Op>,
                 DropDefaultConstGlobalOpInitializer<GlobalF64Op>>(context);
}

void GlobalRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalRefOp>>(context);
}

namespace {

/// Inlines immutable global constants into their loads.
template <typename LOAD_OP, typename GLOBAL_OP, typename CONST_OP,
          typename CONST_ZERO_OP>
struct InlineConstGlobalLoadIntegerOp : public OpRewritePattern<LOAD_OP> {
  using OpRewritePattern<LOAD_OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(LOAD_OP op,
                                PatternRewriter &rewriter) const override {
    auto globalAttr = op->template getAttrOfType<FlatSymbolRefAttr>("global");
    auto globalOp =
        op->template getParentOfType<VM::ModuleOp>()
            .template lookupSymbol<GLOBAL_OP>(globalAttr.getValue());
    if (!globalOp) return failure();
    if (globalOp.is_mutable()) return failure();
    if (globalOp.initial_value()) {
      rewriter.replaceOpWithNewOp<CONST_OP>(
          op, globalOp.initial_value().getValue());
    } else {
      rewriter.replaceOpWithNewOp<CONST_ZERO_OP>(op);
    }
    return success();
  }
};

}  // namespace

void GlobalLoadI32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadIntegerOp<GlobalLoadI32Op, GlobalI32Op,
                                                ConstI32Op, ConstI32ZeroOp>>(
      context);
}

void GlobalLoadI64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadIntegerOp<GlobalLoadI64Op, GlobalI64Op,
                                                ConstI64Op, ConstI64ZeroOp>>(
      context);
}

void GlobalLoadF32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadIntegerOp<GlobalLoadF32Op, GlobalF32Op,
                                                ConstF32Op, ConstF32ZeroOp>>(
      context);
}

void GlobalLoadF64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadIntegerOp<GlobalLoadF64Op, GlobalF64Op,
                                                ConstF64Op, ConstF64ZeroOp>>(
      context);
}

namespace {

/// Inlines immutable global constants into their loads.
struct InlineConstGlobalLoadRefOp : public OpRewritePattern<GlobalLoadRefOp> {
  using OpRewritePattern<GlobalLoadRefOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadRefOp op,
                                PatternRewriter &rewriter) const override {
    auto globalAttr = op->getAttrOfType<FlatSymbolRefAttr>("global");
    auto globalOp =
        op->getParentOfType<VM::ModuleOp>().lookupSymbol<GlobalRefOp>(
            globalAttr.getValue());
    if (!globalOp) return failure();
    if (globalOp.is_mutable()) return failure();
    rewriter.replaceOpWithNewOp<ConstRefZeroOp>(op, op.getType());
    return success();
  }
};

}  // namespace

void GlobalLoadRefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadRefOp>(context);
}

namespace {

template <typename INDIRECT, typename DIRECT>
struct PropagateGlobalLoadAddress : public OpRewritePattern<INDIRECT> {
  using OpRewritePattern<INDIRECT>::OpRewritePattern;

  LogicalResult matchAndRewrite(INDIRECT op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<DIRECT>(op, op.value().getType(),
                                          addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalLoadIndirectI32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectI32Op, GlobalLoadI32Op>>(
      context);
}

void GlobalLoadIndirectI64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectI64Op, GlobalLoadI64Op>>(
      context);
}

void GlobalLoadIndirectF32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectF32Op, GlobalLoadF32Op>>(
      context);
}

void GlobalLoadIndirectF64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectF64Op, GlobalLoadF64Op>>(
      context);
}

void GlobalLoadIndirectRefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
    if (auto addressOp =
            dyn_cast_or_null<GlobalAddressOp>(op.global().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<DIRECT>(op, op.value(), addressOp.global());
      return success();
    }
    return failure();
  }
};

}  // namespace

void GlobalStoreIndirectI32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectI32Op, GlobalStoreI32Op>>(
      context);
}

void GlobalStoreIndirectI64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectI64Op, GlobalStoreI64Op>>(
      context);
}

void GlobalStoreIndirectF32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectF32Op, GlobalStoreF32Op>>(
      context);
}

void GlobalStoreIndirectF64Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectF64Op, GlobalStoreF64Op>>(
      context);
}

void GlobalStoreIndirectRefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectRefOp, GlobalStoreRefOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

namespace {

template <typename GeneralOp, typename ZeroOp>
struct FoldZeroConstPrimitive final : public OpRewritePattern<GeneralOp> {
  using OpRewritePattern<GeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GeneralOp constOp,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(constOp.result(), m_Zero())) {
      rewriter.replaceOpWithNewOp<ZeroOp>(constOp);
      return success();
    }
    return failure();
  }
};

}  // namespace

OpFoldResult ConstI32Op::fold(ArrayRef<Attribute> operands) { return value(); }

void ConstI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<ConstI32Op, ConstI32ZeroOp>>(context);
}

OpFoldResult ConstI64Op::fold(ArrayRef<Attribute> operands) { return value(); }

void ConstI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<ConstI64Op, ConstI64ZeroOp>>(context);
}

OpFoldResult ConstF32Op::fold(ArrayRef<Attribute> operands) { return value(); }

void ConstF32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<ConstF32Op, ConstF32ZeroOp>>(context);
}

OpFoldResult ConstF64Op::fold(ArrayRef<Attribute> operands) { return value(); }

void ConstF64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldZeroConstPrimitive<ConstF64Op, ConstF64ZeroOp>>(context);
}

OpFoldResult ConstI32ZeroOp::fold(ArrayRef<Attribute> operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstI64ZeroOp::fold(ArrayRef<Attribute> operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstF32ZeroOp::fold(ArrayRef<Attribute> operands) {
  return FloatAttr::get(getResult().getType(), 0.0f);
}

OpFoldResult ConstF64ZeroOp::fold(ArrayRef<Attribute> operands) {
  return FloatAttr::get(getResult().getType(), 0.0);
}

OpFoldResult ConstRefZeroOp::fold(ArrayRef<Attribute> operands) {
  // TODO(b/144027097): relace unit attr with a proper null ref attr.
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
  if (matchPattern(op.condition(), m_Zero())) {
    // 0 ? x : y = y
    return op.false_value();
  } else if (matchPattern(op.condition(), m_NonZero())) {
    // !0 ? x : y = x
    return op.true_value();
  } else if (op.true_value() == op.false_value()) {
    // c ? x : x = x
    return op.true_value();
  }
  return {};
}

OpFoldResult SelectI32Op::fold(ArrayRef<Attribute> operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectI64Op::fold(ArrayRef<Attribute> operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectF32Op::fold(ArrayRef<Attribute> operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectF64Op::fold(ArrayRef<Attribute> operands) {
  return foldSelectOp(*this);
}

OpFoldResult SelectRefOp::fold(ArrayRef<Attribute> operands) {
  return foldSelectOp(*this);
}

template <typename T>
static OpFoldResult foldSwitchOp(T op) {
  APInt indexValue;
  if (matchPattern(op.index(), m_ConstantInt(&indexValue))) {
    // Index is constant and we can resolve immediately.
    int64_t index = indexValue.getSExtValue();
    if (index < 0 || index >= op.values().size()) {
      return op.default_value();
    }
    return op.values()[index];
  }

  bool allValuesMatch = true;
  for (auto value : op.values()) {
    if (value != op.default_value()) {
      allValuesMatch = false;
      break;
    }
  }
  if (allValuesMatch) {
    // All values (and the default) are the same so just return it regardless of
    // the provided index.
    return op.default_value();
  }

  return {};
}

OpFoldResult SwitchI32Op::fold(ArrayRef<Attribute> operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchI64Op::fold(ArrayRef<Attribute> operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchF32Op::fold(ArrayRef<Attribute> operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchF64Op::fold(ArrayRef<Attribute> operands) {
  return foldSwitchOp(*this);
}

OpFoldResult SwitchRefOp::fold(ArrayRef<Attribute> operands) {
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
static Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                                  const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<AttrElementT>()) {
    return AttrElementT::get(operand.getType(), calculate(operand.getValue()));
  } else if (auto operand = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    auto elementResult =
        constFoldUnaryOp<AttrElementT>({operand.getSplatValue()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(operand.getType(), elementResult);
  } else if (auto operand = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    return operand.mapValues(
        operand.getType().getElementType(),
        llvm::function_ref<ElementValueT(const ElementValueT &)>(
            [&](const ElementValueT &value) { return calculate(value); }));
  }
  return {};
}

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
static Attribute constFoldFloatUnaryOp(
    ArrayRef<Attribute> operands,
    const std::function<APFloat(APFloat)> &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return FloatAttr::get(operand.getType(), calculate(operand.getValue()));
  } else if (auto operand = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    auto elementResult =
        constFoldFloatUnaryOp({operand.getSplatValue()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(operand.getType(), elementResult);
  } else if (auto operand = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    return operand.mapValues(
        operand.getType().getElementType(),
        llvm::function_ref<APInt(const APFloat &)>([&](const APFloat &value) {
          return calculate(value).bitcastToAPInt();
        }));
  }
  return {};
}

/// Performs const folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                                   const CalculationT &calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (auto lhs = operands[0].dyn_cast_or_null<AttrElementT>()) {
    auto rhs = operands[1].dyn_cast_or_null<AttrElementT>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    return AttrElementT::get(lhs.getType(),
                             calculate(lhs.getValue(), rhs.getValue()));
  } else if (auto lhs = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    // TODO(benvanik): handle splat/otherwise.
    auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto elementResult = constFoldBinaryOp<AttrElementT>(
        {lhs.getSplatValue(), rhs.getSplatValue()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(lhs.getType(), elementResult);
  } else if (auto lhs = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    auto rhs = operands[1].dyn_cast_or_null<ElementsAttr>();
    if (!rhs || lhs.getType() != rhs.getType()) return {};
    auto lhsIt = lhs.getValues<AttrElementT>().begin();
    auto rhsIt = rhs.getValues<AttrElementT>().begin();
    SmallVector<Attribute, 4> resultAttrs(lhs.getNumElements());
    for (int64_t i = 0; i < lhs.getNumElements(); ++i) {
      resultAttrs[i] =
          constFoldBinaryOp<AttrElementT>({*lhsIt, *rhsIt}, calculate);
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
static Attribute constFoldTernaryOp(ArrayRef<Attribute> operands,
                                    const CalculationT &calculate) {
  assert(operands.size() == 3 && "ternary op takes two operands");
  if (auto a = operands[0].dyn_cast_or_null<AttrElementT>()) {
    auto b = operands[1].dyn_cast_or_null<AttrElementT>();
    auto c = operands[2].dyn_cast_or_null<AttrElementT>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    return AttrElementT::get(
        a.getType(), calculate(a.getValue(), b.getValue(), c.getValue()));
  } else if (auto a = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    // TODO(benvanik): handle splat/otherwise.
    auto b = operands[1].dyn_cast_or_null<SplatElementsAttr>();
    auto c = operands[2].dyn_cast_or_null<SplatElementsAttr>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    auto elementResult = constFoldTernaryOp<AttrElementT>(
        {a.getSplatValue(), b.getSplatValue(), c.getSplatValue()}, calculate);
    if (!elementResult) return {};
    return DenseElementsAttr::get(a.getType(), elementResult);
  } else if (auto a = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    auto b = operands[1].dyn_cast_or_null<ElementsAttr>();
    auto c = operands[2].dyn_cast_or_null<ElementsAttr>();
    if (!b || !c || a.getType() != b.getType() || a.getType() != c.getType()) {
      return {};
    }
    auto aIt = a.getValues<AttrElementT>().begin();
    auto bIt = b.getValues<AttrElementT>().begin();
    auto cIt = c.getValues<AttrElementT>().begin();
    SmallVector<Attribute, 4> resultAttrs(a.getNumElements());
    for (int64_t i = 0; i < a.getNumElements(); ++i) {
      resultAttrs[i] =
          constFoldTernaryOp<AttrElementT>({*aIt, *bIt, *cIt}, calculate);
      if (!resultAttrs[i]) return {};
      ++aIt;
      ++bIt;
      ++cIt;
    }
    return DenseElementsAttr::get(a.getType(), resultAttrs);
  }
  return {};
}

template <class AttrElementT, typename ADD, typename SUB,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldAddOp(ADD op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x + 0 = x or 0 + y = y (commutative)
    return op.lhs();
  }
  if (auto subOp = dyn_cast_or_null<SUB>(op.lhs().getDefiningOp())) {
    if (subOp.lhs() == op.rhs()) return subOp.rhs();
    if (subOp.rhs() == op.rhs()) return subOp.lhs();
  } else if (auto subOp = dyn_cast_or_null<SUB>(op.rhs().getDefiningOp())) {
    if (subOp.lhs() == op.lhs()) return subOp.rhs();
    if (subOp.rhs() == op.lhs()) return subOp.lhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      operands,
      [](const ElementValueT &a, const ElementValueT &b) { return a + b; });
}

OpFoldResult AddI32Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp<IntegerAttr, AddI32Op, SubI32Op>(*this, operands);
}

OpFoldResult AddI64Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp<IntegerAttr, AddI64Op, SubI64Op>(*this, operands);
}

template <class AttrElementT, typename SUB, typename ADD,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldSubOp(SUB op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x - 0 = x
    return op.lhs();
  }
  if (auto addOp = dyn_cast_or_null<ADD>(op.lhs().getDefiningOp())) {
    if (addOp.lhs() == op.rhs()) return addOp.rhs();
    if (addOp.rhs() == op.rhs()) return addOp.lhs();
  } else if (auto addOp = dyn_cast_or_null<ADD>(op.rhs().getDefiningOp())) {
    if (addOp.lhs() == op.lhs()) return addOp.rhs();
    if (addOp.rhs() == op.lhs()) return addOp.lhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      operands,
      [](const ElementValueT &a, const ElementValueT &b) { return a - b; });
}

OpFoldResult SubI32Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp<IntegerAttr, SubI32Op, AddI32Op>(*this, operands);
}

OpFoldResult SubI64Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp<IntegerAttr, SubI64Op, AddI64Op>(*this, operands);
}

template <class AttrElementT, typename T,
          class ElementValueT = typename AttrElementT::ValueType>
static OpFoldResult foldMulOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x * 0 = 0 or 0 * y = 0 (commutative)
    return zeroOfType(op.getType());
  } else if (matchPattern(op.rhs(), m_One())) {
    // x * 1 = x or 1 * y = y (commutative)
    return op.lhs();
  }
  return constFoldBinaryOp<AttrElementT>(
      operands,
      [](const ElementValueT &a, const ElementValueT &b) { return a * b; });
}

template <class AttrElementT, typename T, typename CONST_OP,
          class ElementValueT = typename AttrElementT::ValueType>
struct FoldConstantMulOperand : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    AttrElementT c1, c2;
    if (!matchPattern(op.rhs(), m_Constant(&c1))) return failure();
    if (auto mulOp = dyn_cast_or_null<T>(op.lhs().getDefiningOp())) {
      if (matchPattern(mulOp.rhs(), m_Constant(&c2))) {
        auto c = rewriter.createOrFold<CONST_OP>(
            rewriter.getFusedLoc({mulOp.getLoc(), op.getLoc()}),
            constFoldBinaryOp<AttrElementT>(
                {c1, c2}, [](const ElementValueT &a, const ElementValueT &b) {
                  return a * b;
                }));
        rewriter.replaceOpWithNewOp<T>(op, op.getType(), mulOp.lhs(), c);
        return success();
      }
    }
    return failure();
  }
};

OpFoldResult MulI32Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp<IntegerAttr>(*this, operands);
}

void MulI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<IntegerAttr, MulI32Op, ConstI32Op>>(
      context);
}

OpFoldResult MulI64Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp<IntegerAttr>(*this, operands);
}

void MulI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<IntegerAttr, MulI64Op, ConstI64Op>>(
      context);
}

template <typename T>
static OpFoldResult foldDivSOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.rhs(), m_One())) {
    // x / 1 = x
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a.sdiv(b); });
}

OpFoldResult DivI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldDivSOp(*this, operands);
}

OpFoldResult DivI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldDivSOp(*this, operands);
}

template <typename T>
static OpFoldResult foldDivUOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.rhs(), m_One())) {
    // x / 1 = x
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a.udiv(b); });
}

OpFoldResult DivI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldDivUOp(*this, operands);
}

OpFoldResult DivI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldDivUOp(*this, operands);
}

template <typename T>
static OpFoldResult foldRemSOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero()) ||
             matchPattern(op.rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a.srem(b); });
}

OpFoldResult RemI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldRemSOp(*this, operands);
}

OpFoldResult RemI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldRemSOp(*this, operands);
}

template <typename T>
static OpFoldResult foldRemUOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero()) ||
             matchPattern(op.rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a.urem(b); });
}

OpFoldResult RemI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldRemUOp(*this, operands);
}

OpFoldResult RemI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldRemUOp(*this, operands);
}

template <typename T>
static OpFoldResult foldFMAOp(T op, ArrayRef<Attribute> operands) {
  // a * b + c
  if (matchPattern(op.a(), m_Zero()) || matchPattern(op.b(), m_Zero())) {
    return op.c();
  }
  return constFoldTernaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b, const APInt &c) {
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
    if (matchPattern(fmaOp.a(), m_One())) {
      // 1 * b + c = b + c
      rewriter.replaceOpWithNewOp<AddOp>(fmaOp, fmaOp.getType(), fmaOp.b(),
                                         fmaOp.c());
      return success();
    } else if (matchPattern(fmaOp.b(), m_One())) {
      // a * 1 + c = a + c
      rewriter.replaceOpWithNewOp<AddOp>(fmaOp, fmaOp.getType(), fmaOp.a(),
                                         fmaOp.c());
      return success();
    } else if (matchPattern(fmaOp.c(), m_Zero())) {
      // a * b + 0 = a * b
      rewriter.replaceOpWithNewOp<MulOp>(fmaOp, fmaOp.getType(), fmaOp.a(),
                                         fmaOp.b());
      return success();
    }
    return failure();
  }
};

OpFoldResult FMAI32Op::fold(ArrayRef<Attribute> operands) {
  return foldFMAOp(*this, operands);
}

OpFoldResult FMAI64Op::fold(ArrayRef<Attribute> operands) {
  return foldFMAOp(*this, operands);
}

void FMAI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAI32Op, MulI32Op, AddI32Op>>(context);
}

void FMAI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAI64Op, MulI64Op, AddI64Op>>(context);
}

//===----------------------------------------------------------------------===//
// Floating-point arithmetic
//===----------------------------------------------------------------------===//

OpFoldResult AddF32Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp<FloatAttr, AddF32Op, SubF32Op>(*this, operands);
}

OpFoldResult AddF64Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp<FloatAttr, AddF64Op, SubF64Op>(*this, operands);
}

OpFoldResult SubF32Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp<FloatAttr, SubF32Op, AddF32Op>(*this, operands);
}

OpFoldResult SubF64Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp<FloatAttr, SubF64Op, AddF64Op>(*this, operands);
}

OpFoldResult MulF32Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp<FloatAttr>(*this, operands);
}

void MulF32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<FloatAttr, MulF32Op, ConstF32Op>>(
      context);
}

OpFoldResult MulF64Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp<FloatAttr>(*this, operands);
}

void MulF64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<FoldConstantMulOperand<FloatAttr, MulF64Op, ConstF64Op>>(
      context);
}

template <typename T>
static OpFoldResult foldDivFOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x / 0 = death
    op.emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.rhs(), m_One())) {
    // x / 1 = x
    return op.lhs();
  }
  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) {
        APFloat c = a;
        c.divide(b, APFloat::rmNearestTiesToAway);
        return c;
      });
}

OpFoldResult DivF32Op::fold(ArrayRef<Attribute> operands) {
  return foldDivFOp(*this, operands);
}

OpFoldResult DivF64Op::fold(ArrayRef<Attribute> operands) {
  return foldDivFOp(*this, operands);
}

template <typename T>
static OpFoldResult foldRemFOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x % 0 = death
    op.emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(op.lhs(), m_Zero()) ||
             matchPattern(op.rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(operands,
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat c = a;
                                        c.remainder(b);
                                        return c;
                                      });
}

OpFoldResult RemF32Op::fold(ArrayRef<Attribute> operands) {
  return foldRemFOp(*this, operands);
}

OpFoldResult RemF64Op::fold(ArrayRef<Attribute> operands) {
  return foldRemFOp(*this, operands);
}

template <typename T>
static OpFoldResult foldFMAFOp(T op, ArrayRef<Attribute> operands) {
  // a * b + c
  if (matchPattern(op.a(), m_Zero()) || matchPattern(op.b(), m_Zero())) {
    return op.c();
  }
  return constFoldTernaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b, const APFloat &c) {
        APFloat d = a;
        d.fusedMultiplyAdd(b, c, APFloat::rmNearestTiesToAway);
        return d;
      });
}

OpFoldResult FMAF32Op::fold(ArrayRef<Attribute> operands) {
  return foldFMAFOp(*this, operands);
}

OpFoldResult FMAF64Op::fold(ArrayRef<Attribute> operands) {
  return foldFMAFOp(*this, operands);
}

void FMAF32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAF32Op, MulF32Op, AddF32Op>>(context);
}

void FMAF64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<CanonicalizeFMA<FMAF64Op, MulF64Op, AddF64Op>>(context);
}

OpFoldResult AbsF32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.clearSign();
    return b;
  });
}

OpFoldResult AbsF64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.clearSign();
    return b;
  });
}

OpFoldResult NegF32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.changeSign();
    return b;
  });
}

OpFoldResult NegF64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.changeSign();
    return b;
  });
}

OpFoldResult CeilF32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardPositive);
    return b;
  });
}

OpFoldResult CeilF64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardPositive);
    return b;
  });
}

OpFoldResult FloorF32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardNegative);
    return b;
  });
}

OpFoldResult FloorF64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    auto b = a;
    b.roundToIntegral(APFloat::rmTowardNegative);
    return b;
  });
}

//===----------------------------------------------------------------------===//
// Floating-point math
//===----------------------------------------------------------------------===//

OpFoldResult SqrtF32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    return APFloat(sqrtf(a.convertToFloat()));
  });
}

OpFoldResult SqrtF64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldFloatUnaryOp(operands, [](const APFloat &a) {
    return APFloat(sqrt(a.convertToDouble()));
  });
}

//===----------------------------------------------------------------------===//
// Integer bit manipulation
//===----------------------------------------------------------------------===//

template <typename T>
static OpFoldResult foldNotOp(T op, ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](APInt a) {
    a.flipAllBits();
    return a;
  });
}

OpFoldResult NotI32Op::fold(ArrayRef<Attribute> operands) {
  return foldNotOp(*this, operands);
}

OpFoldResult NotI64Op::fold(ArrayRef<Attribute> operands) {
  return foldNotOp(*this, operands);
}

template <typename T>
static OpFoldResult foldAndOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x & 0 = 0 or 0 & y = 0 (commutative)
    return zeroOfType(op.getType());
  } else if (op.lhs() == op.rhs()) {
    // x & x = x
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a & b; });
}

OpFoldResult AndI32Op::fold(ArrayRef<Attribute> operands) {
  return foldAndOp(*this, operands);
}

OpFoldResult AndI64Op::fold(ArrayRef<Attribute> operands) {
  return foldAndOp(*this, operands);
}

template <typename T>
static OpFoldResult foldOrOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x | 0 = x or 0 | y = y (commutative)
    return op.lhs();
  } else if (op.lhs() == op.rhs()) {
    // x | x = x
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a | b; });
}

OpFoldResult OrI32Op::fold(ArrayRef<Attribute> operands) {
  return foldOrOp(*this, operands);
}

OpFoldResult OrI64Op::fold(ArrayRef<Attribute> operands) {
  return foldOrOp(*this, operands);
}

template <typename T>
static OpFoldResult foldXorOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x ^ 0 = x or 0 ^ y = y (commutative)
    return op.lhs();
  } else if (op.lhs() == op.rhs()) {
    // x ^ x = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a ^ b; });
}

OpFoldResult XorI32Op::fold(ArrayRef<Attribute> operands) {
  return foldXorOp(*this, operands);
}

OpFoldResult XorI64Op::fold(ArrayRef<Attribute> operands) {
  return foldXorOp(*this, operands);
}

//===----------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===----------------------------------------------------------------------===//

template <typename T>
static OpFoldResult foldShlOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.operand(), m_Zero())) {
    // 0 << y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.amount(), m_Zero())) {
    // x << 0 = x
    return op.operand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.shl(b); });
}

OpFoldResult ShlI32Op::fold(ArrayRef<Attribute> operands) {
  return foldShlOp(*this, operands);
}

OpFoldResult ShlI64Op::fold(ArrayRef<Attribute> operands) {
  return foldShlOp(*this, operands);
}

template <typename T>
static OpFoldResult foldShrSOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.operand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.amount(), m_Zero())) {
    // x >> 0 = x
    return op.operand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.ashr(b); });
}

OpFoldResult ShrI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldShrSOp(*this, operands);
}

OpFoldResult ShrI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldShrSOp(*this, operands);
}

template <typename T>
static OpFoldResult foldShrUOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.operand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(op.getType());
  } else if (matchPattern(op.amount(), m_Zero())) {
    // x >> 0 = x
    return op.operand();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.lshr(b); });
}

OpFoldResult ShrI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldShrUOp(*this, operands);
}

OpFoldResult ShrI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldShrUOp(*this, operands);
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<ElementValueT(ElementValueT)>>
static Attribute constFoldConversionOp(Type resultType,
                                       ArrayRef<Attribute> operands,
                                       const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<AttrElementT>()) {
    return AttrElementT::get(resultType, calculate(operand.getValue()));
  }
  return {};
}

OpFoldResult TruncI32I8Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI32I16Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I8Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI64I16Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(32); });
}

OpFoldResult TruncF64F32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<FloatAttr>(
      FloatType::getF32(getContext()), operands,
      [&](const APFloat &a) { return APFloat(a.convertToFloat()); });
}

OpFoldResult ExtI8I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(8).sext(32); });
}

OpFoldResult ExtI8I32UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(8).zext(32); });
}

OpFoldResult ExtI16I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(16).sext(32); });
}

OpFoldResult ExtI16I32UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 32), operands,
      [&](const APInt &a) { return a.trunc(16).zext(32); });
}

OpFoldResult ExtI8I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.trunc(8).sext(64); });
}

OpFoldResult ExtI8I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.trunc(8).zext(64); });
}

OpFoldResult ExtI16I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.trunc(16).sext(64); });
}

OpFoldResult ExtI16I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.trunc(16).zext(64); });
}

OpFoldResult ExtI32I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.sext(64); });
}

OpFoldResult ExtI32I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(getContext(), 64), operands,
      [&](const APInt &a) { return a.zext(64); });
}

OpFoldResult ExtF32F64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<FloatAttr>(
      FloatType::getF64(getContext()), operands,
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
        op.getLoc(), rewriter.getIntegerType(SZ_T), op.operand());
    rewriter.replaceOpWithNewOp<OP_B>(op, op.result().getType(), tmp);
    return success();
  }
};

}  // namespace

void TruncI64I8Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      TruncI64I8Op, TruncI64I32Op, 32, TruncI32I8Op>>(context);
}

void TruncI64I16Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      TruncI64I16Op, TruncI64I32Op, 32, TruncI32I16Op>>(context);
}

void ExtI8I64SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI8I64SOp, ExtI8I32SOp, 32, ExtI32I64SOp>>(context);
}

void ExtI8I64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI8I64UOp, ExtI8I32UOp, 32, ExtI32I64UOp>>(context);
}

void ExtI16I64SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI16I64SOp, ExtI16I32SOp, 32, ExtI32I64SOp>>(context);
}

void ExtI16I64UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<PseudoIntegerConversionToSplitConversionOp<
      ExtI16I64UOp, ExtI16I32UOp, 32, ExtI32I64UOp>>(context);
}

template <
    class SrcAttrElementT, class DstAttrElementT,
    class SrcElementValueT = typename SrcAttrElementT::ValueType,
    class DstElementValueT = typename DstAttrElementT::ValueType,
    class CalculationT = std::function<DstElementValueT(SrcElementValueT)>>
static Attribute constFoldCastOp(Type resultType, ArrayRef<Attribute> operands,
                                 const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<SrcAttrElementT>()) {
    return DstAttrElementT::get(resultType, calculate(operand.getValue()));
  }
  return {};
}

OpFoldResult CastSI32F32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      Float32Type::get(getContext()), operands, [&](const APInt &a) {
        APFloat b = APFloat(0.0f);
        b.convertFromAPInt(a, /*IsSigned=*/true, APFloat::rmNearestTiesToAway);
        return b;
      });
}

OpFoldResult CastUI32F32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      Float32Type::get(getContext()), operands, [&](const APInt &a) {
        APFloat b = APFloat(0.0f);
        b.convertFromAPInt(a, /*IsSigned=*/false, APFloat::rmNearestTiesToAway);
        return b;
      });
}

OpFoldResult CastF32SI32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      IntegerType::get(getContext(), 32), operands, [&](const APFloat &a) {
        bool isExact = false;
        llvm::APSInt b;
        a.convertToInteger(b, APFloat::rmNearestTiesToAway, &isExact);
        return b;
      });
}

OpFoldResult CastF32UI32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      IntegerType::get(getContext(), 32), operands, [&](const APFloat &a) {
        bool isExact = false;
        llvm::APSInt b;
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

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<APInt(ElementValueT)>>
static Attribute constFoldCmpOp(ArrayRef<Attribute> operands,
                                const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<AttrElementT>()) {
    auto boolType = IntegerType::get(operand.getContext(), 32);
    return IntegerAttr::get(boolType, calculate(operand.getValue()));
  } else if (auto operand = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    auto boolType = IntegerType::get(operand.getContext(), 32);
    return operand.mapValues(
        boolType,
        llvm::function_ref<APInt(const ElementValueT &)>(
            [&](const ElementValueT &value) { return calculate(value); }));
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
    if (!op.result().hasOneUse()) {
      // Can't change if there are multiple users.
      return failure();
    }
    if (auto xorOp = dyn_cast_or_null<XorI32Op>(*op.result().user_begin())) {
      Attribute rhs;
      if (xorOp.lhs() == op.result() &&
          matchPattern(xorOp.rhs(), m_Constant(&rhs)) &&
          rhs.cast<IntegerAttr>().getInt() == 1) {
        auto invValue = rewriter.createOrFold<INV>(
            op.getLoc(), op.result().getType(), op.lhs(), op.rhs());
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
static OpFoldResult foldCmpEQOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x == x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.eq(b); });
}

OpFoldResult CmpEQI32Op::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQOp(*this, operands);
}

void CmpEQI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQI32Op, CmpNEI32Op>>(context);
}

OpFoldResult CmpEQI64Op::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQOp(*this, operands);
}

void CmpEQI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQI64Op, CmpNEI64Op>>(context);
}

template <typename T>
static OpFoldResult foldCmpNEOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x != x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.ne(b); });
}

OpFoldResult CmpNEI32Op::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEOp(*this, operands);
}

OpFoldResult CmpNEI64Op::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEOp(*this, operands);
}

namespace {

/// Changes a cmp.ne.i32 check against 0 to a cmp.nz.i32.
template <typename NE_OP, typename NZ_OP>
struct CmpNEZeroToCmpNZ : public OpRewritePattern<NE_OP> {
  using OpRewritePattern<NE_OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(NE_OP op,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(op.rhs(), m_Zero())) {
      rewriter.replaceOpWithNewOp<NZ_OP>(op, op.getType(), op.lhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpNEI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEI32Op, CmpEQI32Op>,
                 CmpNEZeroToCmpNZ<CmpNEI32Op, CmpNZI32Op>>(context);
}

void CmpNEI64Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEI64Op, CmpEQI64Op>,
                 CmpNEZeroToCmpNZ<CmpNEI64Op, CmpNZI64Op>>(context);
}

template <typename T>
static OpFoldResult foldCmpLTSOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.slt(b); });
}

OpFoldResult CmpLTI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTSOp(*this, operands);
}

OpFoldResult CmpLTI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTSOp(*this, operands);
}

void CmpLTI32SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

void CmpLTI64SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

template <typename T>
static OpFoldResult foldCmpLTUOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.ult(b); });
}

OpFoldResult CmpLTI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTUOp(*this, operands);
}

OpFoldResult CmpLTI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTUOp(*this, operands);
}

void CmpLTI32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

void CmpLTI64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

namespace {

/// Rewrites a vm.cmp.lte.* pseudo op to a vm.cmp.lt.* op.
template <typename T, typename U>
struct RewritePseudoCmpLTEToLT : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // !(lhs > rhs)
    auto condValue =
        rewriter.createOrFold<U>(op.getLoc(), op.getType(), op.rhs(), op.lhs());
    rewriter.replaceOpWithNewOp<XorI32Op>(
        op, op.getType(), condValue,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpLTESOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.sle(b); });
}

OpFoldResult CmpLTEI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTESOp(*this, operands);
}

OpFoldResult CmpLTEI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTESOp(*this, operands);
}

void CmpLTEI32SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32SOp, CmpGTI32SOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI32SOp, CmpLTI32SOp>>(context);
}

void CmpLTEI64SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI64SOp, CmpGTI64SOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpLTEUOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.ule(b); });
}

OpFoldResult CmpLTEI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEUOp(*this, operands);
}

OpFoldResult CmpLTEI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEUOp(*this, operands);
}

void CmpLTEI32UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32UOp, CmpGTI32UOp>>(context);
  results.insert<RewritePseudoCmpLTEToLT<CmpLTEI32UOp, CmpLTI32UOp>>(context);
}

void CmpLTEI64UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
    rewriter.replaceOpWithNewOp<U>(op, op.getType(), op.rhs(), op.lhs());
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpGTSOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.sgt(b); });
}

OpFoldResult CmpGTI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTSOp(*this, operands);
}

OpFoldResult CmpGTI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTSOp(*this, operands);
}

void CmpGTI32SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32SOp, CmpLTEI32SOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI32SOp, CmpLTI32SOp>>(context);
}

void CmpGTI64SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI64SOp, CmpLTEI64SOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpGTUOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.ugt(b); });
}

OpFoldResult CmpGTI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTUOp(*this, operands);
}

OpFoldResult CmpGTI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTUOp(*this, operands);
}

void CmpGTI32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32UOp, CmpLTEI32UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTI32UOp, CmpLTI32UOp>>(context);
}

void CmpGTI64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
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
    auto condValue =
        rewriter.createOrFold<U>(op.getLoc(), op.getType(), op.lhs(), op.rhs());
    rewriter.replaceOpWithNewOp<XorI32Op>(
        op, op.getType(), condValue,
        rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
    return success();
  }
};

}  // namespace

template <typename T>
static OpFoldResult foldCmpGTESOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.sge(b); });
}

OpFoldResult CmpGTEI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTESOp(*this, operands);
}

OpFoldResult CmpGTEI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTESOp(*this, operands);
}

void CmpGTEI32SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32SOp, CmpLTI32SOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI32SOp, CmpLTI32SOp>>(context);
}

void CmpGTEI64SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI64SOp, CmpLTI64SOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI64SOp, CmpLTI64SOp>>(context);
}

template <typename T>
static OpFoldResult foldCmpGTEUOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) { return a.uge(b); });
}

OpFoldResult CmpGTEI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEUOp(*this, operands);
}

OpFoldResult CmpGTEI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEUOp(*this, operands);
}

void CmpGTEI32UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32UOp, CmpLTI32UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI32UOp, CmpLTI32UOp>>(context);
}

void CmpGTEI64UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI64UOp, CmpLTI64UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEI64UOp, CmpLTI64UOp>>(context);
}

OpFoldResult CmpNZI32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](const APInt &a) { return APInt(32, a.getBoolValue()); });
}

OpFoldResult CmpNZI64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](const APInt &a) { return APInt(64, a.getBoolValue()); });
}

//===----------------------------------------------------------------------===//
// Floating-point comparison
//===----------------------------------------------------------------------===//

enum CmpFOrdering {
  ORDERED = 0,
  UNORDERED = 1,
};

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpEQFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x == x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(
      operands, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpEqual;
        } else {
          return result == APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpEQF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpEQF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpEQF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpEQF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpEQFOp<UNORDERED>(*this, operands);
}

void CmpEQF32OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF32OOp, CmpNEF32OOp>>(context);
}

void CmpEQF64OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF64OOp, CmpNEF64OOp>>(context);
}

void CmpEQF32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF32UOp, CmpNEF32UOp>>(context);
}

void CmpEQF64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQF64UOp, CmpNEF64UOp>>(context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpNEFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x != x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(
      operands, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result != APFloat::cmpEqual;
        } else {
          return result != APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpNEF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpNEF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpNEF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpNEF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpNEFOp<UNORDERED>(*this, operands);
}

void CmpNEF32OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF32OOp, CmpEQF32OOp>>(context);
}

void CmpNEF64OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF64OOp, CmpEQF64OOp>>(context);
}

void CmpNEF32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF32UOp, CmpEQF32UOp>>(context);
}

void CmpNEF64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEF64UOp, CmpEQF64UOp>>(context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpLTFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x < x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(operands, [&](const APFloat &a,
                                                    const APFloat &b) {
    auto result = a.compare(b);
    if (ordering == ORDERED) {
      return result == APFloat::cmpLessThan;
    } else {
      return result == APFloat::cmpLessThan || result == APFloat::cmpUnordered;
    }
  });
}

OpFoldResult CmpLTF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpLTF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpLTF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpLTF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTFOp<UNORDERED>(*this, operands);
}

void CmpLTF32OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

void CmpLTF64OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

void CmpLTF32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

void CmpLTF64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpLTEFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x <= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(
      operands, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpLessThan || result == APFloat::cmpEqual;
        } else {
          return result == APFloat::cmpLessThan ||
                 result == APFloat::cmpEqual || result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpLTEF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpLTEF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpLTEF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpLTEF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpLTEFOp<UNORDERED>(*this, operands);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpGTFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x > x = false
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(
      operands, [&](const APFloat &a, const APFloat &b) {
        auto result = a.compare(b);
        if (ordering == ORDERED) {
          return result == APFloat::cmpGreaterThan;
        } else {
          return result == APFloat::cmpGreaterThan ||
                 result == APFloat::cmpUnordered;
        }
      });
}

OpFoldResult CmpGTF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpGTF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpGTF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpGTF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTFOp<UNORDERED>(*this, operands);
}

void CmpGTF32OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF32OOp, CmpLTEF32OOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF32OOp, CmpLTF32OOp>>(context);
}

void CmpGTF64OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF64OOp, CmpLTEF64OOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF64OOp, CmpLTF64OOp>>(context);
}

void CmpGTF32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF32UOp, CmpLTEF32UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF32UOp, CmpLTF32UOp>>(context);
}

void CmpGTF64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTF64UOp, CmpLTEF64UOp>>(context);
  results.insert<RewritePseudoCmpGTToLT<CmpGTF64UOp, CmpLTF64UOp>>(context);
}

template <CmpFOrdering ordering, typename T>
static OpFoldResult foldCmpGTEFOp(T op, ArrayRef<Attribute> operands) {
  if (op.lhs() == op.rhs()) {
    // x >= x = true
    return oneOfType(op.getType());
  }
  return constFoldBinaryOp<FloatAttr>(operands, [&](const APFloat &a,
                                                    const APFloat &b) {
    auto result = a.compare(b);
    if (ordering == ORDERED) {
      return result == APFloat::cmpGreaterThan || result == APFloat::cmpEqual;
    } else {
      return result == APFloat::cmpGreaterThan || result == APFloat::cmpEqual ||
             result == APFloat::cmpUnordered;
    }
  });
}

OpFoldResult CmpGTEF32OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpGTEF64OOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEFOp<ORDERED>(*this, operands);
}

OpFoldResult CmpGTEF32UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEFOp<UNORDERED>(*this, operands);
}

OpFoldResult CmpGTEF64UOp::fold(ArrayRef<Attribute> operands) {
  return foldCmpGTEFOp<UNORDERED>(*this, operands);
}

void CmpGTEF32OOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF32OOp, CmpLTF32OOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF32OOp, CmpLTF32OOp>>(context);
}

void CmpGTEF64OOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF64OOp, CmpLTF64OOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF64OOp, CmpLTF64OOp>>(context);
}

void CmpGTEF32UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEF32UOp, CmpLTF32UOp>>(context);
  results.insert<RewritePseudoCmpGTEToLT<CmpGTEF32UOp, CmpLTF32UOp>>(context);
}

void CmpGTEF64UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
    rewriter.replaceOpWithNewOp<U>(op, op.getType(), op.operand(),
                                   rewriter.create<CZ>(op.getLoc()));
    return success();
  }
};

}  // namespace

OpFoldResult CmpNZF32OOp::fold(ArrayRef<Attribute> operands) {
  return constFoldCmpOp<FloatAttr>(
      operands, [&](const APFloat &a) { return APInt(32, a.isNonZero()); });
}

OpFoldResult CmpNZF64OOp::fold(ArrayRef<Attribute> operands) {
  return constFoldCmpOp<FloatAttr>(
      operands, [&](const APFloat &a) { return APInt(32, a.isNonZero()); });
}

OpFoldResult CmpNZF32UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldCmpOp<FloatAttr>(operands, [&](const APFloat &a) {
    return APInt(32, a.isNonZero() || a.isNaN());
  });
}

OpFoldResult CmpNZF64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldCmpOp<FloatAttr>(operands, [&](const APFloat &a) {
    return APInt(32, a.isNonZero() || a.isNaN());
  });
}

void CmpNZF32OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF32OOp, CmpNEF32OOp, ConstF32ZeroOp>>(
          context);
}

void CmpNZF64OOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF64OOp, CmpNEF64OOp, ConstF64ZeroOp>>(
          context);
}

void CmpNZF32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF32UOp, CmpNEF32UOp, ConstF32ZeroOp>>(
          context);
}

void CmpNZF64UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results
      .insert<RewritePseudoCmpNZToNE<CmpNZF64UOp, CmpNEF64UOp, ConstF64ZeroOp>>(
          context);
}

OpFoldResult CmpNaNF32Op::fold(ArrayRef<Attribute> operands) {
  if (auto operand = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return operand.getValue().isNaN() ? oneOfType(getType())
                                      : zeroOfType(getType());
  }
  return {};
}

OpFoldResult CmpNaNF64Op::fold(ArrayRef<Attribute> operands) {
  if (auto operand = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return operand.getValue().isNaN() ? oneOfType(getType())
                                      : zeroOfType(getType());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// vm.ref comparison
//===----------------------------------------------------------------------===//

OpFoldResult CmpEQRefOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x == x = true
    return oneOfType(getType());
  } else if (operands[0] && operands[1]) {
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
    if (matchPattern(op.rhs(), m_Constant(&rhs))) {
      auto cmpNz =
          rewriter.create<CmpNZRefOp>(op.getLoc(), op.getType(), op.lhs());
      rewriter.replaceOpWithNewOp<XorI32Op>(
          op, op.getType(), cmpNz,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(op.getLoc(), 1));
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpEQRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<NullCheckCmpEQRefToCmpNZRef>(context);
}

OpFoldResult CmpNERefOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
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
    if (matchPattern(op.rhs(), m_Constant(&rhs))) {
      rewriter.replaceOpWithNewOp<CmpNZRefOp>(op, op.getType(), op.lhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpNERefOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<NullCheckCmpNERefToCmpNZRef>(context);
}

OpFoldResult CmpNZRefOp::fold(ArrayRef<Attribute> operands) {
  Attribute operandValue;
  if (matchPattern(operand(), m_Constant(&operandValue))) {
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

void BranchOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
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
    if (matchPattern(op.condition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getTrueDest(),
                                            op.getTrueOperands());
      return success();
    } else if (matchPattern(op.condition(), m_Zero())) {
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
    //   if (matchPattern(xorOp.rhs(), m_Constant(&rhs)) &&
    //       rhs.cast<IntegerAttr>().getInt() == 1) {
    //     rewriter.replaceOpWithNewOp<CondBranchOp>(
    //         op, xorOp.lhs(), op.getFalseDest(), op.getFalseOperands(),
    //         op.getTrueDest(), op.getTrueOperands());
    //     return success();
    //   }
    // }
    return failure();
  }
};

}  // namespace

void CondBranchOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
        op->template getParentOfType<ModuleOp>(), op.callee());

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

void CallOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
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
    for (const auto &segmentSize : op.segment_sizes()) {
      if (segmentSize.getSExtValue() != -1) {
        return failure();
      }
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, op.callee(),
                                        llvm::to_vector<4>(op.getResultTypes()),
                                        op.getOperands());
    return success();
  }
};

}  // namespace

void CallVariadicOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
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
    auto *failBlock = rewriter.createBlock(block, {op.status().getType()});
    block->moveBefore(failBlock);
    rewriter.setInsertionPointToStart(failBlock);
    rewriter.create<FailOp>(
        op.getLoc(), failBlock->getArgument(0),
        op.message().hasValue() ? op.message().getValue() : "");

    // Replace the original cond_fail with our cond_branch, splitting the block
    // and continuing if the condition is not taken.
    auto *continueBlock = rewriter.splitBlock(
        block, op.getOperation()->getNextNode()->getIterator());
    rewriter.setInsertionPointToEnd(block);
    rewriter.replaceOpWithNewOp<CondBranchOp>(op, op.condition(), failBlock,
                                              ValueRange{op.status()},
                                              continueBlock, ValueRange{});

    return success();
  }
};

}  // namespace

void CondFailOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
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
                                                      op.messageAttr());
    return success();
  }
};

}  // namespace

void CheckEQOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckEQOp, CmpEQI32Op, CmpEQI64Op,
                                        CmpEQF32OOp, CmpEQF64OOp, CmpEQRefOp>>(
      context);
}

void CheckNEOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNEOp, CmpNEI32Op, CmpNEI64Op,
                                        CmpNEF32OOp, CmpNEF64OOp, CmpNERefOp>>(
      context);
}

void CheckNZOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNZOp, CmpNZI32Op, CmpNZI64Op,
                                        CmpNZF32OOp, CmpNZF64OOp, CmpNZRefOp>>(
      context);
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
    if (!matchPattern(op.condition(), m_Constant(&condValue))) {
      return failure();
    }

    if (condValue.getValue() != 0) {
      // True - always break (to the same destination).
      rewriter.replaceOpWithNewOp<BreakOp>(op, op.getDest(), op.destOperands());
    } else {
      // False - skip the break.
      rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDest(),
                                            op.destOperands());
    }
    return success();
  }
};

}  // namespace

void TraceOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugOp<TraceOp>>(context);
}

void PrintOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugOp<PrintOp>>(context);
}

void BreakOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<RemoveDisabledDebugAsyncOp<BreakOp>>(context);
}

void CondBreakOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<RemoveDisabledDebugAsyncOp<CondBreakOp>,
                 SimplifyConstCondBreakPred>(context);
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
