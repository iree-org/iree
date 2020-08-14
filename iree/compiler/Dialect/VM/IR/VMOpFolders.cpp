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

#include <algorithm>

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
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
  if (type.isa<FloatType>()) return builder.getFloatAttr(type, 1.0);
  if (auto integerTy = type.dyn_cast<IntegerType>())
    return builder.getIntegerAttr(integerTy, APInt(integerTy.getWidth(), 1));
  if (type.isa<RankedTensorType, VectorType>()) {
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
    auto value = op.initial_valueAttr().template cast<IntegerAttr>();
    if (value.getValue() != 0) return failure();
    rewriter.replaceOpWithNewOp<T>(op, op.sym_name(), op.is_mutable(),
                                   op.type(),
                                   llvm::to_vector<4>(op.getDialectAttrs()));
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
    auto globalAttr = op.template getAttrOfType<FlatSymbolRefAttr>("global");
    auto globalOp =
        op.template getParentOfType<VM::ModuleOp>()
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

namespace {

/// Inlines immutable global constants into their loads.
struct InlineConstGlobalLoadRefOp : public OpRewritePattern<GlobalLoadRefOp> {
  using OpRewritePattern<GlobalLoadRefOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadRefOp op,
                                PatternRewriter &rewriter) const override {
    auto globalAttr = op.getAttrOfType<FlatSymbolRefAttr>("global");
    auto globalOp =
        op.getParentOfType<VM::ModuleOp>().lookupSymbol<GlobalRefOp>(
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
class PropagateGlobalLoadAddress : public OpRewritePattern<INDIRECT> {
  using OpRewritePattern<INDIRECT>::OpRewritePattern;

 public:
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

void GlobalLoadIndirectRefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalLoadAddress<GlobalLoadIndirectRefOp, GlobalLoadRefOp>>(
      context);
}

namespace {

template <typename INDIRECT, typename DIRECT>
class PropagateGlobalStoreAddress : public OpRewritePattern<INDIRECT> {
  using OpRewritePattern<INDIRECT>::OpRewritePattern;

 public:
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

void GlobalStoreIndirectRefOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<
      PropagateGlobalStoreAddress<GlobalStoreIndirectRefOp, GlobalStoreRefOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

OpFoldResult ConstI32Op::fold(ArrayRef<Attribute> operands) { return value(); }

OpFoldResult ConstI64Op::fold(ArrayRef<Attribute> operands) { return value(); }

OpFoldResult ConstI32ZeroOp::fold(ArrayRef<Attribute> operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstI64ZeroOp::fold(ArrayRef<Attribute> operands) {
  return IntegerAttr::get(getResult().getType(), 0);
}

OpFoldResult ConstRefZeroOp::fold(ArrayRef<Attribute> operands) {
  // TODO(b/144027097): relace unit attr with a proper null ref_ptr attr.
  return UnitAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// ref_ptr operations
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

OpFoldResult SwitchRefOp::fold(ArrayRef<Attribute> operands) {
  return foldSwitchOp(*this);
}

//===----------------------------------------------------------------------===//
// Native integer arithmetic
//===----------------------------------------------------------------------===//

namespace {

/// Performs const folding `calculate` with element-wise behavior on the given
/// attribute in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<ElementValueT(ElementValueT)>>
Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
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

/// Performs const folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              std::function<ElementValueT(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
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

}  // namespace

template <typename T>
static OpFoldResult foldAddOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x + 0 = x or 0 + y = y (commutative)
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

OpFoldResult AddI32Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp(*this, operands);
}

OpFoldResult AddI64Op::fold(ArrayRef<Attribute> operands) {
  return foldAddOp(*this, operands);
}

template <typename T>
static OpFoldResult foldSubOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x - 0 = x
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

OpFoldResult SubI32Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp(*this, operands);
}

OpFoldResult SubI64Op::fold(ArrayRef<Attribute> operands) {
  return foldSubOp(*this, operands);
}

template <typename T>
static OpFoldResult foldMulOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.rhs(), m_Zero())) {
    // x * 0 = 0 or 0 * y = 0 (commutative)
    return zeroOfType(op.getType());
  } else if (matchPattern(op.rhs(), m_One())) {
    // x * 1 = x or 1 * y = y (commutative)
    return op.lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

OpFoldResult MulI32Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp(*this, operands);
}

OpFoldResult MulI64Op::fold(ArrayRef<Attribute> operands) {
  return foldMulOp(*this, operands);
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
      operands, [](APInt a, APInt b) { return a.sdiv(b); });
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
      operands, [](APInt a, APInt b) { return a.udiv(b); });
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
      operands, [](APInt a, APInt b) { return a.srem(b); });
}

OpFoldResult RemI32SOp::fold(ArrayRef<Attribute> operands) {
  return foldRemSOp(*this, operands);
}

OpFoldResult RemI64SOp::fold(ArrayRef<Attribute> operands) {
  return foldRemSOp(*this, operands);
}

template <typename T>
static OpFoldResult foldRemUOp(T op, ArrayRef<Attribute> operands) {
  if (matchPattern(op.lhs(), m_Zero()) || matchPattern(op.rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(op.getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.urem(b); });
}

OpFoldResult RemI32UOp::fold(ArrayRef<Attribute> operands) {
  return foldRemUOp(*this, operands);
}

OpFoldResult RemI64UOp::fold(ArrayRef<Attribute> operands) {
  return foldRemUOp(*this, operands);
}

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
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
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
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
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
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
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
  } else if (op.amount() == 0) {
    // x << 0 = x
    return op.operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.shl(op.amount()); });
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
  } else if (op.amount() == 0) {
    // x >> 0 = x
    return op.operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.ashr(op.amount()); });
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
  } else if (op.amount() == 0) {
    // x >> 0 = x
    return op.operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.lshr(op.amount()); });
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
Attribute constFoldConversionOp(Type resultType, ArrayRef<Attribute> operands,
                                const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto operand = operands[0].dyn_cast_or_null<AttrElementT>()) {
    return AttrElementT::get(resultType, calculate(operand.getValue()));
  }
  return {};
}

OpFoldResult TruncI32I8Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI32I16Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I8Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI64I16Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(16).zext(32); });
}

OpFoldResult TruncI64I32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(32); });
}

OpFoldResult ExtI8I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(8).sext(32); });
}

OpFoldResult ExtI8I32UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(8).zext(32); });
}

OpFoldResult ExtI16I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(16).sext(32); });
}

OpFoldResult ExtI16I32UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(32, getContext()), operands,
      [&](APInt a) { return a.trunc(16).zext(32); });
}

OpFoldResult ExtI8I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.trunc(8).sext(64); });
}

OpFoldResult ExtI8I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.trunc(8).zext(64); });
}

OpFoldResult ExtI16I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.trunc(16).sext(64); });
}

OpFoldResult ExtI16I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.trunc(16).zext(64); });
}

OpFoldResult ExtI32I64SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.sext(64); });
}

OpFoldResult ExtI32I64UOp::fold(ArrayRef<Attribute> operands) {
  return constFoldConversionOp<IntegerAttr>(
      IntegerType::get(64, getContext()), operands,
      [&](APInt a) { return a.zext(64); });
}

namespace {

template <typename SRC_OP, typename OP_A, int SZ_T, typename OP_B>
class PseudoIntegerConversionToSplitConversionOp
    : public OpRewritePattern<SRC_OP> {
  using OpRewritePattern<SRC_OP>::OpRewritePattern;

 public:
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

//===----------------------------------------------------------------------===//
// Native reduction (horizontal) arithmetic
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

namespace {

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
      operands, [&](APInt a, APInt b) { return a.eq(b); });
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
      operands, [&](APInt a, APInt b) { return a.ne(b); });
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
      operands, [&](APInt a, APInt b) { return a.slt(b); });
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
      operands, [&](APInt a, APInt b) { return a.ult(b); });
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
      operands, [&](APInt a, APInt b) { return a.sle(b); });
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
      operands, [&](APInt a, APInt b) { return a.ule(b); });
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
      operands, [&](APInt a, APInt b) { return a.sgt(b); });
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
      operands, [&](APInt a, APInt b) { return a.ugt(b); });
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
      operands, [&](APInt a, APInt b) { return a.sge(b); });
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
      operands, [&](APInt a, APInt b) { return a.uge(b); });
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
      operands, [&](APInt a) { return APInt(32, a.getBoolValue()); });
}

OpFoldResult CmpNZI64Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return APInt(64, a.getBoolValue()); });
}

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
        op.template getParentOfType<ModuleOp>(), op.callee());

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
          typename CmpRefOp>
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
    } else if (operandType.isInteger(64)) {
      condValue = rewriter.template createOrFold<CmpI64Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands());
    } else if (operandType.isInteger(32)) {
      condValue = rewriter.template createOrFold<CmpI32Op>(
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
  results.insert<
      RewriteCheckToCondFail<CheckEQOp, CmpEQI32Op, CmpEQI64Op, CmpEQRefOp>>(
      context);
}

void CheckNEOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<
      RewriteCheckToCondFail<CheckNEOp, CmpNEI32Op, CmpNEI64Op, CmpNERefOp>>(
      context);
}

void CheckNZOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<
      RewriteCheckToCondFail<CheckNZOp, CmpNZI32Op, CmpNZI64Op, CmpNZRefOp>>(
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
