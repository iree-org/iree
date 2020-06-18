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
  switch (type.getKind()) {
    case StandardTypes::BF16:
    case StandardTypes::F16:
    case StandardTypes::F32:
    case StandardTypes::F64:
      return builder.getFloatAttr(type, 1.0);
    case StandardTypes::Integer: {
      auto width = type.cast<IntegerType>().getWidth();
      if (width == 1) return builder.getBoolAttr(true);
      return builder.getIntegerAttr(type, APInt(width, 1));
    }
    case StandardTypes::Vector:
    case StandardTypes::RankedTensor: {
      auto vtType = type.cast<ShapedType>();
      auto element = oneOfType(vtType.getElementType());
      if (!element) return {};
      return DenseElementsAttr::get(vtType, element);
    }
    default:
      break;
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
struct DropDefaultConstGlobalOpInitializer
    : public OpRewritePattern<GlobalI32Op> {
  using OpRewritePattern<GlobalI32Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalI32Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.initial_value().hasValue()) return failure();
    auto value = op.initial_valueAttr().cast<IntegerAttr>();
    if (value.getValue() != 0) return failure();
    rewriter.replaceOpWithNewOp<GlobalI32Op>(
        op, op.sym_name(), op.is_mutable(), op.type(),
        llvm::to_vector<4>(op.getDialectAttrs()));
    return success();
  }
};

}  // namespace

void GlobalI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalI32Op>,
                 DropDefaultConstGlobalOpInitializer>(context);
}

void GlobalRefOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<InlineConstGlobalOpInitializer<GlobalRefOp>>(context);
}

namespace {

/// Inlines immutable global constants into their loads.
struct InlineConstGlobalLoadI32Op : public OpRewritePattern<GlobalLoadI32Op> {
  using OpRewritePattern<GlobalLoadI32Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadI32Op op,
                                PatternRewriter &rewriter) const override {
    auto globalAttr = op.getAttrOfType<FlatSymbolRefAttr>("global");
    auto globalOp =
        op.getParentOfType<VM::ModuleOp>().lookupSymbol<GlobalI32Op>(
            globalAttr.getValue());
    if (!globalOp) return failure();
    if (globalOp.is_mutable()) return failure();
    if (globalOp.initial_value()) {
      rewriter.replaceOpWithNewOp<ConstI32Op>(
          op, globalOp.initial_value().getValue());
    } else {
      rewriter.replaceOpWithNewOp<ConstI32ZeroOp>(op);
    }
    return success();
  }
};

}  // namespace

void GlobalLoadI32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<InlineConstGlobalLoadI32Op>(context);
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

OpFoldResult ConstI32ZeroOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult AddI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x + 0 = x or 0 + y = y (commutative)
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

OpFoldResult SubI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x - 0 = x
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

OpFoldResult MulI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x * 0 = 0 or 0 * y = 0 (commutative)
    return zeroOfType(getType());
  } else if (matchPattern(rhs(), m_One())) {
    // x * 1 = x or 1 * y = y (commutative)
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

OpFoldResult DivI32SOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x / 0 = death
    emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(lhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(getType());
  } else if (matchPattern(rhs(), m_One())) {
    // x / 1 = x
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.sdiv(b); });
}

OpFoldResult DivI32UOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x / 0 = death
    emitOpError() << "is a divide by constant zero";
    return {};
  } else if (matchPattern(lhs(), m_Zero())) {
    // 0 / y = 0
    return zeroOfType(getType());
  } else if (matchPattern(rhs(), m_One())) {
    // x / 1 = x
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.udiv(b); });
}

OpFoldResult RemI32SOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x % 0 = death
    emitOpError() << "is a remainder by constant zero";
    return {};
  } else if (matchPattern(lhs(), m_Zero()) || matchPattern(rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.srem(b); });
}

OpFoldResult RemI32UOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(lhs(), m_Zero()) || matchPattern(rhs(), m_One())) {
    // x % 1 = 0
    // 0 % y = 0
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.urem(b); });
}

OpFoldResult NotI32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](APInt a) {
    a.flipAllBits();
    return a;
  });
}

OpFoldResult AndI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x & 0 = 0 or 0 & y = 0 (commutative)
    return zeroOfType(getType());
  } else if (lhs() == rhs()) {
    // x & x = x
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

OpFoldResult OrI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x | 0 = x or 0 | y = y (commutative)
    return lhs();
  } else if (lhs() == rhs()) {
    // x | x = x
    return lhs();
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

OpFoldResult XorI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(rhs(), m_Zero())) {
    // x ^ 0 = x or 0 ^ y = y (commutative)
    return lhs();
  } else if (lhs() == rhs()) {
    // x ^ x = 0
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===----------------------------------------------------------------------===//

OpFoldResult ShlI32Op::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(operand(), m_Zero())) {
    // 0 << y = 0
    return zeroOfType(getType());
  } else if (amount() == 0) {
    // x << 0 = x
    return operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.shl(amount()); });
}

OpFoldResult ShrI32SOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(operand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(getType());
  } else if (amount() == 0) {
    // x >> 0 = x
    return operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.ashr(amount()); });
}

OpFoldResult ShrI32UOp::fold(ArrayRef<Attribute> operands) {
  if (matchPattern(operand(), m_Zero())) {
    // 0 >> y = 0
    return zeroOfType(getType());
  } else if (amount() == 0) {
    // x >> 0 = x
    return operand();
  }
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.lshr(amount()); });
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

OpFoldResult TruncI8Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.trunc(8).zext(32); });
}

OpFoldResult TruncI16Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.trunc(16).zext(32); });
}

OpFoldResult ExtI8I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.trunc(8).sext(32); });
}

OpFoldResult ExtI16I32SOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return a.trunc(16).sext(32); });
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

OpFoldResult CmpEQI32Op::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x == x = true
    return oneOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.eq(b); });
}

void CmpEQI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpEQI32Op, CmpNEI32Op>>(context);
}

namespace {

/// Changes a cmp.ne.i32 check against 0 to a cmp.nz.i32.
struct CmpNEI32ZeroToCmpNZI32 : public OpRewritePattern<CmpNEI32Op> {
  using OpRewritePattern<CmpNEI32Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(CmpNEI32Op op,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(op.rhs(), m_Zero())) {
      rewriter.replaceOpWithNewOp<CmpNZI32Op>(op, op.getType(), op.lhs());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CmpNEI32Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpNEI32Op, CmpEQI32Op>,
                 CmpNEI32ZeroToCmpNZI32>(context);
}

OpFoldResult CmpNEI32Op::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x != x = false
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.ne(b); });
}

OpFoldResult CmpLTI32SOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x < x = false
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.slt(b); });
}

void CmpLTI32SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTI32SOp, CmpGTEI32SOp>>(context);
}

OpFoldResult CmpLTI32UOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x < x = false
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.ult(b); });
}

void CmpLTI32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTI32UOp, CmpGTEI32UOp>>(context);
}

OpFoldResult CmpLTEI32SOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x <= x = true
    return oneOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.sle(b); });
}

void CmpLTEI32SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32SOp, CmpGTI32SOp>>(context);
}

OpFoldResult CmpLTEI32UOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x <= x = true
    return oneOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.ule(b); });
}

void CmpLTEI32UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpLTEI32UOp, CmpGTI32UOp>>(context);
}

OpFoldResult CmpGTI32SOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x > x = false
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.sgt(b); });
}

void CmpGTI32SOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32SOp, CmpLTEI32SOp>>(context);
}

OpFoldResult CmpGTI32UOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x > x = false
    return zeroOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.ugt(b); });
}

void CmpGTI32UOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTI32UOp, CmpLTEI32UOp>>(context);
}

OpFoldResult CmpGTEI32SOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x >= x = true
    return oneOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.sge(b); });
}

void CmpGTEI32SOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32SOp, CmpLTI32SOp>>(context);
}

OpFoldResult CmpGTEI32UOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) {
    // x >= x = true
    return oneOfType(getType());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [&](APInt a, APInt b) { return a.uge(b); });
}

void CmpGTEI32UOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SwapInvertedCmpOps<CmpGTEI32UOp, CmpLTI32UOp>>(context);
}

OpFoldResult CmpNZI32Op::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(
      operands, [&](APInt a) { return APInt(32, a.getBoolValue()); });
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
template <typename CheckOp, typename CmpI32Op, typename CmpRefOp>
struct RewriteCheckToCondFail : public OpRewritePattern<CheckOp> {
  using OpRewritePattern<CheckOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CheckOp op,
                                PatternRewriter &rewriter) const override {
    Type condType = rewriter.getI32Type();
    Value condValue;
    if (op.getOperation()->getOperand(0).getType().template isa<RefType>()) {
      condValue = rewriter.template createOrFold<CmpRefOp>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands(), ArrayRef<NamedAttribute>{});
    } else {
      condValue = rewriter.template createOrFold<CmpI32Op>(
          op.getLoc(), ArrayRef<Type>{condType},
          op.getOperation()->getOperands(), ArrayRef<NamedAttribute>{});
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
  results.insert<RewriteCheckToCondFail<CheckEQOp, CmpEQI32Op, CmpEQRefOp>>(
      context);
}

void CheckNEOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNEOp, CmpNEI32Op, CmpNERefOp>>(
      context);
}

void CheckNZOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RewriteCheckToCondFail<CheckNZOp, CmpNZI32Op, CmpNZRefOp>>(
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
