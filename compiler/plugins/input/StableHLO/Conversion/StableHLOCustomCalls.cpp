// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowers supported StableHLO custom calls and preserves runtime shape
// assertions. Assertion analysis and IR materialization have separate mutation
// boundaries.

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/StableHLOStatusCheck.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCUSTOMCALLS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// Computes householder using vector `v` and a `tau` matrice
// with the k-th element. See householder transformation as below:
// https://en.wikipedia.org/wiki/Householder_transformation
static Value computeHouseholder(Value v, Value tau, Value k,
                                ImplicitLocOpBuilder b) {
  auto vTy = cast<ShapedType>(v.getType());

  SmallVector<int64_t> hShape(vTy.getShape());
  hShape.push_back(hShape.back());

  auto hTy = RankedTensorType::get(hShape, vTy.getElementType());
  Value empty = tensor::EmptyOp::create(b, hShape, vTy.getElementType());

  auto outMap = b.getMultiDimIdentityMap(hShape.size());

  SmallVector<AffineExpr> exprs;
  for (int i = 0, s = vTy.getRank(); i < s; ++i) {
    exprs.push_back(b.getAffineDimExpr(i));
  }

  AffineMap vMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  exprs.back() = b.getAffineDimExpr(vTy.getRank());
  AffineMap vTMap = AffineMap::get(hTy.getRank(), 0, exprs, b.getContext());

  SmallVector<AffineMap> affineMaps = {vMap, vTMap, outMap};

  SmallVector<utils::IteratorType> iterTypes(hShape.size(),
                                             utils::IteratorType::parallel);

  Value zero =
      arith::ConstantOp::create(b, b.getZeroAttr(vTy.getElementType()));
  Value one =
      arith::ConstantOp::create(b, b.getFloatAttr(vTy.getElementType(), 1.0));

  return linalg::GenericOp::create(
             b, hTy, ValueRange{v, v}, empty, affineMaps, iterTypes,
             [&](OpBuilder &bb, Location loc, ValueRange args) {
               ImplicitLocOpBuilder b(loc, bb);
               SmallVector<Value> indices;
               for (int i = 0, s = hTy.getRank(); i < s; ++i) {
                 indices.push_back(linalg::IndexOp::create(b, loc, i));
               }

               SmallVector<Value> tauIndices(indices.begin(),
                                             indices.end() - 2);
               tauIndices.push_back(k);
               Value t = tensor::ExtractOp::create(b, tau, tauIndices);

               // Generates the lower triangularization of the matrix with
               // one values on the diagonal.
               auto tri = [&](Value v, Value i) {
                 Value eq =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::eq, i, k);
                 Value lt =
                     arith::CmpIOp::create(b, arith::CmpIPredicate::ult, i, k);
                 Value sel = arith::SelectOp::create(b, eq, one, v);
                 return arith::SelectOp::create(b, lt, zero, sel);
               };

               Value v = tri(args[0], indices[indices.size() - 2]);
               Value vT = tri(args[1], indices[indices.size() - 1]);

               Value h = arith::MulFOp::create(b, v, vT);
               h = arith::MulFOp::create(b, h, t);

               Value isDiag = arith::CmpIOp::create(
                   b, arith::CmpIPredicate::eq, indices[indices.size() - 2],
                   indices[indices.size() - 1]);
               Value diag = arith::SelectOp::create(b, isDiag, one, zero);
               Value sub = arith::SubFOp::create(b, diag, h);

               linalg::YieldOp::create(b, sub);
             })
      .getResult(0);
}

// Slices the k-th column of matrix and computes the householder transformation
// for using the `tau` value.
static Value computeHouseholderSlice(Value matrix, Value tau, Value k,
                                     ImplicitLocOpBuilder b) {
  auto matrixTy = cast<ShapedType>(matrix.getType());
  int rank = matrixTy.getRank();

  SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
  SmallVector<int64_t> vShape(matrixTy.getShape());
  vShape[vShape.size() - 1] = 1;

  SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
  vOffsets[vOffsets.size() - 1] = k;

  SmallVector<OpFoldResult> vSizes;
  for (auto v : vShape) {
    vSizes.push_back(b.getIndexAttr(v));
  }

  auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
  Value v = tensor::ExtractSliceOp::create(b, sliceTy, matrix, vOffsets, vSizes,
                                           vStrides);

  SmallVector<ReassociationIndices> reass;
  for (int i = 0; i < rank - 2; ++i) {
    reass.push_back({i});
  }
  reass.push_back({rank - 2, rank - 1});

  ArrayRef<int64_t> collapseVShape(vShape.begin(), vShape.end() - 1);
  auto collapseVTy =
      RankedTensorType::get(collapseVShape, matrixTy.getElementType());
  Value collapseV = tensor::CollapseShapeOp::create(b, collapseVTy, v, reass);

  Value householder = computeHouseholder(collapseV, tau, k, b);
  return householder;
}

struct HouseholderReflectorRewriter final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  using OpAdaptor = mlir::stablehlo::CustomCallOp::Adaptor;

  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getCallTargetName() != "ProductOfElementaryHouseholderReflectors") {
      return rewriter.notifyMatchFailure(
          op, "not ProductOfElementaryHouseholderReflectors");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto matrix = op.getOperand(0);
    auto tau = op.getOperand(1);
    auto matrixTy = cast<ShapedType>(matrix.getType());
    auto tauTy = cast<ShapedType>(tau.getType());
    auto rank = matrixTy.getRank();

    if (isa<ComplexType>(matrixTy.getElementType())) {
      return rewriter.notifyMatchFailure(op, "complex types not supported");
    }

    if (rank < 2) {
      return rewriter.notifyMatchFailure(op, "requires minimum rank 2 matrix");
    }

    // Implementation needs to be checked to work with variable dimension
    // lengths. Should be relatively straightforward.
    if (!matrixTy.hasStaticShape() || !tauTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "not supported for dynamic shapes");
    }

    Value zero = arith::ConstantIndexOp::create(b, 0);
    Value one = arith::ConstantIndexOp::create(b, 1);
    Value k = arith::ConstantIndexOp::create(b, tauTy.getShape().back());
    Value householder0 = computeHouseholderSlice(matrix, tau, zero, b);
    auto scf = scf::ForOp::create(
        b, one, k, one, ValueRange{householder0},
        [&](OpBuilder &bb, Location loc, Value iv, ValueRange args) {
          ImplicitLocOpBuilder b(loc, bb);
          Value householder = computeHouseholderSlice(matrix, tau, iv, b);

          std::vector<int64_t> batch(rank - 2);
          for (int i = 0; i < rank - 2; ++i) {
            batch[i] = i;
          }
          std::vector<int64_t> lhsContract = {rank - 1};
          std::vector<int64_t> rhsContract = {rank - 2};

          auto dotNums = mlir::stablehlo::DotDimensionNumbersAttr::get(
              b.getContext(), batch, batch, lhsContract, rhsContract);
          Value dot = mlir::stablehlo::DotGeneralOp::create(
              b, householder0.getType(), args[0], householder, dotNums, nullptr,
              mlir::stablehlo::DotAlgorithmAttr{});
          scf::YieldOp::create(b, loc, dot);
        });

    SmallVector<OpFoldResult> vOffsets(rank, b.getIndexAttr(0));
    SmallVector<OpFoldResult> vStrides(rank, b.getIndexAttr(1));
    SmallVector<int64_t> vShape(matrixTy.getShape());
    SmallVector<OpFoldResult> vSizes;
    for (auto v : vShape) {
      vSizes.push_back(b.getIndexAttr(v));
    }

    auto sliceTy = RankedTensorType::get(vShape, matrixTy.getElementType());
    Value v = tensor::ExtractSliceOp::create(b, sliceTy, scf.getResult(0),
                                             vOffsets, vSizes, vStrides);

    rewriter.replaceOp(op, v);
    return success();
  }
};

// A signed widening of an i32 dimension size; JAX with 64-bit types computes
// on sizes widened this way, and a size widens exactly.
static bool isWidenedSize(mlir::stablehlo::ConvertOp convert) {
  auto wideType = dyn_cast<IntegerType>(getElementTypeOrSelf(convert));
  return wideType && !wideType.isUnsigned() && wideType.getWidth() > 32 &&
         convert.getOperand()
             .getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
}

namespace shape_assertions {

struct DimensionKey {
  Value tensor;
  int64_t dimension;

  bool operator==(const DimensionKey &other) const {
    return tensor == other.tensor && dimension == other.dimension;
  }
};

struct DimensionSize {
  DimensionKey key;
  mlir::stablehlo::GetDimensionSizeOp size;
};

// An unsigned range of a dimension and a divisor of it; a missing entry is
// unbounded.
struct DimensionRange {
  std::optional<int64_t> umin;
  std::optional<int64_t> umax;
  std::optional<int64_t> udiv;
};

// `scale * source + offset`.
struct DimensionExpression {
  DimensionSize source;
  int64_t scale;
  int64_t offset;
};

// Two dimensions an assertion proves equal, one it bounds, or one it proves
// equal to an expression of another.
struct AssertionFact {
  SmallVector<DimensionSize, 2> dims;
  DimensionRange range;
  std::optional<DimensionExpression> expression;
  mlir::stablehlo::CustomCallOp assertion;
};

struct BlockAssertionFacts {
  Block *block;
  SmallVector<AssertionFact> facts;
};

struct Interval {
  int64_t lo;
  int64_t hi;
};
using IntervalCache = llvm::DenseMap<Value, std::optional<Interval>>;

struct Divisibility {
  DimensionSize dim;
  int64_t divisor;
};

// scale * dimension / divisor + offset over mathematical integers. The
// divisor is known to divide the dimension. Finite-width computations are
// congruent to this expression; each consumer checks its required exact range.
struct AffineSizeExpression {
  DimensionSize dim;
  int64_t scale;
  int64_t offset;
  int64_t low;
  int64_t high;
  int64_t divisor;
};

struct DimensionClassFacts {
  SmallVector<DimensionKey> members;
  SmallVector<mlir::stablehlo::GetDimensionSizeOp> sizeQueries;
  DimensionRange range;
};

struct ExpressionTieFact {
  DimensionSize target;
  DimensionExpression expression;
  // Stable index into the owning block result's classes; absent sources use
  // their original dimension query rather than an inactive singleton class.
  std::optional<unsigned> sourceClass;
};

// Owns fact containers but borrows original IR until immediate materialization.
// No mutable proof caches or use pointers escape into this result.
struct BlockShapeAssertionFacts {
  Operation *lastAssertion;
  SmallVector<DimensionClassFacts> classes;
  SmallVector<ExpressionTieFact> expressions;
};

static bool hasUsableAssumption(const DimensionRange &range) {
  return (range.umin || range.umax || range.udiv) &&
         !(range.umin && range.umax && *range.umin > *range.umax);
}

static BlockShapeAssertionFacts
mergeBlockFacts(const BlockAssertionFacts &block);

static std::optional<DimensionSize> matchDimensionSize(Value value,
                                                       Block *block) {
  if (auto widen = value.getDefiningOp<mlir::stablehlo::ConvertOp>();
      widen && widen->getBlock() == block && isWidenedSize(widen)) {
    value = widen.getOperand();
  }
  auto sizeOp = value.getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
  if (!sizeOp || sizeOp->getBlock() != block) {
    return std::nullopt;
  }
  auto sizeType = dyn_cast<RankedTensorType>(sizeOp.getType());
  auto tensorType = dyn_cast<RankedTensorType>(sizeOp.getOperand().getType());
  int64_t dim = sizeOp.getDimension();
  if (!sizeType || sizeType.getRank() != 0 ||
      !sizeType.getElementType().isInteger(32) || !tensorType || dim < 0 ||
      dim >= tensorType.getRank() || !tensorType.isDynamicDim(dim)) {
    return std::nullopt;
  }
  return DimensionSize{{sizeOp.getOperand(), dim}, sizeOp};
}

static std::optional<int64_t> matchIntegerConstant(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant))) {
    return std::nullopt;
  }
  if (!constant.isSignedIntN(64)) {
    return std::nullopt;
  }
  // An i1 true is 1, not -1.
  return constant.getBitWidth() == 1 ? constant.getZExtValue()
                                     : constant.getSExtValue();
}
// Evaluates predicates over one dimension without modifying IR. Caches belong
// to a single block/range evaluation; zero and positive evaluations share only
// the dimension identity, never their inferred intervals.
class ShapePredicateEvaluator {
public:
  Value peelDecidedValue(Value value, Block *block) {
    for (;;) {
      Operation *op = value.getDefiningOp();
      if (!op || op->getBlock() != block) {
        return value;
      }
      if (auto convert = dyn_cast<mlir::stablehlo::ConvertOp>(op);
          convert && convert.getOperand().getType() == convert.getType()) {
        value = convert.getOperand();
        continue;
      }
      auto select = dyn_cast<mlir::stablehlo::SelectOp>(op);
      if (!select) {
        return value;
      }
      std::optional<bool> taken = decidePredicate(select.getPred(), block);
      if (!taken) {
        return value;
      }
      value = *taken ? select.getOnTrue() : select.getOnFalse();
    }
  }

private:
  struct EvaluationContext {
    Block *block;
    Interval sizes;
    std::optional<DimensionKey> &dimension;
    IntervalCache cache;
  };
  std::optional<bool> decidePredicate(Value predicate, Block *block) {
    std::optional<DimensionKey> key;
    EvaluationContext zeroContext{block, {0, 0}, key, {}};
    EvaluationContext positiveContext{
        block, {1, std::numeric_limits<int32_t>::max()}, key, {}};
    std::optional<Interval> zero = evaluate(predicate, zeroContext, 0);
    std::optional<Interval> positive = evaluate(predicate, positiveContext, 0);
    if (!zero || !positive || zero->lo != zero->hi ||
        positive->lo != positive->hi || zero->lo != positive->lo) {
      return std::nullopt;
    }
    return zero->lo != 0;
  }

  std::optional<Interval> evaluate(Value value, EvaluationContext &context,
                                   int depth) {
    if (auto it = context.cache.find(value); it != context.cache.end()) {
      return it->second;
    }
    auto result = evaluateUncached(value, context, depth);
    context.cache[value] = result;
    return result;
  }

  std::optional<Interval>
  evaluateUncached(Value value, EvaluationContext &context, int depth) {
    Block *block = context.block;
    Interval sizes = context.sizes;
    auto &key = context.dimension;
    if (std::optional<int64_t> constant = matchIntegerConstant(value)) {
      return Interval{*constant, *constant};
    }
    if (std::optional<DimensionSize> dim = matchDimensionSize(value, block)) {
      if (key && !(*key == dim->key)) {
        return std::nullopt;
      }
      key = dim->key;
      return sizes;
    }
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != block || depth > 32) {
      return std::nullopt;
    }
    auto operand = [&](unsigned i) {
      return evaluate(op->getOperand(i), context, depth + 1);
    };
    auto isI1 = [](Value v) { return getElementTypeOrSelf(v).isInteger(1); };
    if (auto convert = dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
      if (convert.getOperand().getType() != convert.getType()) {
        return std::nullopt;
      }
      return operand(0);
    }
    if (isa<mlir::stablehlo::SignOp>(op)) {
      std::optional<Interval> x = operand(0);
      if (!x) {
        return std::nullopt;
      }
      auto sign = [](int64_t v) -> int64_t { return (v > 0) - (v < 0); };
      return Interval{sign(x->lo), sign(x->hi)};
    }
    if (isa<mlir::stablehlo::NotOp>(op) && isI1(value)) {
      std::optional<Interval> x = operand(0);
      if (!x) {
        return std::nullopt;
      }
      return Interval{1 - x->hi, 1 - x->lo};
    }
    if (auto select = dyn_cast<mlir::stablehlo::SelectOp>(op)) {
      std::optional<Interval> pred = operand(0);
      std::optional<Interval> onTrue = operand(1);
      std::optional<Interval> onFalse = operand(2);
      if (!pred || !onTrue || !onFalse) {
        return std::nullopt;
      }
      if (pred->lo == pred->hi) {
        return pred->lo ? onTrue : onFalse;
      }
      return Interval{std::min(onTrue->lo, onFalse->lo),
                      std::max(onTrue->hi, onFalse->hi)};
    }
    if (op->getNumOperands() != 2) {
      return std::nullopt;
    }
    auto lhs = operand(0);
    auto rhs = operand(1);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return evaluateBinary(op, value, *lhs, *rhs);
  }

  static std::optional<Interval> evaluateBinary(Operation *op, Value value,
                                                Interval lhs, Interval rhs) {
    auto isI1 = [](Value v) { return getElementTypeOrSelf(v).isInteger(1); };
    if (isI1(value) && isa<mlir::stablehlo::AndOp, mlir::stablehlo::OrOp>(op)) {
      if (isa<mlir::stablehlo::AndOp>(op)) {
        return Interval{lhs.lo & rhs.lo, lhs.hi & rhs.hi};
      }
      return Interval{lhs.lo | rhs.lo, lhs.hi | rhs.hi};
    }
    // Arithmetic on i1 is logical, not modelled here.
    if (isI1(value) && !isa<mlir::stablehlo::CompareOp>(op)) {
      return std::nullopt;
    }
    if (isa<mlir::stablehlo::DivOp, mlir::stablehlo::RemOp>(op)) {
      // Only non-negative values by a positive constant.
      if (rhs.lo != rhs.hi || rhs.lo <= 0 || lhs.lo < 0) {
        return std::nullopt;
      }
      if (isa<mlir::stablehlo::DivOp>(op)) {
        return Interval{lhs.lo / rhs.lo, lhs.hi / rhs.lo};
      }
      return lhs.hi < rhs.lo ? lhs : Interval{0, rhs.lo - 1};
    }
    if (isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
            mlir::stablehlo::MulOp>(op)) {
      return evaluateArithmetic(
          op, lhs, rhs, getElementTypeOrSelf(value).getIntOrFloatBitWidth());
    }
    return evaluateComparison(op, lhs, rhs);
  }

  static std::optional<Interval> evaluateArithmetic(Operation *op, Interval lhs,
                                                    Interval rhs,
                                                    unsigned width) {
    // Fails where the element type would wrap.
    auto fits = [&](Interval x) -> std::optional<Interval> {
      if (width >= 64 || (x.lo >= -(int64_t{1} << (width - 1)) &&
                          x.hi < (int64_t{1} << (width - 1)))) {
        return x;
      }
      return std::nullopt;
    };
    if (isa<mlir::stablehlo::AddOp>(op)) {
      auto low = llvm::checkedAdd(lhs.lo, rhs.lo);
      auto high = llvm::checkedAdd(lhs.hi, rhs.hi);
      if (!low || !high) {
        return std::nullopt;
      }
      return fits(Interval{*low, *high});
    }
    if (isa<mlir::stablehlo::SubtractOp>(op)) {
      auto low = llvm::checkedSub(lhs.lo, rhs.hi);
      auto high = llvm::checkedSub(lhs.hi, rhs.lo);
      if (!low || !high) {
        return std::nullopt;
      }
      return fits(Interval{*low, *high});
    }
    assert(isa<mlir::stablehlo::MulOp>(op) && "expected interval arithmetic");
    auto lowLow = llvm::checkedMul(lhs.lo, rhs.lo);
    auto lowHigh = llvm::checkedMul(lhs.lo, rhs.hi);
    auto highLow = llvm::checkedMul(lhs.hi, rhs.lo);
    auto highHigh = llvm::checkedMul(lhs.hi, rhs.hi);
    if (!lowLow || !lowHigh || !highLow || !highHigh) {
      return std::nullopt;
    }
    return fits(Interval{std::min({*lowLow, *lowHigh, *highLow, *highHigh}),
                         std::max({*lowLow, *lowHigh, *highLow, *highHigh})});
  }

  static std::optional<Interval> evaluateComparison(Operation *op, Interval lhs,
                                                    Interval rhs) {
    auto compare = dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compare) {
      return std::nullopt;
    }
    auto compareType = compare.getCompareType();
    if (compareType &&
        *compareType == mlir::stablehlo::ComparisonType::UNSIGNED &&
        (lhs.lo < 0 || rhs.lo < 0)) {
      return std::nullopt;
    }
    if (compareType &&
        *compareType != mlir::stablehlo::ComparisonType::SIGNED &&
        *compareType != mlir::stablehlo::ComparisonType::UNSIGNED &&
        *compareType != mlir::stablehlo::ComparisonType::NOTYPE) {
      return std::nullopt;
    }
    // Negative unsigned operands were declined above, so the ordering is
    // identical to signed ordering here. Use MLIR's APInt-backed predicate
    // evaluation without inventing no-wrap assumptions for arithmetic.
    intrange::CmpPredicate predicate;
    switch (compare.getComparisonDirection()) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      predicate = intrange::CmpPredicate::eq;
      break;
    case mlir::stablehlo::ComparisonDirection::NE:
      predicate = intrange::CmpPredicate::ne;
      break;
    case mlir::stablehlo::ComparisonDirection::LT:
      predicate = intrange::CmpPredicate::slt;
      break;
    case mlir::stablehlo::ComparisonDirection::LE:
      predicate = intrange::CmpPredicate::sle;
      break;
    case mlir::stablehlo::ComparisonDirection::GT:
      predicate = intrange::CmpPredicate::sgt;
      break;
    case mlir::stablehlo::ComparisonDirection::GE:
      predicate = intrange::CmpPredicate::sge;
      break;
    }
    auto asRange = [](Interval interval) {
      return ConstantIntRanges::fromSigned(APInt(64, interval.lo, true),
                                           APInt(64, interval.hi, true));
    };
    std::optional<bool> result =
        intrange::evaluatePred(predicate, asRange(lhs), asRange(rhs));
    if (!result) {
      return Interval{0, 1};
    }
    return Interval{*result, *result};
  }
};

// Collects and merges assertion facts. All mutation, dominance-based use
// replacement, and attribute/operation construction belong to materialization.
class ShapeAssertionFactAnalysis {
public:
  SmallVector<BlockShapeAssertionFacts, 0> run(FunctionOpInterface function) {
    collectDivisibility(function);
    SmallVector<BlockShapeAssertionFacts, 0> results;
    for (const BlockAssertionFacts &block : collectFacts(function)) {
      results.push_back(mergeBlockFacts(block));
    }
    return results;
  }

private:
  ShapePredicateEvaluator predicates;
  llvm::DenseMap<Block *, SmallVector<Divisibility>> divisibility;
  std::optional<Divisibility>
  matchDivisibility(mlir::stablehlo::CompareOp compare, Block *block) {
    if (compare.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::EQ) {
      return std::nullopt;
    }
    Value remainder = compare.getLhs();
    std::optional<int64_t> zero = matchIntegerConstant(compare.getRhs());
    if (!zero) {
      remainder = compare.getRhs();
      zero = matchIntegerConstant(compare.getLhs());
    }
    if (!zero || *zero != 0) {
      return std::nullopt;
    }
    auto rem = predicates.peelDecidedValue(remainder, block)
                   .getDefiningOp<mlir::stablehlo::RemOp>();
    if (!rem || rem->getBlock() != block) {
      return std::nullopt;
    }
    std::optional<int64_t> divisor = matchIntegerConstant(rem.getRhs());
    std::optional<DimensionSize> dim = matchDimensionSize(
        predicates.peelDecidedValue(rem.getLhs(), block), block);
    if (!divisor || *divisor <= 0 || !dim) {
      return std::nullopt;
    }
    return Divisibility{*dim, *divisor};
  }
  bool isDivisible(DimensionKey key, int64_t divisor, Block *block) {
    return llvm::any_of(divisibility.lookup(block), [&](const Divisibility &d) {
      return d.dim.key == key && d.divisor % divisor == 0;
    });
  }

  // Iterative recognition of constant arithmetic steps avoids native-stack
  // growth with expression depth. Checked accumulation declines unsafe proofs.
  std::optional<AffineSizeExpression> matchAffineExpression(Value value,
                                                            Block *block) {
    // `factor * term + addend` for one step.
    struct Step {
      int64_t factor;
      int64_t addend;
    };
    SmallVector<Step> steps;
    std::optional<DimensionSize> dim;
    int64_t divisor = 1;
    while (!(dim = matchDimensionSize(
                 value = predicates.peelDecidedValue(value, block), block))) {
      Operation *op = value.getDefiningOp();
      if (!op || op->getBlock() != block) {
        return std::nullopt;
      }
      // A quotient of the size by a constant an assertion proves divides it.
      if (auto divide = dyn_cast<mlir::stablehlo::DivOp>(op)) {
        std::optional<int64_t> constant = matchIntegerConstant(divide.getRhs());
        dim = matchDimensionSize(
            predicates.peelDecidedValue(divide.getLhs(), block), block);
        if (!constant || *constant <= 0 || !dim ||
            !isDivisible(dim->key, *constant, block)) {
          return std::nullopt;
        }
        divisor = *constant;
        break;
      }
      if (op->getNumOperands() != 2) {
        return std::nullopt;
      }
      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);
      std::optional<int64_t> lhsConstant = matchIntegerConstant(lhs);
      std::optional<int64_t> rhsConstant = matchIntegerConstant(rhs);
      if (lhsConstant.has_value() == rhsConstant.has_value()) {
        return std::nullopt;
      }
      int64_t constant = lhsConstant ? *lhsConstant : *rhsConstant;
      if (isa<mlir::stablehlo::AddOp>(op)) {
        steps.push_back({1, constant});
      } else if (isa<mlir::stablehlo::SubtractOp>(op)) {
        if (lhsConstant) {
          steps.push_back({-1, constant});
        } else {
          auto negated = llvm::checkedSub(int64_t{0}, constant);
          if (!negated) {
            return std::nullopt;
          }
          steps.push_back({1, *negated});
        }
      } else if (isa<mlir::stablehlo::MulOp>(op) && constant != 0) {
        steps.push_back({constant, 0});
      } else {
        return std::nullopt;
      }
      value = lhsConstant ? rhs : lhs;
    }
    AffineSizeExpression affine{
        *dim, 1, 0, 0, std::numeric_limits<int32_t>::max() / divisor, divisor};
    for (const Step &step : llvm::reverse(steps)) {
      // Multiplies by the factor, then adds the addend, failing on int64
      // overflow.
      std::optional<int64_t> scale =
          llvm::checkedMul(affine.scale, step.factor);
      std::optional<int64_t> offset =
          llvm::checkedMul(affine.offset, step.factor);
      std::optional<int64_t> low = llvm::checkedMul(affine.low, step.factor);
      std::optional<int64_t> high = llvm::checkedMul(affine.high, step.factor);
      if (!scale || !offset || !low || !high) {
        return std::nullopt;
      }
      if (step.factor < 0) {
        std::swap(low, high);
      }
      offset = llvm::checkedAdd(*offset, step.addend);
      low = llvm::checkedAdd(*low, step.addend);
      high = llvm::checkedAdd(*high, step.addend);
      if (!offset || !low || !high) {
        return std::nullopt;
      }
      // Keeps scale / divisor in lowest terms, so an integer multiple of the
      // quotient reads as a multiple of the size.
      // std::gcd requires the signed magnitude to be representable.
      if (affine.divisor != 1 &&
          *scale == std::numeric_limits<int64_t>::min()) {
        return std::nullopt;
      }
      int64_t common =
          affine.divisor == 1 ? 1 : std::gcd(*scale, affine.divisor);
      affine = AffineSizeExpression{affine.dim, *scale / common,
                                    *offset,    *low,
                                    *high,      affine.divisor / common};
    }
    return affine;
  }

  std::optional<AssertionFact> matchBound(mlir::stablehlo::CompareOp compare,
                                          Block *block) {
    using mlir::stablehlo::ComparisonDirection;
    // The direction once both sides swap or change sign.
    auto mirror = [](ComparisonDirection direction) {
      switch (direction) {
      case ComparisonDirection::GE:
        return ComparisonDirection::LE;
      case ComparisonDirection::GT:
        return ComparisonDirection::LT;
      case ComparisonDirection::LE:
        return ComparisonDirection::GE;
      case ComparisonDirection::LT:
        return ComparisonDirection::GT;
      default:
        return direction;
      }
    };
    std::optional<mlir::stablehlo::ComparisonType> compareType =
        compare.getCompareType();
    if (compareType &&
        *compareType != mlir::stablehlo::ComparisonType::SIGNED &&
        *compareType != mlir::stablehlo::ComparisonType::NOTYPE) {
      return std::nullopt;
    }
    ComparisonDirection direction = compare.getComparisonDirection();
    std::optional<int64_t> limit = matchIntegerConstant(compare.getRhs());
    std::optional<AffineSizeExpression> affine =
        matchAffineExpression(compare.getLhs(), block);
    if (!limit) {
      // Puts the constant on the right.
      limit = matchIntegerConstant(compare.getLhs());
      affine = matchAffineExpression(compare.getRhs(), block);
      direction = mirror(direction);
    }
    // Bounds read the i32 value as exact, so it must not wrap.
    if (!limit || !affine || (affine->scale != 1 && affine->scale != -1) ||
        affine->low < std::numeric_limits<int32_t>::min() ||
        affine->high > std::numeric_limits<int32_t>::max()) {
      return std::nullopt;
    }
    // `scale * size + offset <dir> limit` is `size <dir'> bound`.
    auto bound = llvm::checkedSub(*limit, affine->offset);
    if (!bound) {
      return std::nullopt;
    }
    if (affine->scale < 0) {
      bound = llvm::checkedSub(int64_t{0}, *bound);
      if (!bound) {
        return std::nullopt;
      }
      direction = mirror(direction);
    }
    DimensionRange range;
    if (affine->divisor > 1) {
      range.udiv = affine->divisor;
    }
    switch (direction) {
    case ComparisonDirection::GE:
      range.umin = *bound;
      break;
    case ComparisonDirection::GT:
      range.umin = llvm::checkedAdd(*bound, int64_t{1});
      if (!range.umin) {
        return std::nullopt;
      }
      break;
    case ComparisonDirection::LE:
      range.umax = *bound;
      break;
    case ComparisonDirection::LT:
      range.umax = llvm::checkedSub(*bound, int64_t{1});
      if (!range.umax) {
        return std::nullopt;
      }
      break;
    default:
      return std::nullopt;
    }
    // A bound on the quotient bounds the size it exactly divides.
    for (std::optional<int64_t> *end : {&range.umin, &range.umax}) {
      if (*end && !(*end = llvm::checkedMul(**end, affine->divisor))) {
        return std::nullopt;
      }
    }
    // Every size meets a lower bound of at most 0 or an upper bound of
    // INT32_MAX; none meets a negative upper bound.
    if (range.umin && *range.umin <= 0) {
      return std::nullopt;
    }
    if (range.umax && (*range.umax < 0 ||
                       *range.umax >= std::numeric_limits<int32_t>::max())) {
      return std::nullopt;
    }
    return AssertionFact{{affine->dim}, range, std::nullopt, nullptr};
  }

  // An i32 expression in [-2^31, 2^32) congruent to a nonnegative i32 size
  // must equal that size as a mathematical integer. Do not apply the stronger
  // no-wrap precondition used by ordered comparisons to this equality proof.
  std::optional<AssertionFact>
  matchExpressionEquality(Value target, Value expression, Block *block) {
    std::optional<DimensionSize> size = matchDimensionSize(target, block);
    std::optional<AffineSizeExpression> affine =
        matchAffineExpression(expression, block);
    if (!size || !affine || affine->scale <= 0 || affine->divisor != 1 ||
        affine->dim.key == size->key ||
        affine->low < std::numeric_limits<int32_t>::min() ||
        affine->high >= (int64_t{1} << 32)) {
      return std::nullopt;
    }
    return AssertionFact{
        {*size},
        DimensionRange{},
        DimensionExpression{affine->dim, affine->scale, affine->offset},
        nullptr};
  }

  std::optional<AssertionFact>
  matchAssertionFact(mlir::stablehlo::CustomCallOp call) {
    if (call.getCallTargetName() != "shape_assertion" ||
        call.getNumOperands() == 0 || call.getNumResults() != 0) {
      return std::nullopt;
    }
    if (!call->getAttrOfType<StringAttr>("error_message")) {
      return std::nullopt;
    }
    auto predicateType =
        dyn_cast<RankedTensorType>(call.getOperand(0).getType());
    auto compare =
        call.getOperand(0).getDefiningOp<mlir::stablehlo::CompareOp>();
    Block *block = call->getBlock();
    if (!predicateType || predicateType.getRank() != 0 ||
        !predicateType.getElementType().isInteger(1) || !compare ||
        compare->getBlock() != block || !compare->isBeforeInBlock(call)) {
      return std::nullopt;
    }

    std::optional<AssertionFact> fact;
    if (compare.getComparisonDirection() ==
        mlir::stablehlo::ComparisonDirection::EQ) {
      std::optional<DimensionSize> lhs =
          matchDimensionSize(compare.getLhs(), block);
      std::optional<DimensionSize> rhs =
          matchDimensionSize(compare.getRhs(), block);
      if (lhs && rhs) {
        fact = AssertionFact{
            {*lhs, *rhs}, DimensionRange{}, std::nullopt, nullptr};
      } else if (lhs) {
        fact =
            matchExpressionEquality(compare.getLhs(), compare.getRhs(), block);
      } else if (rhs) {
        fact =
            matchExpressionEquality(compare.getRhs(), compare.getLhs(), block);
      }
      if (!fact) {
        if (std::optional<Divisibility> divides =
                matchDivisibility(compare, block)) {
          fact = AssertionFact{
              {divides->dim},
              DimensionRange{std::nullopt, std::nullopt, divides->divisor},
              std::nullopt,
              nullptr};
        }
      }
    } else {
      fact = matchBound(compare, block);
    }
    if (fact) {
      fact->assertion = call;
    }
    return fact;
  }

  void collectDivisibility(FunctionOpInterface function) {
    function->walk([&](mlir::stablehlo::CustomCallOp call) {
      if (call.getCallTargetName() != "shape_assertion" ||
          call.getNumOperands() == 0 || call.getNumResults() != 0 ||
          !call->getAttrOfType<StringAttr>("error_message")) {
        return;
      }
      auto compare =
          call.getOperand(0).getDefiningOp<mlir::stablehlo::CompareOp>();
      Block *block = call->getBlock();
      if (!compare || compare->getBlock() != block) {
        return;
      }
      if (std::optional<Divisibility> divides =
              matchDivisibility(compare, block)) {
        divisibility[block].push_back(*divides);
      }
    });
  }

  SmallVector<BlockAssertionFacts> collectFacts(FunctionOpInterface function) {
    SmallVector<BlockAssertionFacts> blockFacts;
    llvm::DenseMap<Block *, unsigned> blockIndex;
    function->walk([&](mlir::stablehlo::CustomCallOp call) {
      auto fact = matchAssertionFact(call);
      if (!fact) {
        return;
      }
      auto [it, inserted] =
          blockIndex.try_emplace(call->getBlock(), blockFacts.size());
      if (inserted) {
        blockFacts.push_back({call->getBlock(), {*fact}});
      } else {
        blockFacts[it->second].facts.push_back(*fact);
      }
    });

    return blockFacts;
  }
};
// Keep current inference policy: LCM overflow drops the divisor; inconsistent
// bounds suppress assumptions but do not invalidate a separately proven tie.
static void mergeRange(DimensionRange &range, const DimensionRange &incoming) {
  if (incoming.umin) {
    range.umin = std::max(range.umin.value_or(0), *incoming.umin);
  }
  if (incoming.umax) {
    range.umax = std::min(range.umax.value_or(*incoming.umax), *incoming.umax);
  }
  if (incoming.udiv) {
    int64_t previous = range.udiv.value_or(1);
    range.udiv = llvm::checkedMul(previous / std::gcd(previous, *incoming.udiv),
                                  *incoming.udiv);
  }
}

static SmallVector<const AssertionFact *> selectExpressionTies(
    ArrayRef<AssertionFact> facts,
    const llvm::DenseMap<std::pair<Value, int64_t>, unsigned> &keyIndex) {
  // Keep expressions independent: the target is uniquely defined and absent
  // from equality classes, and its source is not another expression target.
  llvm::DenseMap<std::pair<Value, int64_t>, unsigned> targets;
  for (const AssertionFact &fact : facts) {
    if (fact.expression) {
      DimensionKey key = fact.dims.front().key;
      ++targets[{key.tensor, key.dimension}];
    }
  }

  SmallVector<const AssertionFact *> result;
  for (const AssertionFact &fact : facts) {
    if (!fact.expression) {
      continue;
    }
    DimensionKey target = fact.dims.front().key;
    DimensionKey source = fact.expression->source.key;
    if (!keyIndex.contains({target.tensor, target.dimension}) &&
        targets.lookup({target.tensor, target.dimension}) == 1 &&
        !targets.lookup({source.tensor, source.dimension})) {
      result.push_back(&fact);
    }
  }
  return result;
}

static BlockShapeAssertionFacts
mergeBlockFacts(const BlockAssertionFacts &block) {
  SmallVector<DimensionKey> keys;
  llvm::DenseMap<std::pair<Value, int64_t>, unsigned> keyIndex;
  llvm::IntEqClasses classes;
  auto getOrAddKeyIndex = [&](DimensionKey key) {
    auto [it, inserted] =
        keyIndex.try_emplace({key.tensor, key.dimension}, keys.size());
    if (inserted) {
      keys.push_back(key);
      classes.grow(keys.size());
    }
    return it->second;
  };
  for (const AssertionFact &fact : block.facts) {
    if (!fact.expression) {
      unsigned lhs = getOrAddKeyIndex(fact.dims.front().key);
      unsigned rhs = getOrAddKeyIndex(fact.dims.back().key);
      classes.join(lhs, rhs);
    }
  }
  llvm::MapVector<unsigned, DimensionClassFacts> merged;
  for (auto [index, key] : llvm::enumerate(keys)) {
    merged[classes.findLeader(index)].members.push_back(key);
  }
  Operation *lastAssertion = block.facts.front().assertion;
  for (const AssertionFact &fact : block.facts) {
    if (lastAssertion->isBeforeInBlock(fact.assertion)) {
      lastAssertion = fact.assertion;
    }
    if (fact.expression) {
      continue;
    }
    DimensionKey key = fact.dims.front().key;
    auto &dimensionClass = merged[classes.findLeader(
        keyIndex.lookup({key.tensor, key.dimension}))];
    for (const DimensionSize &dim : fact.dims) {
      dimensionClass.sizeQueries.push_back(dim.size);
    }
    mergeRange(dimensionClass.range, fact.range);
  }
  // Materialize only classes with an equality or usable assumption. Inactive
  // singleton sources remain available to expression ties through their query.
  BlockShapeAssertionFacts result{lastAssertion, {}, {}};
  llvm::DenseMap<unsigned, unsigned> activeClasses;
  for (auto &[root, dimensionClass] : merged) {
    if (dimensionClass.members.size() > 1 ||
        hasUsableAssumption(dimensionClass.range)) {
      activeClasses[root] = result.classes.size();
      result.classes.push_back(std::move(dimensionClass));
    }
  }
  for (const AssertionFact *fact :
       selectExpressionTies(block.facts, keyIndex)) {
    ExpressionTieFact tie{fact->dims.front(), *fact->expression, std::nullopt};
    DimensionKey source = tie.expression.source.key;
    auto sourceIt = keyIndex.find({source.tensor, source.dimension});
    if (sourceIt != keyIndex.end()) {
      auto active = activeClasses.find(classes.findLeader(sourceIt->second));
      if (active != activeClasses.end()) {
        tie.sourceClass = active->second;
      }
    }
    result.expressions.push_back(tie);
  }
  return result;
}

using OriginalUses = llvm::MapVector<Value, SmallVector<OpOperand *>>;
static void snapshotUses(Value value, OriginalUses &usesByValue) {
  if (usesByValue.count(value)) {
    return;
  }
  auto &uses = usesByValue[value];
  for (OpOperand &use : value.getUses()) {
    uses.push_back(&use);
  }
}

static IREE::Util::IntAssumptionAttr
buildAssumption(IRRewriter &rewriter, const DimensionRange &range) {
  if (!hasUsableAssumption(range)) {
    return {};
  }
  auto toUnsigned =
      [](std::optional<int64_t> value) -> std::optional<uint64_t> {
    if (value) {
      return static_cast<uint64_t>(*value);
    }
    return std::nullopt;
  };
  return rewriter.getAttr<IREE::Util::IntAssumptionAttr>(
      toUnsigned(range.umin), toUnsigned(range.umax), toUnsigned(range.udiv));
}
class BlockShapeFactMaterializer {
public:
  BlockShapeFactMaterializer(const BlockShapeAssertionFacts &facts,
                             IRRewriter &rewriter, DominanceInfo &dominance)
      : facts(facts), rewriter(rewriter), dominance(dominance),
        loc(facts.lastAssertion->getLoc()) {}
  void run() {
    if (facts.classes.empty() && facts.expressions.empty()) {
      return;
    }
    snapshotOriginalUses();
    rewriter.setInsertionPointAfter(facts.lastAssertion);
    buildDimensionClasses();
    buildExpressionTies();
    buildTensorTies();
    replaceTensorUses();
    replaceSizeUses();
  }

private:
  const BlockShapeAssertionFacts &facts;
  IRRewriter &rewriter;
  DominanceInfo &dominance;
  Location loc;
  // Snapshots never include newly built operations. No owner erasure or operand
  // resizing is permitted until these borrowed use pointers have been consumed.
  OriginalUses tensorUses;
  OriginalUses sizeUses;
  SmallVector<std::pair<DimensionKey, Value>> tiedDims;
  llvm::MapVector<Value, Value> canonicalSizeTensorForQuery;
  SmallVector<Value> canonicalScalars;
  llvm::MapVector<Value, Value> tiedTensors;
  void snapshotOriginalUses() {
    for (const DimensionClassFacts &dimensionClass : facts.classes) {
      for (const DimensionKey &key : dimensionClass.members) {
        snapshotUses(key.tensor, tensorUses);
      }
      for (auto size : dimensionClass.sizeQueries) {
        snapshotUses(size.getResult(), sizeUses);
      }
    }
    for (const ExpressionTieFact &tie : facts.expressions) {
      snapshotUses(tie.target.key.tensor, tensorUses);
      snapshotUses(tie.target.size->getResult(0), sizeUses);
    }
  }

  void buildDimensionClasses() {
    for (const DimensionClassFacts &dimensionClass : facts.classes) {
      Value canonicalSize = dimensionClass.sizeQueries.front()->getResult(0);
      Value canonicalScalar =
          tensor::ExtractOp::create(rewriter, loc, canonicalSize, ValueRange{});
      // Facts sit on the i32 sizes, so a dimension and a size query read back
      // as an index are one value.
      if (IREE::Util::IntAssumptionAttr assumption =
              buildAssumption(rewriter, dimensionClass.range)) {
        canonicalScalar = IREE::Util::AssumeIntOp::create(
                              rewriter, loc, canonicalScalar, assumption)
                              .getResult(0);
        // Size queries read the bounded value too.
        canonicalSize = tensor::FromElementsOp::create(
            rewriter, loc, canonicalSize.getType(), canonicalScalar);
      }
      Value canonicalDim = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), canonicalScalar);
      canonicalScalars.push_back(canonicalScalar);
      for (const DimensionKey &key : dimensionClass.members) {
        tiedDims.push_back({key, canonicalDim});
      }
      for (mlir::stablehlo::GetDimensionSizeOp sizeOp :
           dimensionClass.sizeQueries) {
        canonicalSizeTensorForQuery[sizeOp.getResult()] = canonicalSize;
      }
    }
  }

  void buildExpressionTies() {
    for (const ExpressionTieFact &tie : facts.expressions) {
      DimensionExpression expression = tie.expression;
      Value scalar;
      if (tie.sourceClass) {
        scalar = canonicalScalars[*tie.sourceClass];
      }
      if (!scalar) {
        scalar = tensor::ExtractOp::create(
            rewriter, loc, expression.source.size.getResult(), ValueRange{});
      }
      // Where the assertion holds, the i32 arithmetic is exact.
      auto scalarType = cast<IntegerType>(scalar.getType());
      if (expression.scale != 1) {
        Value scale = arith::ConstantIntOp::create(rewriter, loc, scalarType,
                                                   expression.scale);
        scalar = arith::MulIOp::create(rewriter, loc, scalar, scale);
      }
      if (expression.offset != 0) {
        Value offset = arith::ConstantIntOp::create(rewriter, loc, scalarType,
                                                    expression.offset);
        scalar = arith::AddIOp::create(rewriter, loc, scalar, offset);
      } else if (expression.scale != 1) {
        auto assumption = rewriter.getAttr<IREE::Util::IntAssumptionAttr>(
            /*umin=*/std::nullopt, /*umax=*/std::nullopt,
            static_cast<uint64_t>(expression.scale));
        scalar =
            IREE::Util::AssumeIntOp::create(rewriter, loc, scalar, assumption)
                .getResult(0);
      }
      DimensionSize target = tie.target;
      tiedDims.push_back(
          {target.key, arith::IndexCastOp::create(
                           rewriter, loc, rewriter.getIndexType(), scalar)});
      Value sizeTensor = target.size.getResult();
      canonicalSizeTensorForQuery[sizeTensor] = tensor::FromElementsOp::create(
          rewriter, loc, sizeTensor.getType(), scalar);
    }
  }

  void buildTensorTies() {
    auto lookupTiedDim = [&](DimensionKey key) -> Value {
      for (const auto &[tiedKey, dim] : tiedDims) {
        if (tiedKey == key) {
          return dim;
        }
      }
      return {};
    };
    for (const auto &[key, dim] : tiedDims) {
      Value tensorValue = key.tensor;
      if (tiedTensors.count(tensorValue)) {
        continue;
      }
      auto tensorType = cast<RankedTensorType>(tensorValue.getType());
      SmallVector<Value> dynamicDims;
      for (int64_t i = 0; i < tensorType.getRank(); ++i) {
        if (!tensorType.isDynamicDim(i)) {
          continue;
        }
        Value tiedDim = lookupTiedDim({tensorValue, i});
        dynamicDims.push_back(
            tiedDim ? tiedDim
                    : tensor::DimOp::create(rewriter, loc, tensorValue, i));
      }
      tiedTensors[tensorValue] = IREE::Flow::TensorTieShapeOp::create(
          rewriter, loc, tensorType, tensorValue, dynamicDims);
    }
  }

  void replaceTensorUses() {
    for (auto &[tensorValue, uses] : tensorUses) {
      Operation *tieOp = tiedTensors[tensorValue].getDefiningOp();
      for (OpOperand *use : uses) {
        if (dominance.properlyDominates(tieOp, use->getOwner())) {
          use->set(tiedTensors[tensorValue]);
        }
      }
    }
  }

  struct WidenedSizeUses {
    Type type;
    Value canonicalSize;
    SmallVector<OpOperand *> uses;
  };

  // Snapshot widening uses and their resolved replacements before changing any
  // size operands. This covers widenings both before and after the assertions.
  SmallVector<WidenedSizeUses> snapshotWidenedUses() {
    SmallVector<WidenedSizeUses> widenedUses;
    for (auto &[sizeValue, uses] : sizeUses) {
      for (OpOperand *use : uses) {
        auto widen = dyn_cast<mlir::stablehlo::ConvertOp>(use->getOwner());
        if (!widen || !isWidenedSize(widen)) {
          continue;
        }
        SmallVector<OpOperand *> later;
        for (OpOperand &widenUse : widen.getResult().getUses()) {
          if (dominance.properlyDominates(facts.lastAssertion,
                                          widenUse.getOwner())) {
            later.push_back(&widenUse);
          }
        }
        if (!later.empty()) {
          Value canonicalSize = canonicalSizeTensorForQuery.lookup(sizeValue);
          assert(canonicalSize && "expected a canonical size for every query");
          widenedUses.push_back({widen.getType(), canonicalSize, later});
        }
      }
    }
    return widenedUses;
  }

  void replaceSizeUses() {
    const auto widenedUses = snapshotWidenedUses();
    for (auto &[sizeValue, uses] : sizeUses) {
      Value canonicalSize = canonicalSizeTensorForQuery.lookup(sizeValue);
      assert(canonicalSize && "expected a canonical size for every query");
      for (OpOperand *use : uses) {
        if (dominance.properlyDominates(facts.lastAssertion, use->getOwner())) {
          use->set(canonicalSize);
        }
      }
    }
    for (const WidenedSizeUses &widened : widenedUses) {
      Value bounded = mlir::stablehlo::ConvertOp::create(
          rewriter, loc, widened.type, widened.canonicalSize);
      for (OpOperand *use : widened.uses) {
        use->set(bounded);
      }
    }
  }
};
} // namespace shape_assertions

// Facts apply only after their assertions. Finish analysis before constructing
// ties so neither proof queries nor recorded original keys observe mutated IR.
static void propagateShapeAssertionFacts(FunctionOpInterface function) {
  shape_assertions::ShapeAssertionFactAnalysis analysis;
  const auto facts = analysis.run(function);
  DominanceInfo dominance(function.getOperation());
  IRRewriter rewriter(function.getContext());
  for (const auto &block : facts) {
    shape_assertions::BlockShapeFactMaterializer(block, rewriter, dominance)
        .run();
  }
}
static const char *getShapeAssertionBlocker(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() == 0 || op.getNumResults() != 0) {
    return "expected a predicate and no results";
  }
  auto predicateType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  if (!predicateType || predicateType.getRank() != 0 ||
      !predicateType.getElementType().isInteger(1)) {
    return "expected a scalar i1 predicate";
  }
  auto message = op->getAttrOfType<StringAttr>("error_message");
  if (!message) {
    return "expected an error_message string";
  }
  return nullptr;
}

static LogicalResult rewriteShapeAssertion(mlir::stablehlo::CustomCallOp op,
                                           PatternRewriter &rewriter) {
  if (const char *blocker = getShapeAssertionBlocker(op)) {
    return rewriter.notifyMatchFailure(op, blocker);
  }
  auto message = op->getAttrOfType<StringAttr>("error_message");
  Value predicate = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                              op.getOperand(0), ValueRange{});
  // Semantic assertions must survive executable debug stripping.
  emitStatusCheck(rewriter, op.getLoc(), predicate, message.getValue());
  rewriter.eraseOp(op);
  return success();
}

// A coupled status component uses the same scatter geometry as the data. It
// records the first failing assertion at each destination, including across
// repeated indices. Reducing it afterwards keeps the failure check on the host
// without introducing shader status ABI requirements.
static Value createScatterStatusTensor(Value shapedValue,
                                       PatternRewriter &rewriter,
                                       Location loc) {
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, shapedValue),
      rewriter.getI32Type());
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  return linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
}

// The failure branch retains the current data and status. Building the reducer
// continuation in the success branch prevents subsequent operations (such as a
// division protected by the assertion) from running after a failed check.
static scf::IfOp createScatterAssertionGuard(Value condition,
                                             ValueRange currentValues,
                                             Value failureStatus,
                                             PatternRewriter &rewriter,
                                             Location loc) {
  SmallVector<Value> failureValues(currentValues);
  failureValues.push_back(failureStatus);
  auto guard = scf::IfOp::create(
      rewriter, loc, ValueRange(failureValues).getTypes(), condition,
      /*addThenBlock=*/true, /*addElseBlock=*/true);
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPointToStart(guard.elseBlock());
  scf::YieldOp::create(rewriter, loc, failureValues);
  return guard;
}

static void buildCheckedScatterReducer(mlir::stablehlo::ScatterOp source,
                                       mlir::stablehlo::ScatterOp target,
                                       PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  Block &sourceBlock = source.getUpdateComputation().front();
  int64_t inputCount = source.getInputs().size();
  SmallVector<Type> argumentTypes(sourceBlock.getArgumentTypes());
  auto statusType = RankedTensorType::get({}, rewriter.getI32Type());
  argumentTypes.insert(argumentTypes.begin() + inputCount, statusType);
  argumentTypes.push_back(statusType);
  SmallVector<Location> locations(argumentTypes.size(), source.getLoc());
  Block *block = rewriter.createBlock(&target.getUpdateComputation(), {},
                                      argumentTypes, locations);
  IRMapping mapping;
  mapping.map(sourceBlock.getArguments().take_front(inputCount),
              block->getArguments().take_front(inputCount));
  mapping.map(sourceBlock.getArguments().drop_front(inputCount),
              block->getArguments().slice(inputCount + 1, inputCount));
  ValueRange currentValues = block->getArguments().take_front(inputCount);
  Value previousStatus = block->getArgument(inputCount);
  Location loc = source.getLoc();
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value scalarStatus = tensor::ExtractOp::create(rewriter, loc, previousStatus);
  Value wasSuccessful = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, scalarStatus, zero);
  auto guard = createScatterAssertionGuard(wasSuccessful, currentValues,
                                           previousStatus, rewriter, loc);
  mlir::stablehlo::ReturnOp::create(rewriter, loc, guard.getResults());
  rewriter.setInsertionPointToStart(guard.thenBlock());
  Value successStatus = tensor::FromElementsOp::create(
      rewriter, loc, statusType, ValueRange{zero});
  int64_t assertionIndex = 0;
  for (Operation &operation : sourceBlock.without_terminator()) {
    auto assertion = dyn_cast<mlir::stablehlo::CustomCallOp>(operation);
    if (!assertion || assertion.getCallTargetName() != "shape_assertion") {
      rewriter.clone(operation, mapping);
      continue;
    }
    Value predicate = tensor::ExtractOp::create(
        rewriter, assertion.getLoc(),
        mapping.lookupOrDefault(assertion.getOperand(0)));
    Value failureStatus = arith::ConstantIntOp::create(
        rewriter, assertion.getLoc(), ++assertionIndex, 32);
    failureStatus = tensor::FromElementsOp::create(
        rewriter, assertion.getLoc(), statusType, ValueRange{failureStatus});
    auto continuation = createScatterAssertionGuard(
        predicate, currentValues, failureStatus, rewriter, assertion.getLoc());
    scf::YieldOp::create(rewriter, assertion.getLoc(),
                         continuation.getResults());
    rewriter.setInsertionPointToStart(continuation.thenBlock());
  }
  SmallVector<Value> results;
  for (Value result : sourceBlock.getTerminator()->getOperands()) {
    results.push_back(mapping.lookupOrDefault(result));
  }
  results.push_back(successStatus);
  scf::YieldOp::create(rewriter, loc, results);
}

static Value createScatterFailureReduction(Value status,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  auto type = cast<RankedTensorType>(status.getType());
  auto scalarType = RankedTensorType::get({}, rewriter.getI32Type());
  Value zero = mlir::stablehlo::ConstantOp::create(
      rewriter, loc, DenseIntElementsAttr::get(scalarType, {0}));
  SmallVector<int64_t> dimensions(type.getRank());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  auto reduce = mlir::stablehlo::ReduceOp::create(
      rewriter, loc, TypeRange{scalarType}, ValueRange{status},
      ValueRange{zero}, rewriter.getDenseI64ArrayAttr(dimensions));
  {
    OpBuilder::InsertionGuard insertionGuard(rewriter);
    Block *block = rewriter.createBlock(&reduce.getBody(), {},
                                        {scalarType, scalarType}, {loc, loc});
    Value maximum = mlir::stablehlo::MaxOp::create(
        rewriter, loc, block->getArgument(0), block->getArgument(1));
    mlir::stablehlo::ReturnOp::create(rewriter, loc, maximum);
  }
  return tensor::ExtractOp::create(rewriter, loc, reduce.getResult(0));
}

// Before (inside the update computation):
//   stablehlo.custom_call @shape_assertion(%predicate) ...
//   stablehlo.return %updated
// After adding a coupled status component:
//   %next:2 = scf.if %predicate -> (tensor<T>, tensor<i32>) {
//     scf.yield %updated, %success : tensor<T>, tensor<i32>
//   } else {
//     scf.yield %current, %failure : tensor<T>, tensor<i32>
//   }
//   stablehlo.return %next#0, %next#1
// A reduction selects a failed check for the host to report. Scatter does not
// specify an update order, so independent failing destinations need not report
// the same message. Other effects and nested checks retain the existing
// lowering.
struct ScatterShapeAssertionLowering final
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    Block &block = op.getUpdateComputation().front();
    SmallVector<mlir::stablehlo::CustomCallOp> assertions;
    for (Operation &operation : block.without_terminator()) {
      auto assertion = dyn_cast<mlir::stablehlo::CustomCallOp>(operation);
      if (!assertion || assertion.getCallTargetName() != "shape_assertion") {
        if (!isMemoryEffectFree(&operation)) {
          return rewriter.notifyMatchFailure(
              op, "expected pure computation between direct shape assertions");
        }
        continue;
      }
      if (const char *blocker = getShapeAssertionBlocker(assertion)) {
        return rewriter.notifyMatchFailure(op, blocker);
      }
      assertions.push_back(assertion);
    }
    if (assertions.empty()) {
      return failure();
    }
    if (assertions.size() > std::numeric_limits<int32_t>::max()) {
      return rewriter.notifyMatchFailure(op,
                                         "too many scatter shape assertions");
    }
    TypeRange currentTypes(
        block.getArguments().take_front(op.getInputs().size()));
    if (!llvm::equal(currentTypes, block.getTerminator()->getOperandTypes())) {
      return rewriter.notifyMatchFailure(
          op, "scatter input promotion must precede assertion lifting");
    }

    Location loc = op.getLoc();
    SmallVector<Value> inputs(op.getInputs());
    inputs.push_back(createScatterStatusTensor(inputs.front(), rewriter, loc));
    SmallVector<Value> updates(op.getUpdates());
    // The status update argument is never read. Initialising it would introduce
    // a dynamic temporary inside the scatter dispatch after producer fusion.
    updates.push_back(tensor::EmptyOp::create(
        rewriter, loc, tensor::getMixedSizes(rewriter, loc, updates.front()),
        rewriter.getI32Type()));
    SmallVector<Type> resultTypes(op.getResultTypes());
    resultTypes.push_back(inputs.back().getType());
    auto scatter = mlir::stablehlo::ScatterOp::create(
        rewriter, loc, resultTypes, inputs, op.getScatterIndices(), updates,
        op.getScatterDimensionNumbers(), op.getIndicesAreSorted(),
        op.getUniqueIndices());
    scatter->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    buildCheckedScatterReducer(op, scatter, rewriter);
    Value failure = createScatterFailureReduction(scatter.getResults().back(),
                                                  rewriter, loc);
    for (auto [index, assertion] : llvm::enumerate(assertions)) {
      Value code = arith::ConstantIntOp::create(rewriter, assertion.getLoc(),
                                                index + 1, 32);
      Value passed =
          arith::CmpIOp::create(rewriter, assertion.getLoc(),
                                arith::CmpIPredicate::ne, failure, code);
      emitStatusCheck(
          rewriter, assertion.getLoc(), passed,
          assertion->getAttrOfType<StringAttr>("error_message").getValue());
    }
    rewriter.replaceOp(op, scatter.getResults().drop_back());
    return success();
  }
};

struct ShapeAssertionLowering final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != "shape_assertion") {
      return failure();
    }
    return rewriteShapeAssertion(op, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition.
//===----------------------------------------------------------------------===//

struct LegalizeStableHLOCustomCalls final
    : impl::LegalizeStableHLOCustomCallsBase<LegalizeStableHLOCustomCalls> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    mlir::stablehlo::StablehloDialect, tensor::TensorDialect,
                    IREE::Flow::FlowDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto f = getOperation();
    MLIRContext *ctx = f.getContext();

    propagateShapeAssertionFacts(f);

    // Lift reducer checks before lowering individual custom calls. Otherwise
    // the status checks would become device effects inside the scatter body.
    RewritePatternSet scatterPatterns(ctx);
    scatterPatterns.add<ScatterShapeAssertionLowering>(ctx);
    if (failed(applyPatternsGreedily(f, std::move(scatterPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(ctx);
    patterns.add<HouseholderReflectorRewriter, ShapeAssertionLowering>(ctx);
    if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
