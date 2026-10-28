// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops,
// taking care of CHLO's broadcasting semantics

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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <functional>
#include <numeric>

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

// Unifies dimensions when an assertion directly checks their equality, ties
// one to a multiple of another plus a constant, and bounds those an assertion
// compares with a constant. The ties are inserted after the assertions so the
// asserted predicate is always evaluated using the original values.
static void propagateShapeAssertionFacts(FunctionOpInterface function) {
  struct DimKey {
    Value tensor;
    int64_t dimension;

    bool operator==(const DimKey &other) const {
      return tensor == other.tensor && dimension == other.dimension;
    }
  };
  struct DimSize {
    DimKey key;
    mlir::stablehlo::GetDimensionSizeOp size;
  };
  // An unsigned range of a dimension and a divisor of it; a missing entry is
  // unbounded.
  struct DimRange {
    std::optional<int64_t> umin;
    std::optional<int64_t> umax;
    std::optional<int64_t> udiv;
  };
  // `scale * source + offset`.
  struct Expression {
    DimSize source;
    int64_t scale;
    int64_t offset;
  };
  // Two dimensions an assertion proves equal, one it bounds, or one it proves
  // equal to an expression of another.
  struct Fact {
    SmallVector<DimSize, 2> dims;
    DimRange range;
    std::optional<Expression> expression;
    mlir::stablehlo::CustomCallOp assertion;
  };
  struct BlockFacts {
    Block *block;
    SmallVector<Fact> facts;
  };

  auto getDimSize = [](Value value, Block *block) -> std::optional<DimSize> {
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
    return DimSize{{sizeOp.getOperand(), dim}, sizeOp};
  };
  auto getConstant = [](Value value) -> std::optional<int64_t> {
    APInt constant;
    if (!matchPattern(value, m_ConstantInt(&constant))) {
      return std::nullopt;
    }
    // An i1 true is 1, not -1.
    return constant.getBitWidth() == 1 ? constant.getZExtValue()
                                       : constant.getSExtValue();
  };

  // A closed range of a scalar; an i1 is in [0, 1].
  struct Interval {
    int64_t lo;
    int64_t hi;
  };
  // Values already bounded in one evaluation; shared operands are bounded once.
  using IntervalCache = llvm::DenseMap<Value, std::optional<Interval>>;
  // Bounds |value| when every size it reads is the one dimension `key`, in
  // `sizes`. Fails on a second dimension and on ops it does not model.
  std::function<std::optional<Interval>(
      Value, Block *, Interval, std::optional<DimKey> &, IntervalCache &, int)>
      evaluate;
  std::function<std::optional<Interval>(
      Value, Block *, Interval, std::optional<DimKey> &, IntervalCache &, int)>
      evaluateUncached = [&](Value value, Block *block, Interval sizes,
                             std::optional<DimKey> &key, IntervalCache &cache,
                             int depth) -> std::optional<Interval> {
    if (std::optional<int64_t> constant = getConstant(value)) {
      return Interval{*constant, *constant};
    }
    if (std::optional<DimSize> dim = getDimSize(value, block)) {
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
      return evaluate(op->getOperand(i), block, sizes, key, cache, depth + 1);
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
    std::optional<Interval> lhs = operand(0);
    std::optional<Interval> rhs = operand(1);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    if (isI1(value) && isa<mlir::stablehlo::AndOp, mlir::stablehlo::OrOp>(op)) {
      if (isa<mlir::stablehlo::AndOp>(op)) {
        return Interval{lhs->lo & rhs->lo, lhs->hi & rhs->hi};
      }
      return Interval{lhs->lo | rhs->lo, lhs->hi | rhs->hi};
    }
    // Arithmetic on i1 is logical, not modelled here.
    if (isI1(value) && !isa<mlir::stablehlo::CompareOp>(op)) {
      return std::nullopt;
    }
    // Fails where the element type would wrap.
    unsigned width = getElementTypeOrSelf(value).getIntOrFloatBitWidth();
    auto fits = [&](Interval x) -> std::optional<Interval> {
      if (width >= 64 || (x.lo >= -(int64_t{1} << (width - 1)) &&
                          x.hi < (int64_t{1} << (width - 1)))) {
        return x;
      }
      return std::nullopt;
    };
    if (isa<mlir::stablehlo::DivOp, mlir::stablehlo::RemOp>(op)) {
      // Only non-negative values by a positive constant.
      if (rhs->lo != rhs->hi || rhs->lo <= 0 || lhs->lo < 0) {
        return std::nullopt;
      }
      if (isa<mlir::stablehlo::DivOp>(op)) {
        return Interval{lhs->lo / rhs->lo, lhs->hi / rhs->lo};
      }
      return lhs->hi < rhs->lo ? *lhs : Interval{0, rhs->lo - 1};
    }
    if (isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp,
            mlir::stablehlo::MulOp>(op)) {
      std::optional<int64_t> a, b, c, d;
      if (isa<mlir::stablehlo::AddOp>(op)) {
        a = llvm::checkedAdd(lhs->lo, rhs->lo);
        b = llvm::checkedAdd(lhs->hi, rhs->hi);
        if (!a || !b) {
          return std::nullopt;
        }
        return fits(Interval{*a, *b});
      }
      if (isa<mlir::stablehlo::SubtractOp>(op)) {
        a = llvm::checkedSub(lhs->lo, rhs->hi);
        b = llvm::checkedSub(lhs->hi, rhs->lo);
        if (!a || !b) {
          return std::nullopt;
        }
        return fits(Interval{*a, *b});
      }
      a = llvm::checkedMul(lhs->lo, rhs->lo);
      b = llvm::checkedMul(lhs->lo, rhs->hi);
      c = llvm::checkedMul(lhs->hi, rhs->lo);
      d = llvm::checkedMul(lhs->hi, rhs->hi);
      if (!a || !b || !c || !d) {
        return std::nullopt;
      }
      return fits(
          Interval{std::min({*a, *b, *c, *d}), std::max({*a, *b, *c, *d})});
    }
    auto compare = dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compare) {
      return std::nullopt;
    }
    std::optional<mlir::stablehlo::ComparisonType> compareType =
        compare.getCompareType();
    if (compareType &&
        *compareType == mlir::stablehlo::ComparisonType::UNSIGNED &&
        (lhs->lo < 0 || rhs->lo < 0)) {
      return std::nullopt;
    }
    if (compareType &&
        *compareType != mlir::stablehlo::ComparisonType::SIGNED &&
        *compareType != mlir::stablehlo::ComparisonType::UNSIGNED &&
        *compareType != mlir::stablehlo::ComparisonType::NOTYPE) {
      return std::nullopt;
    }
    // `lhs < rhs` holds for all, for none, or for some values.
    auto less = [](Interval x, Interval y) -> Interval {
      if (x.hi < y.lo) {
        return {1, 1};
      }
      if (x.lo >= y.hi) {
        return {0, 0};
      }
      return {0, 1};
    };
    auto negate = [](Interval x) { return Interval{1 - x.hi, 1 - x.lo}; };
    Interval equal =
        lhs->lo == lhs->hi && rhs->lo == rhs->hi && lhs->lo == rhs->lo
            ? Interval{1, 1}
        : lhs->hi < rhs->lo || rhs->hi < lhs->lo ? Interval{0, 0}
                                                 : Interval{0, 1};
    switch (compare.getComparisonDirection()) {
    case mlir::stablehlo::ComparisonDirection::EQ:
      return equal;
    case mlir::stablehlo::ComparisonDirection::NE:
      return negate(equal);
    case mlir::stablehlo::ComparisonDirection::LT:
      return less(*lhs, *rhs);
    case mlir::stablehlo::ComparisonDirection::GE:
      return negate(less(*lhs, *rhs));
    case mlir::stablehlo::ComparisonDirection::GT:
      return less(*rhs, *lhs);
    case mlir::stablehlo::ComparisonDirection::LE:
      return negate(less(*rhs, *lhs));
    }
    return std::nullopt;
  };
  evaluate = [&](Value value, Block *block, Interval sizes,
                 std::optional<DimKey> &key, IntervalCache &cache,
                 int depth) -> std::optional<Interval> {
    if (auto it = cache.find(value); it != cache.end()) {
      return it->second;
    }
    std::optional<Interval> result =
        evaluateUncached(value, block, sizes, key, cache, depth);
    cache[value] = result;
    return result;
  };
  // Decides a predicate over one non-negative dimension by evaluating it for
  // a size of zero and for every positive size.
  auto decide = [&](Value predicate, Block *block) -> std::optional<bool> {
    std::optional<DimKey> key;
    IntervalCache zeroCache;
    IntervalCache positiveCache;
    std::optional<Interval> zero =
        evaluate(predicate, block, Interval{0, 0}, key, zeroCache, 0);
    std::optional<Interval> positive = evaluate(
        predicate, block, Interval{1, std::numeric_limits<int32_t>::max()}, key,
        positiveCache, 0);
    if (!zero || !positive || zero->lo != zero->hi ||
        positive->lo != positive->hi || zero->lo != positive->lo) {
      return std::nullopt;
    }
    return zero->lo != 0;
  };
  // Follows same-type converts and selects whose predicate is decided.
  auto peel = [&](Value value, Block *block) {
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
      std::optional<bool> taken = decide(select.getPred(), block);
      if (!taken) {
        return value;
      }
      value = *taken ? select.getOnTrue() : select.getOnFalse();
    }
  };

  // A size and a positive constant dividing it, from an assertion that the
  // remainder of the size by that constant is zero.
  struct Divisibility {
    DimSize dim;
    int64_t divisor;
  };
  auto getDivisibility = [&](mlir::stablehlo::CompareOp compare,
                             Block *block) -> std::optional<Divisibility> {
    if (compare.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::EQ) {
      return std::nullopt;
    }
    Value remainder = compare.getLhs();
    std::optional<int64_t> zero = getConstant(compare.getRhs());
    if (!zero) {
      remainder = compare.getRhs();
      zero = getConstant(compare.getLhs());
    }
    if (!zero || *zero != 0) {
      return std::nullopt;
    }
    auto rem = peel(remainder, block).getDefiningOp<mlir::stablehlo::RemOp>();
    if (!rem || rem->getBlock() != block) {
      return std::nullopt;
    }
    std::optional<int64_t> divisor = getConstant(rem.getRhs());
    std::optional<DimSize> dim = getDimSize(peel(rem.getLhs(), block), block);
    if (!divisor || *divisor <= 0 || !dim) {
      return std::nullopt;
    }
    return Divisibility{*dim, *divisor};
  };
  // Divisors each block's assertions prove, by dimension.
  llvm::DenseMap<Block *, SmallVector<Divisibility>> divisibility;
  auto isDivisible = [&](DimKey key, int64_t divisor, Block *block) {
    return llvm::any_of(divisibility.lookup(block), [&](const Divisibility &d) {
      return d.dim.key == key && d.divisor % divisor == 0;
    });
  };

  // `scale * size / divisor + offset` over the integers, with the range it
  // takes over every size in [0, INT32_MAX]; the divisor divides the size. The
  // i32 value computed is congruent to it modulo 2^32; callers check the range
  // that makes the two equal.
  struct Affine {
    DimSize dim;
    int64_t scale;
    int64_t offset;
    int64_t low;
    int64_t high;
    int64_t divisor;
  };
  // Walks down a chain of steps by constants to a size, then applies the
  // steps upwards; the walk keeps no stack frame per step.
  auto getAffine = [&](Value value, Block *block) -> std::optional<Affine> {
    // `factor * term + addend` for one step.
    struct Step {
      int64_t factor;
      int64_t addend;
    };
    SmallVector<Step> steps;
    std::optional<DimSize> dim;
    int64_t divisor = 1;
    while (!(dim = getDimSize(value = peel(value, block), block))) {
      Operation *op = value.getDefiningOp();
      if (!op || op->getBlock() != block) {
        return std::nullopt;
      }
      // A quotient of the size by a constant an assertion proves divides it.
      if (auto divide = dyn_cast<mlir::stablehlo::DivOp>(op)) {
        std::optional<int64_t> constant = getConstant(divide.getRhs());
        dim = getDimSize(peel(divide.getLhs(), block), block);
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
      std::optional<int64_t> lhsConstant = getConstant(lhs);
      std::optional<int64_t> rhsConstant = getConstant(rhs);
      if (lhsConstant.has_value() == rhsConstant.has_value()) {
        return std::nullopt;
      }
      int64_t constant = lhsConstant ? *lhsConstant : *rhsConstant;
      if (isa<mlir::stablehlo::AddOp>(op)) {
        steps.push_back({1, constant});
      } else if (isa<mlir::stablehlo::SubtractOp>(op)) {
        steps.push_back(lhsConstant ? Step{-1, constant} : Step{1, -constant});
      } else if (isa<mlir::stablehlo::MulOp>(op) && constant != 0) {
        steps.push_back({constant, 0});
      } else {
        return std::nullopt;
      }
      value = lhsConstant ? rhs : lhs;
    }
    Affine affine{
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
      int64_t common =
          affine.divisor == 1 ? 1 : std::gcd(*scale, affine.divisor);
      affine = Affine{affine.dim, *scale / common, *offset,
                      *low,       *high,           affine.divisor / common};
    }
    return affine;
  };

  auto getBound = [&](mlir::stablehlo::CompareOp compare,
                      Block *block) -> std::optional<Fact> {
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
    std::optional<int64_t> limit = getConstant(compare.getRhs());
    std::optional<Affine> affine = getAffine(compare.getLhs(), block);
    if (!limit) {
      // Puts the constant on the right.
      limit = getConstant(compare.getLhs());
      affine = getAffine(compare.getRhs(), block);
      direction = mirror(direction);
    }
    // Bounds read the i32 value as exact, so it must not wrap.
    if (!limit || !affine || std::abs(affine->scale) != 1 ||
        affine->low < std::numeric_limits<int32_t>::min() ||
        affine->high > std::numeric_limits<int32_t>::max()) {
      return std::nullopt;
    }
    // `scale * size + offset <dir> limit` is `size <dir'> bound`.
    int64_t bound = *limit - affine->offset;
    if (affine->scale < 0) {
      bound = -bound;
      direction = mirror(direction);
    }
    DimRange range;
    if (affine->divisor > 1) {
      range.udiv = affine->divisor;
    }
    switch (direction) {
    case ComparisonDirection::GE:
      range.umin = bound;
      break;
    case ComparisonDirection::GT:
      range.umin = bound + 1;
      break;
    case ComparisonDirection::LE:
      range.umax = bound;
      break;
    case ComparisonDirection::LT:
      range.umax = bound - 1;
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
    return Fact{{affine->dim}, range, std::nullopt, nullptr};
  };

  // A size equal to `scale * source + offset` computed in i32. The i32 value
  // equals the exact one when the exact range lies in [-2^31, 2^32): a size is
  // in [0, 2^31), and the next congruent value would leave that range.
  auto getExpression = [&](Value target, Value expression,
                           Block *block) -> std::optional<Fact> {
    std::optional<DimSize> size = getDimSize(target, block);
    std::optional<Affine> affine = getAffine(expression, block);
    if (!size || !affine || affine->scale <= 0 || affine->divisor != 1 ||
        affine->dim.key == size->key ||
        affine->low < std::numeric_limits<int32_t>::min() ||
        affine->high >= (int64_t{1} << 32)) {
      return std::nullopt;
    }
    return Fact{{*size},
                DimRange{},
                Expression{affine->dim, affine->scale, affine->offset},
                nullptr};
  };

  auto getFact =
      [&](mlir::stablehlo::CustomCallOp call) -> std::optional<Fact> {
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

    std::optional<Fact> fact;
    if (compare.getComparisonDirection() ==
        mlir::stablehlo::ComparisonDirection::EQ) {
      std::optional<DimSize> lhs = getDimSize(compare.getLhs(), block);
      std::optional<DimSize> rhs = getDimSize(compare.getRhs(), block);
      if (lhs && rhs) {
        fact = Fact{{*lhs, *rhs}, DimRange{}, std::nullopt, nullptr};
      } else if (lhs) {
        fact = getExpression(compare.getLhs(), compare.getRhs(), block);
      } else if (rhs) {
        fact = getExpression(compare.getRhs(), compare.getLhs(), block);
      }
      if (!fact) {
        if (std::optional<Divisibility> divides =
                getDivisibility(compare, block)) {
          fact = Fact{{divides->dim},
                      DimRange{std::nullopt, std::nullopt, divides->divisor},
                      std::nullopt,
                      nullptr};
        }
      }
    } else {
      fact = getBound(compare, block);
    }
    if (fact) {
      fact->assertion = call;
    }
    return fact;
  };

  // Divisibility is collected first: bounds and ties on a quotient need it.
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
    if (std::optional<Divisibility> divides = getDivisibility(compare, block)) {
      divisibility[block].push_back(*divides);
    }
  });

  SmallVector<BlockFacts> blockFacts;
  llvm::DenseMap<Block *, unsigned> blockIndex;
  function->walk([&](mlir::stablehlo::CustomCallOp call) {
    auto fact = getFact(call);
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

  DominanceInfo dominance(function.getOperation());
  IRRewriter rewriter(function.getContext());
  for (BlockFacts &blockEntry : blockFacts) {
    SmallVector<DimKey> keys;
    llvm::DenseMap<std::pair<Value, int64_t>, unsigned> keyIndex;
    llvm::IntEqClasses classes;
    auto getKeyIndex = [&](DimKey key) {
      auto [it, inserted] =
          keyIndex.try_emplace({key.tensor, key.dimension}, keys.size());
      if (inserted) {
        keys.push_back(key);
        classes.grow(keys.size());
      }
      return it->second;
    };
    auto findRoot = [&](unsigned index) { return classes.findLeader(index); };

    SmallVector<Fact *> classFacts;
    SmallVector<Fact *> expressionFacts;
    for (Fact &fact : blockEntry.facts) {
      (fact.expression ? expressionFacts : classFacts).push_back(&fact);
    }
    for (Fact *fact : classFacts) {
      unsigned lhs = getKeyIndex(fact->dims.front().key);
      unsigned rhs = getKeyIndex(fact->dims.back().key);
      classes.join(lhs, rhs);
    }
    // An expression ties a dimension no other fact names, from a source no
    // expression ties.
    llvm::DenseMap<std::pair<Value, int64_t>, unsigned> expressionTargets;
    for (Fact *fact : expressionFacts) {
      DimKey target = fact->dims.front().key;
      ++expressionTargets[{target.tensor, target.dimension}];
    }
    SmallVector<Fact *> tiedExpressions;
    for (Fact *fact : expressionFacts) {
      DimKey target = fact->dims.front().key;
      DimKey source = fact->expression->source.key;
      if (!keyIndex.contains({target.tensor, target.dimension}) &&
          expressionTargets.lookup({target.tensor, target.dimension}) == 1 &&
          !expressionTargets.lookup({source.tensor, source.dimension})) {
        tiedExpressions.push_back(fact);
      }
    }

    Operation *lastAssertion = blockEntry.facts.front().assertion;
    for (const Fact &fact : blockEntry.facts) {
      if (lastAssertion->isBeforeInBlock(fact.assertion)) {
        lastAssertion = fact.assertion;
      }
    }

    llvm::MapVector<unsigned, SmallVector<DimKey>> membersByRoot;
    for (unsigned i = 0; i < keys.size(); ++i) {
      membersByRoot[findRoot(i)].push_back(keys[i]);
    }
    llvm::MapVector<unsigned, SmallVector<mlir::stablehlo::GetDimensionSizeOp>>
        sizesByRoot;
    llvm::MapVector<unsigned, DimRange> rangesByRoot;
    for (Fact *fact : classFacts) {
      unsigned root = findRoot(getKeyIndex(fact->dims.front().key));
      for (const DimSize &dim : fact->dims) {
        sizesByRoot[root].push_back(dim.size);
      }
      DimRange &range = rangesByRoot[root];
      if (fact->range.umin) {
        range.umin = std::max(range.umin.value_or(0), *fact->range.umin);
      }
      if (fact->range.umax) {
        range.umax =
            std::min(range.umax.value_or(*fact->range.umax), *fact->range.umax);
      }
      if (fact->range.udiv) {
        // An lcm that overflows drops the divisor.
        int64_t previous = range.udiv.value_or(1);
        range.udiv =
            llvm::checkedMul(previous / std::gcd(previous, *fact->range.udiv),
                             *fact->range.udiv);
      }
    }
    // Contradictory bounds fail every run; they give no fact.
    auto getAssumption = [&](unsigned root) -> IREE::Util::IntAssumptionAttr {
      const DimRange &range = rangesByRoot[root];
      if ((!range.umin && !range.umax && !range.udiv) ||
          (range.umin && range.umax && *range.umin > *range.umax)) {
        return nullptr;
      }
      auto toUnsigned = [](std::optional<int64_t> value) {
        return value ? std::optional<uint64_t>(*value) : std::nullopt;
      };
      return rewriter.getAttr<IREE::Util::IntAssumptionAttr>(
          toUnsigned(range.umin), toUnsigned(range.umax),
          toUnsigned(range.udiv));
    };

    SmallVector<unsigned> activeRoots;
    for (const auto &[root, members] : membersByRoot) {
      if (members.size() > 1 || getAssumption(root)) {
        activeRoots.push_back(root);
      }
    }
    if (activeRoots.empty() && tiedExpressions.empty()) {
      continue;
    }

    // Snapshot uses before inserting ties. Only dominated original uses are
    // rewritten; the comparisons feeding the assertions stay unchanged.
    llvm::MapVector<Value, SmallVector<OpOperand *>> tensorUses;
    llvm::MapVector<Value, SmallVector<OpOperand *>> sizeUses;
    auto snapshotUses =
        [](Value value,
           llvm::MapVector<Value, SmallVector<OpOperand *>> &usesByValue) {
          if (usesByValue.count(value)) {
            return;
          }
          SmallVector<OpOperand *> &uses = usesByValue[value];
          for (OpOperand &use : value.getUses()) {
            uses.push_back(&use);
          }
        };
    for (unsigned root : activeRoots) {
      for (const DimKey &key : membersByRoot[root]) {
        snapshotUses(key.tensor, tensorUses);
      }
      for (mlir::stablehlo::GetDimensionSizeOp sizeOp : sizesByRoot[root]) {
        snapshotUses(sizeOp.getResult(), sizeUses);
      }
    }
    for (Fact *fact : tiedExpressions) {
      snapshotUses(fact->dims.front().key.tensor, tensorUses);
      snapshotUses(fact->dims.front().size.getResult(), sizeUses);
    }

    rewriter.setInsertionPointAfter(lastAssertion);
    Location loc = lastAssertion->getLoc();
    // The index dimension each tied dimension takes, and the tensor<i32> each
    // dominated size query reads.
    SmallVector<std::pair<DimKey, Value>> tiedDims;
    llvm::MapVector<Value, Value> canonicalSizeTensorForQuery;
    llvm::MapVector<unsigned, Value> canonicalScalars;
    for (unsigned root : activeRoots) {
      Value canonicalSize = sizesByRoot[root].front().getResult();
      Value canonicalScalar =
          tensor::ExtractOp::create(rewriter, loc, canonicalSize, ValueRange{});
      // Facts sit on the i32 sizes, so a dimension and a size query read back
      // as an index are one value.
      if (IREE::Util::IntAssumptionAttr assumption = getAssumption(root)) {
        canonicalScalar = IREE::Util::AssumeIntOp::create(
                              rewriter, loc, canonicalScalar, assumption)
                              .getResult(0);
        // Size queries read the bounded value too.
        canonicalSize = tensor::FromElementsOp::create(
            rewriter, loc, canonicalSize.getType(), canonicalScalar);
      }
      Value canonicalDim = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), canonicalScalar);
      canonicalScalars[root] = canonicalScalar;
      for (const DimKey &key : membersByRoot[root]) {
        tiedDims.push_back({key, canonicalDim});
      }
      for (mlir::stablehlo::GetDimensionSizeOp sizeOp : sizesByRoot[root]) {
        canonicalSizeTensorForQuery[sizeOp.getResult()] = canonicalSize;
      }
    }
    for (Fact *fact : tiedExpressions) {
      Expression expression = *fact->expression;
      Value scalar;
      auto sourceIt = keyIndex.find(
          {expression.source.key.tensor, expression.source.key.dimension});
      if (sourceIt != keyIndex.end()) {
        scalar = canonicalScalars.lookup(findRoot(sourceIt->second));
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
      DimSize target = fact->dims.front();
      tiedDims.push_back(
          {target.key, arith::IndexCastOp::create(
                           rewriter, loc, rewriter.getIndexType(), scalar)});
      Value sizeTensor = target.size.getResult();
      canonicalSizeTensorForQuery[sizeTensor] = tensor::FromElementsOp::create(
          rewriter, loc, sizeTensor.getType(), scalar);
    }

    auto getTiedDim = [&](DimKey key) -> Value {
      for (const auto &[tiedKey, dim] : tiedDims) {
        if (tiedKey == key) {
          return dim;
        }
      }
      return {};
    };
    llvm::MapVector<Value, Value> tiedTensors;
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
        Value tiedDim = getTiedDim({tensorValue, i});
        dynamicDims.push_back(
            tiedDim ? tiedDim
                    : tensor::DimOp::create(rewriter, loc, tensorValue, i));
      }
      tiedTensors[tensorValue] = IREE::Flow::TensorTieShapeOp::create(
          rewriter, loc, tensorType, tensorValue, dynamicDims);
    }

    for (auto &[tensorValue, uses] : tensorUses) {
      Operation *tieOp = tiedTensors[tensorValue].getDefiningOp();
      for (OpOperand *use : uses) {
        if (dominance.properlyDominates(tieOp, use->getOwner())) {
          use->set(tiedTensors[tensorValue]);
        }
      }
    }
    // A size widened before the assertions reads the bounded size widened.
    SmallVector<std::pair<mlir::stablehlo::ConvertOp, SmallVector<OpOperand *>>>
        widenedUses;
    for (auto &[sizeValue, uses] : sizeUses) {
      for (OpOperand *use : uses) {
        auto widen = dyn_cast<mlir::stablehlo::ConvertOp>(use->getOwner());
        if (!widen || !isWidenedSize(widen)) {
          continue;
        }
        SmallVector<OpOperand *> later;
        for (OpOperand &widenUse : widen.getResult().getUses()) {
          if (dominance.properlyDominates(lastAssertion, widenUse.getOwner())) {
            later.push_back(&widenUse);
          }
        }
        if (!later.empty()) {
          widenedUses.push_back({widen, later});
        }
      }
    }
    for (auto &[sizeValue, uses] : sizeUses) {
      Value canonicalSize = canonicalSizeTensorForQuery[sizeValue];
      for (OpOperand *use : uses) {
        if (dominance.properlyDominates(lastAssertion, use->getOwner())) {
          use->set(canonicalSize);
        }
      }
    }
    for (auto &[widen, uses] : widenedUses) {
      Value bounded = mlir::stablehlo::ConvertOp::create(
          rewriter, loc, widen.getType(),
          canonicalSizeTensorForQuery[widen.getOperand()]);
      for (OpOperand *use : uses) {
        use->set(bounded);
      }
    }
  }
}

static LogicalResult rewriteShapeAssertion(mlir::stablehlo::CustomCallOp op,
                                           PatternRewriter &rewriter) {
  if (op.getNumOperands() == 0 || op.getNumResults() != 0) {
    return rewriter.notifyMatchFailure(op,
                                       "expected a predicate and no results");
  }
  auto predicateType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  if (!predicateType || predicateType.getRank() != 0 ||
      !predicateType.getElementType().isInteger(1)) {
    return rewriter.notifyMatchFailure(op, "expected a scalar i1 predicate");
  }
  auto message = op->getAttrOfType<StringAttr>("error_message");
  if (!message) {
    return rewriter.notifyMatchFailure(op, "expected an error_message string");
  }
  Value predicate = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                              op.getOperand(0), ValueRange{});
  // Semantic assertions must survive executable debug stripping.
  emitStatusCheck(rewriter, op.getLoc(), predicate, message.getValue());
  rewriter.eraseOp(op);
  return success();
}

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

    RewritePatternSet patterns(ctx);
    patterns.add<HouseholderReflectorRewriter, ShapeAssertionLowering>(ctx);
    if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::stablehlo
