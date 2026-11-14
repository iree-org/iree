// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings that take precedence over the upstream StableHLO-to-Linalg
// patterns: ops whose shapes are known only at runtime, pads with negative or
// interior padding, and gathers and slices whose indices are wider than the
// index type or unsigned.

#include <algorithm>
#include <functional>
#include <optional>

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "compiler/plugins/input/StableHLO/Conversion/StableHLOPaddingUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

Value extractIndex(OpBuilder &b, Location loc, Value shapeTensor, int64_t i) {
  Value index = arith::ConstantIndexOp::create(b, loc, i);
  Value element = tensor::ExtractOp::create(b, loc, shapeTensor, index);
  if (element.getType().isIndex()) {
    return element;
  }
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), element);
}

// Input conversion does not know the target's index width. Every target has
// at least this many bits, so decisions that depend on the width assume it.
constexpr unsigned kNarrowestIndexWidth = 32;

// Converts integer `value` to index, extending by its signedness.
Value castIntegerToIndex(OpBuilder &builder, Location loc, Value value,
                         bool isUnsigned) {
  Type indexType = builder.getIndexType();
  if (isUnsigned) {
    return arith::IndexCastUIOp::create(builder, loc, indexType, value);
  }
  return arith::IndexCastOp::create(builder, loc, indexType, value);
}

// Clamps `rawIndex`, a start index whose source element type is
// `originalElementType`, to [0, `upperBound`] and returns it as an index, for
// any index width. An index of at most kNarrowestIndexWidth bits converts to
// index exactly; an unsigned one is then clamped as unsigned, because a large
// value reads as negative in a 32-bit index. A wider index is clamped in its
// own type first, so that the conversion cannot wrap it into range.
Value clampGatherIndex(OpBuilder &builder, Location loc, Value rawIndex,
                       Type originalElementType, Value upperBound) {
  auto rawType = cast<IntegerType>(rawIndex.getType());
  bool isUnsigned = originalElementType.isUnsignedInteger();
  if (rawType.getWidth() <= kNarrowestIndexWidth) {
    Value index = castIntegerToIndex(builder, loc, rawIndex, isUnsigned);
    if (isUnsigned) {
      return arith::MinUIOp::create(builder, loc, index, upperBound);
    }
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value nonnegative = arith::MaxSIOp::create(builder, loc, index, zero);
    return arith::MinSIOp::create(builder, loc, nonnegative, upperBound);
  }
  Value clamped;
  if (isUnsigned) {
    Value rawUpperBound =
        arith::IndexCastUIOp::create(builder, loc, rawType, upperBound);
    clamped = arith::MinUIOp::create(builder, loc, rawIndex, rawUpperBound);
  } else {
    Value zero =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(rawType, 0));
    Value rawUpperBound =
        arith::IndexCastOp::create(builder, loc, rawType, upperBound);
    Value nonnegative = arith::MaxSIOp::create(builder, loc, rawIndex, zero);
    clamped = arith::MinSIOp::create(builder, loc, nonnegative, rawUpperBound);
  }
  // The clamped value lies in [0, upperBound], so it fits the index type.
  return castIntegerToIndex(builder, loc, clamped, isUnsigned);
}

// An opaque size: a tensor dimension, or a scalar with dimension -1. A nonzero
// width reads its low `width` bits, sign-extended.
struct SizeSymbol {
  Value value;
  int64_t dim;
  unsigned width = 0;
  // Known to be at least 1; not part of the symbol's identity.
  bool positive = false;
  // A known upper bound; not part of the symbol's identity.
  std::optional<uint64_t> umax;

  bool operator==(const SizeSymbol &other) const {
    return value == other.value && dim == other.dim && width == other.width;
  }
};

// An opaque size raised to a positive power.
struct SizeFactor {
  SizeFactor(SizeSymbol symbol, int64_t power = 1)
      : symbol(symbol), power(power) {}

  SizeSymbol symbol;
  int64_t power;
};

// A size as a constant times a product of opaque sizes, plus an offset. Each
// distinct symbol appears in one factor, so repeated multiplication by the
// same size grows a power rather than the factor list.
// `mayWrap` marks a product or sum taken in fewer than 64 bits, read as exact
// without proof. A size with an offset names itself as `whole`, the opaque
// symbol it becomes in a product. A constant size has no offset.
struct SizeProduct {
  int64_t constant = 1;
  SmallVector<SizeFactor> symbols;
  bool mayWrap = false;
  int64_t offset = 0;
  std::optional<SizeSymbol> whole;

  bool isOne() const { return constant == 1 && symbols.empty() && offset == 0; }
  // Returns the symbol when the product is that one symbol to the power one,
  // possibly scaled, or null otherwise.
  const SizeSymbol *getSingleSymbol() const {
    if (symbols.size() != 1 || symbols.front().power != 1) {
      return nullptr;
    }
    return &symbols.front().symbol;
  }
  bool isNonzero() const {
    if (symbols.empty()) {
      return constant != 0;
    }
    bool positive = llvm::all_of(symbols, [](const SizeFactor &factor) {
      return factor.symbol.positive;
    });
    return positive &&
           (offset == 0 ? constant != 0 : constant > 0 && offset >= 0);
  }
};

// Returns `size` as a product, with a sum as its opaque symbol.
SizeProduct asProduct(const SizeProduct &size) {
  if (size.offset == 0) {
    return size;
  }
  return SizeProduct{1, {*size.whole}, size.mayWrap};
}

// Adds two sizes when one of them is constant; other sums are not modelled.
std::optional<SizeProduct> addSizes(const SizeProduct &lhs,
                                    const SizeProduct &rhs, SizeSymbol whole) {
  const SizeProduct *size = &lhs;
  const SizeProduct *constant = &rhs;
  if (!constant->symbols.empty()) {
    std::swap(size, constant);
  }
  if (!constant->symbols.empty()) {
    return std::nullopt;
  }
  bool mayWrap = size->mayWrap || constant->mayWrap;
  if (size->symbols.empty()) {
    std::optional<int64_t> sum =
        llvm::checkedAdd(size->constant, constant->constant);
    if (!sum) {
      return std::nullopt;
    }
    return SizeProduct{*sum, {}, mayWrap};
  }
  std::optional<int64_t> offset =
      llvm::checkedAdd(size->offset, constant->constant);
  if (!offset) {
    return std::nullopt;
  }
  SizeProduct sum = *size;
  sum.mayWrap = mayWrap;
  sum.offset = *offset;
  sum.whole = whole;
  return sum;
}

std::optional<SizeProduct> multiplySizes(const SizeProduct &lhsSize,
                                         const SizeProduct &rhsSize) {
  // A sum keeps its form only when multiplied by one.
  if (lhsSize.isOne()) {
    return rhsSize;
  }
  if (rhsSize.isOne()) {
    return lhsSize;
  }
  SizeProduct lhs = asProduct(lhsSize);
  SizeProduct rhs = asProduct(rhsSize);
  std::optional<int64_t> constant =
      llvm::checkedMul(lhs.constant, rhs.constant);
  if (!constant) {
    return std::nullopt;
  }
  SizeProduct product{*constant, lhs.symbols, lhs.mayWrap || rhs.mayWrap};
  for (const SizeFactor &factor : rhs.symbols) {
    auto it = llvm::find_if(product.symbols, [&](const SizeFactor &existing) {
      return existing.symbol == factor.symbol;
    });
    if (it == product.symbols.end()) {
      product.symbols.push_back(factor);
      continue;
    }
    std::optional<int64_t> power = llvm::checkedAdd(it->power, factor.power);
    if (!power) {
      return std::nullopt;
    }
    it->power = *power;
    // Both factors name the same size, so their range facts combine.
    it->symbol.positive |= factor.symbol.positive;
    if (factor.symbol.umax) {
      it->symbol.umax =
          std::min(it->symbol.umax.value_or(UINT64_MAX), *factor.symbol.umax);
    }
  }
  return product;
}

// Returns whether every factor of `part` divides a factor of `whole`.
bool containsSymbols(ArrayRef<SizeFactor> part, ArrayRef<SizeFactor> whole) {
  return llvm::all_of(part, [whole](const SizeFactor &factor) {
    return llvm::any_of(whole, [&factor](const SizeFactor &candidate) {
      return candidate.symbol == factor.symbol &&
             candidate.power >= factor.power;
    });
  });
}

bool equalSizes(const SizeProduct &lhs, const SizeProduct &rhs) {
  return lhs.constant == rhs.constant && lhs.offset == rhs.offset &&
         lhs.symbols.size() == rhs.symbols.size() &&
         containsSymbols(lhs.symbols, rhs.symbols) &&
         containsSymbols(rhs.symbols, lhs.symbols);
}

// Returns whether `part` is a factor of `whole`, symbol for symbol.
bool dividesSize(const SizeProduct &partSize, const SizeProduct &wholeSize) {
  if (partSize.isOne() || equalSizes(partSize, wholeSize)) {
    return true;
  }
  SizeProduct part = asProduct(partSize);
  SizeProduct whole = asProduct(wholeSize);
  return part.constant != 0 && whole.constant % part.constant == 0 &&
         containsSymbols(part.symbols, whole.symbols);
}

// Returns an opaque size standing for the scalar `value`.
SizeProduct getOpaqueSize(Value value) {
  return SizeProduct{1, {SizeSymbol{value, -1}}};
}

// Returns whether a sum of one positive, bounded symbol stays within the
// signed range of `width` bits, so truncating it changes no value.
bool isSumInSignedRange(const SizeProduct &size, unsigned width) {
  const SizeSymbol *single = size.getSingleSymbol();
  if (!single || size.constant <= 0) {
    return false;
  }
  // The symbol is at least 1, so the smallest value is the constant plus the
  // offset.
  std::optional<int64_t> smallest =
      llvm::checkedAdd(size.constant, size.offset);
  if (!smallest || *smallest < 0) {
    return false;
  }
  const SizeSymbol &symbol = *single;
  if (!symbol.positive || !symbol.umax ||
      *symbol.umax > static_cast<uint64_t>(INT64_MAX)) {
    return false;
  }
  std::optional<int64_t> scaled =
      llvm::checkedMul(size.constant, static_cast<int64_t>(*symbol.umax));
  std::optional<int64_t> largest =
      scaled ? llvm::checkedAdd(*scaled, size.offset) : std::nullopt;
  return largest && *largest < (int64_t{1} << (width - 1));
}

// Reads `size` truncated to `width` bits and sign-extended. Truncating equal
// sizes gives equal sizes; a product becomes opaque, as `truncated`, since
// truncation does not distribute over it. A sum of one bounded symbol is kept
// while its range fits in `width` bits.
SizeProduct truncateSize(const SizeProduct &size, unsigned width,
                         Value truncated) {
  if (size.symbols.empty()) {
    return SizeProduct{
        APInt(64, size.constant, /*isSigned=*/true).trunc(width).getSExtValue(),
        {}};
  }
  if (size.offset != 0) {
    return isSumInSignedRange(size, width) ? size : getOpaqueSize(truncated);
  }
  if (size.constant != 1 || !size.getSingleSymbol()) {
    return getOpaqueSize(truncated);
  }
  SizeSymbol symbol = *size.getSingleSymbol();
  if (symbol.width == 0 || symbol.width > width) {
    symbol.width = width;
  }
  return SizeProduct{1, {symbol}};
}

// Returns the width an index_cast to `type` truncates to, or nullopt for a
// cast that keeps every value: index is 64 bits wide here.
std::optional<unsigned> getTruncationWidth(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIndex() || elementType.getIntOrFloatBitWidth() >= 64) {
    return std::nullopt;
  }
  return elementType.getIntOrFloatBitWidth();
}

// Factors sizes into SizeProducts by following the IR that computes them.
// Sizes are evaluated from an explicit stack rather than by recursion: a size
// whose operand sizes are not yet factored names them, and is evaluated again
// once they are. Every factored size is memoised, so a size shared by several
// queries of one proof, such as an operand dimension that both sides of a
// reshape refer to, is factored once. Memoised sizes are invalid once the IR
// changes, so one factorizer serves one proof.
class SizeFactorizer {
public:
  // Factors an index or integer size through products, casts, assumptions and
  // 0-d tensor round trips. A narrowing cast keeps its width. A wrapped i32
  // product would break the reshape's element count or the broadcast's
  // operand size, so products are read as exact. Returns nullopt when a
  // constant or a power does not fit, or the size depends on itself.
  std::optional<SizeProduct> factorSize(Value value) {
    return evaluate({value, kScalarSize});
  }

  // Factors dimension `dim` of `tensor` through the ops that carry its size.
  std::optional<SizeProduct> factorDim(Value tensor, int64_t dim) {
    auto type = cast<RankedTensorType>(tensor.getType());
    if (!type.isDynamicDim(dim)) {
      return SizeProduct{type.getDimSize(dim), {}};
    }
    return evaluate({tensor, dim});
  }

private:
  using Result = std::optional<SizeProduct>;
  // A scalar size, keyed with kScalarSize, or a dimension of a tensor.
  using SizeKey = std::pair<Value, int64_t>;
  static constexpr int64_t kScalarSize = -1;

  // Factors `root` and every size it depends on, innermost first.
  Result evaluate(SizeKey root) {
    SmallVector<SizeKey> stack = {root};
    while (!stack.empty()) {
      SizeKey key = stack.back();
      if (factored.contains(key)) {
        stack.pop_back();
        continue;
      }
      pending.insert(key);
      unfactored.clear();
      Result size = factorKey(key);
      if (unfactored.empty()) {
        factored[key] = size;
        pending.erase(key);
        stack.pop_back();
        continue;
      }
      for (SizeKey operand : unfactored) {
        // Pending sizes are the ancestors of `key` on the stack, so a pending
        // operand is a cycle, which only graph regions can form.
        if (pending.contains(operand)) {
          return std::nullopt;
        }
        stack.push_back(operand);
      }
    }
    return factored.lookup(root);
  }

  // Returns the memoised factoring of `key`, or records `key` as unfactored
  // and returns nullopt, which the size being evaluated then discards.
  Result lookup(SizeKey key) {
    auto it = factored.find(key);
    if (it != factored.end()) {
      return it->second;
    }
    unfactored.push_back(key);
    return std::nullopt;
  }

  Result lookupSize(Value value) { return lookup({value, kScalarSize}); }

  Result lookupDim(Value tensor, int64_t dim) {
    auto type = cast<RankedTensorType>(tensor.getType());
    if (!type.isDynamicDim(dim)) {
      return SizeProduct{type.getDimSize(dim), {}};
    }
    return lookup({tensor, dim});
  }

  static SizeProduct getOpaqueDim(Value tensor, int64_t dim) {
    return SizeProduct{1, {SizeSymbol{tensor, dim}}};
  }

  Result factorKey(SizeKey key) {
    auto [value, dim] = key;
    if (dim == kScalarSize) {
      return factorScalar(value);
    }
    return factorDynamicDim(value, dim);
  }

  Result factorScalar(Value value) {
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
      std::optional<int64_t> size = constant.trySExtValue();
      if (!size) {
        return std::nullopt;
      }
      return SizeProduct{*size, {}};
    }
    Operation *producer = value.getDefiningOp();
    if (!producer) {
      return getOpaqueSize(value);
    }
    return llvm::TypeSwitch<Operation *, Result>(producer)
        .Case<arith::IndexCastOp, arith::TruncIOp>(
            [&](auto cast) { return factorTruncation(cast.getIn(), value); })
        .Case([&](arith::ExtSIOp extend) { return lookupSize(extend.getIn()); })
        .Case([&](IREE::Util::AssumeIntOp assume) {
          return factorAssumedSize(assume, value);
        })
        .Case([&](tensor::ExtractOp extract) {
          return factorExtractedElement(extract, value);
        })
        .Case([&](arith::MulIOp mul) { return factorProduct(mul); })
        .Case([&](arith::AddIOp add) { return factorSum(add); })
        .Case([&](tensor::DimOp dimOp) -> Result {
          if (std::optional<int64_t> index = dimOp.getConstantIndex()) {
            return lookupDim(dimOp.getSource(), *index);
          }
          return getOpaqueSize(value);
        })
        .Default([&](Operation *) { return getOpaqueSize(value); });
  }

  // Factors `value`, a cast of `input`, keeping a narrowing cast's width.
  Result factorTruncation(Value input, Value value) {
    Result size = lookupSize(input);
    std::optional<unsigned> width = getTruncationWidth(value.getType());
    if (!size || !width) {
      return size;
    }
    return truncateSize(*size, *width, value);
  }

  // Attaches the assumed range of `value` to its single opaque symbol.
  Result factorAssumedSize(IREE::Util::AssumeIntOp assume, Value value) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    Result size = lookupSize(assume.getOperand(index));
    bool isSingleSymbol = size && size->constant == 1 && size->offset == 0 &&
                          size->getSingleSymbol();
    if (!isSingleSymbol) {
      return size;
    }
    auto [umin, umax] = assume.getUnionedUnsignedRange(index);
    SizeSymbol &symbol = size->symbols.front().symbol;
    if (umin && *umin >= 1) {
      symbol.positive = true;
    }
    symbol.umax = umax;
    return size;
  }

  // Factors element `i` of a 0-d or 1-d size vector, as shape reification
  // builds it. Casts of the whole vector apply to the element, innermost
  // first.
  Result factorExtractedElement(tensor::ExtractOp extract, Value value) {
    std::optional<int64_t> index = 0;
    if (extract.getIndices().size() == 1) {
      index = getConstantIntValue(extract.getIndices().front());
    } else if (!extract.getIndices().empty()) {
      index = std::nullopt;
    }
    Value source = extract.getTensor();
    SmallVector<unsigned> widths;
    while (auto castOp = source.getDefiningOp<arith::IndexCastOp>()) {
      if (std::optional<unsigned> width =
              getTruncationWidth(castOp.getType())) {
        widths.push_back(*width);
      }
      source = castOp.getIn();
    }
    Result element = lookupVectorElement(source, index);
    if (!element) {
      return getOpaqueSize(value);
    }
    for (unsigned width : llvm::reverse(widths)) {
      element = truncateSize(*element, width, value);
    }
    return element;
  }

  // Returns element `index` of `vector`, built by from_elements or shape_of.
  Result lookupVectorElement(Value vector, std::optional<int64_t> index) {
    if (!index || *index < 0) {
      return std::nullopt;
    }
    if (auto fromElements = vector.getDefiningOp<tensor::FromElementsOp>()) {
      if (*index >= static_cast<int64_t>(fromElements.getNumOperands())) {
        return std::nullopt;
      }
      return lookupSize(fromElements.getOperand(*index));
    }
    if (auto shapeOf = vector.getDefiningOp<shape::ShapeOfOp>()) {
      return lookupDim(shapeOf.getArg(), *index);
    }
    return std::nullopt;
  }

  Result factorProduct(arith::MulIOp mul) {
    Result lhs = lookupSize(mul.getLhs());
    Result rhs = lookupSize(mul.getRhs());
    Result product = lhs && rhs ? multiplySizes(*lhs, *rhs) : std::nullopt;
    if (product && getTruncationWidth(mul.getType())) {
      product->mayWrap = true;
    }
    return product;
  }

  // Factors a sum with a constant term; other sums stay opaque.
  Result factorSum(arith::AddIOp add) {
    Result lhs = lookupSize(add.getLhs());
    Result rhs = lookupSize(add.getRhs());
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    Value value = add.getResult();
    Result sum = addSizes(*lhs, *rhs, SizeSymbol{value, kScalarSize});
    if (!sum) {
      return getOpaqueSize(value);
    }
    if (getTruncationWidth(add.getType())) {
      sum->mayWrap = true;
    }
    return sum;
  }

  // Factors dynamic dimension `dim` of `tensor` from the op defining it.
  Result factorDynamicDim(Value tensor, int64_t dim) {
    Operation *producer = tensor.getDefiningOp();
    if (!producer) {
      return getOpaqueDim(tensor, dim);
    }
    auto type = cast<RankedTensorType>(tensor.getType());
    return llvm::TypeSwitch<Operation *, Result>(producer)
        .Case([&](IREE::Flow::TensorTieShapeOp tie) {
          int64_t dynamicIndex = llvm::count_if(type.getShape().take_front(dim),
                                                ShapedType::isDynamic);
          return lookupSize(tie.getDynamicDims()[dynamicIndex]);
        })
        .Case([&](tensor::EmptyOp empty) {
          return lookupSize(empty.getDynamicSize(dim));
        })
        .Case([&](tensor::CastOp castOp) {
          return lookupDim(castOp.getSource(), dim);
        })
        .Case([&](mlir::stablehlo::TransposeOp transpose) {
          return lookupDim(transpose.getOperand(),
                           transpose.getPermutation()[dim]);
        })
        .Case([&](mlir::stablehlo::DotGeneralOp dot) {
          return lookupDotGeneralDim(dot, dim);
        })
        .Case([&](mlir::stablehlo::DotOp dot) {
          // Results are lhs but its last, then rhs but its first dimension.
          int64_t lhsFree =
              cast<RankedTensorType>(dot.getLhs().getType()).getRank() - 1;
          return dim < lhsFree ? lookupDim(dot.getLhs(), dim)
                               : lookupDim(dot.getRhs(), dim - lhsFree + 1);
        })
        .Case([&](mlir::stablehlo::ReduceOp reduce) {
          return lookupReduceDim(reduce, cast<OpResult>(tensor), dim);
        })
        .Case([&](mlir::stablehlo::DynamicReshapeOp reshape) {
          return lookupShapeVectorDim(reshape.getOutputShape(), tensor, dim);
        })
        .Case([&](tensor::ReshapeOp reshape) {
          return lookupShapeVectorDim(reshape.getShape(), tensor, dim);
        })
        .Case([&](mlir::stablehlo::DynamicBroadcastInDimOp broadcast) {
          return lookupShapeVectorDim(broadcast.getOutputDimensions(), tensor,
                                      dim);
        })
        .Case([&](tensor::CollapseShapeOp collapse) {
          return factorCollapsedDim(collapse, dim);
        })
        .Case([&](tensor::ExpandShapeOp expand) -> Result {
          OpFoldResult size = expand.getMixedOutputShape()[dim];
          if (auto value = dyn_cast<Value>(size)) {
            return lookupSize(value);
          }
          return getOpaqueDim(tensor, dim);
        })
        .Default([&](Operation *op) -> Result {
          // StableHLO operations may not be converted yet.
          if (op->hasTrait<OpTrait::SameOperandsAndResultShape>() &&
              op->getNumOperands() > 0) {
            return lookupDim(op->getOperand(0), dim);
          }
          if (auto destination = dyn_cast<DestinationStyleOpInterface>(op)) {
            return lookupDim(
                destination.getTiedOpOperand(cast<OpResult>(tensor))->get(),
                dim);
          }
          return getOpaqueDim(tensor, dim);
        });
  }

  // Results are batch, then free lhs, then free rhs dimensions.
  Result lookupDotGeneralDim(mlir::stablehlo::DotGeneralOp dot, int64_t dim) {
    auto numbers = dot.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatch = numbers.getLhsBatchingDimensions();
    if (dim < static_cast<int64_t>(lhsBatch.size())) {
      return lookupDim(dot.getLhs(), lhsBatch[dim]);
    }
    int64_t index = lhsBatch.size();
    for (auto [operand, batch, contracting] :
         {std::make_tuple(dot.getLhs(), lhsBatch,
                          numbers.getLhsContractingDimensions()),
          std::make_tuple(dot.getRhs(), numbers.getRhsBatchingDimensions(),
                          numbers.getRhsContractingDimensions())}) {
      int64_t rank = cast<RankedTensorType>(operand.getType()).getRank();
      for (int64_t i = 0; i < rank; ++i) {
        bool isFree = !llvm::is_contained(batch, i) &&
                      !llvm::is_contained(contracting, i);
        if (isFree && index++ == dim) {
          return lookupDim(operand, i);
        }
      }
    }
    return getOpaqueDim(dot.getResult(), dim);
  }

  // Results keep the inputs' dimensions that are not reduced.
  Result lookupReduceDim(mlir::stablehlo::ReduceOp reduce, OpResult result,
                         int64_t dim) {
    Value input = reduce.getInputs()[result.getResultNumber()];
    int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
    int64_t index = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (!llvm::is_contained(reduce.getDimensions(), i) && index++ == dim) {
        return lookupDim(input, i);
      }
    }
    return getOpaqueDim(result, dim);
  }

  // Returns dimension `dim` of `tensor` from the shape vector its producer
  // took, when that vector is assembled with from_elements.
  Result lookupShapeVectorDim(Value shape, Value tensor, int64_t dim) {
    auto fromElements = shape.getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements ||
        dim >= static_cast<int64_t>(fromElements.getNumOperands())) {
      return getOpaqueDim(tensor, dim);
    }
    return lookupSize(fromElements.getOperand(dim));
  }

  // A collapsed dimension is the product of its source dimensions. Every
  // source dimension is looked up before any is multiplied, so one pass
  // names all the unfactored ones.
  Result factorCollapsedDim(tensor::CollapseShapeOp collapse, int64_t dim) {
    SmallVector<ReassociationIndices> groups =
        collapse.getReassociationIndices();
    SmallVector<Result> factors;
    for (int64_t source : groups[dim]) {
      factors.push_back(lookupDim(collapse.getSrc(), source));
    }
    SizeProduct product;
    for (const Result &factor : factors) {
      Result extended = factor ? multiplySizes(product, *factor) : std::nullopt;
      if (!extended) {
        return std::nullopt;
      }
      product = *extended;
    }
    return product;
  }

  // Memoised factorings, including failures.
  DenseMap<SizeKey, Result> factored;
  // Sizes evaluated at least once and waiting for their operand sizes.
  llvm::DenseSet<SizeKey> pending;
  // The operand sizes the current evaluation found unfactored.
  SmallVector<SizeKey> unfactored;
};

// A run of operand dimensions and the run of result dimensions it reshapes
// to, with equal products. One run is a single dimension, so a group either
// collapses operand dimensions, expands one, or keeps one.
struct ReshapeGroup {
  int64_t numOperandDims = 1;
  int64_t numResultDims = 1;

  bool collapses() const { return numOperandDims > 1; }
  bool expands() const { return numResultDims > 1; }
};

// Matches the leading dimensions of `operandSizes` and `resultSizes` as one
// group, extending the side whose product divides the other's until the
// products are equal.
std::optional<ReshapeGroup>
matchReshapeGroup(ArrayRef<SizeProduct> operandSizes,
                  ArrayRef<SizeProduct> resultSizes) {
  ReshapeGroup group;
  SizeProduct operandProduct = operandSizes.front();
  SizeProduct resultProduct = resultSizes.front();
  while (!equalSizes(operandProduct, resultProduct)) {
    std::optional<SizeProduct> extended;
    if (!group.expands() && dividesSize(operandProduct, resultProduct) &&
        group.numOperandDims < static_cast<int64_t>(operandSizes.size())) {
      extended =
          multiplySizes(operandProduct, operandSizes[group.numOperandDims++]);
      operandProduct = extended.value_or(operandProduct);
    } else if (!group.collapses() &&
               dividesSize(resultProduct, operandProduct) &&
               group.numResultDims < static_cast<int64_t>(resultSizes.size())) {
      extended =
          multiplySizes(resultProduct, resultSizes[group.numResultDims++]);
      resultProduct = extended.value_or(resultProduct);
    }
    if (!extended) {
      return std::nullopt;
    }
  }
  return group;
}

// Adds trailing unit dimensions, which only one side can have left, to the
// last group. Returns false when they are not all units, or when adding them
// would make the last group both collapse and expand.
bool appendTrailingUnitDims(ArrayRef<SizeProduct> remainingOperandSizes,
                            ArrayRef<SizeProduct> remainingResultSizes,
                            MutableArrayRef<ReshapeGroup> groups) {
  auto isOne = [](const SizeProduct &size) { return size.isOne(); };
  if (remainingOperandSizes.empty() && remainingResultSizes.empty()) {
    return true;
  }
  if (groups.empty() || !llvm::all_of(remainingOperandSizes, isOne) ||
      !llvm::all_of(remainingResultSizes, isOne)) {
    return false;
  }
  ReshapeGroup &last = groups.back();
  if (!remainingOperandSizes.empty() && last.expands()) {
    return false;
  }
  if (!remainingResultSizes.empty() && last.collapses()) {
    return false;
  }
  last.numOperandDims += remainingOperandSizes.size();
  last.numResultDims += remainingResultSizes.size();
  return true;
}

// Groups operand and result dimensions contiguously and greedily, so that
// each group either collapses operand dimensions into one result dimension,
// expands one operand dimension into result dimensions, or keeps one.
std::optional<SmallVector<ReshapeGroup>>
groupReshapeDims(ArrayRef<SizeProduct> operandSizes,
                 ArrayRef<SizeProduct> resultSizes) {
  SmallVector<ReshapeGroup> groups;
  while (!operandSizes.empty() && !resultSizes.empty()) {
    std::optional<ReshapeGroup> group =
        matchReshapeGroup(operandSizes, resultSizes);
    if (!group) {
      return std::nullopt;
    }
    operandSizes = operandSizes.drop_front(group->numOperandDims);
    resultSizes = resultSizes.drop_front(group->numResultDims);
    groups.push_back(*group);
  }
  if (!appendTrailingUnitDims(operandSizes, resultSizes, groups)) {
    return std::nullopt;
  }
  return groups;
}

// Returns the factored size of every dimension of `tensor`.
std::optional<SmallVector<SizeProduct>>
factorAllDims(SizeFactorizer &factorizer, Value tensor) {
  SmallVector<SizeProduct> sizes;
  for (int64_t i = 0; i < cast<RankedTensorType>(tensor.getType()).getRank();
       ++i) {
    std::optional<SizeProduct> size = factorizer.factorDim(tensor, i);
    if (!size) {
      return std::nullopt;
    }
    sizes.push_back(*size);
  }
  return sizes;
}

// Returns the factored sizes in `shape`, a from_elements size vector, when
// each agrees with the static sizes of `resultType`.
std::optional<SmallVector<SizeProduct>>
factorShapeVector(SizeFactorizer &factorizer, tensor::FromElementsOp shape,
                  RankedTensorType resultType) {
  SmallVector<SizeProduct> sizes;
  for (auto [i, element] : llvm::enumerate(shape.getOperands())) {
    std::optional<SizeProduct> size = factorizer.factorSize(element);
    bool contradictsStaticSize =
        size && !resultType.isDynamicDim(i) &&
        !equalSizes(*size, SizeProduct{resultType.getDimSize(i), {}});
    if (!size || contradictsStaticSize) {
      return std::nullopt;
    }
    sizes.push_back(*size);
  }
  return sizes;
}

// Returns whether products that may have wrapped can still be compared as
// exact. They can when no size wrapped, or when every operand size is nonzero:
// an empty tensor reshapes to any sizes with a zero.
bool canCompareAsExact(ArrayRef<SizeProduct> operandSizes,
                       ArrayRef<SizeProduct> resultSizes) {
  auto mayWrap = [](const SizeProduct &size) { return size.mayWrap; };
  auto isNonzero = [](const SizeProduct &size) { return size.isNonzero(); };
  return (llvm::none_of(operandSizes, mayWrap) &&
          llvm::none_of(resultSizes, mayWrap)) ||
         llvm::all_of(operandSizes, isNonzero);
}

// Proves how reshaping `operand` to `shape`, a vector of sizes assembled with
// tensor.from_elements, groups their dimensions.
std::optional<SmallVector<ReshapeGroup>>
getProvenReshapeGroups(Value operand, Value shape,
                       RankedTensorType resultType) {
  auto fromElements = shape.getDefiningOp<tensor::FromElementsOp>();
  if (!fromElements || static_cast<int64_t>(fromElements.getNumOperands()) !=
                           resultType.getRank()) {
    return std::nullopt;
  }
  SizeFactorizer factorizer;
  std::optional<SmallVector<SizeProduct>> operandSizes =
      factorAllDims(factorizer, operand);
  if (!operandSizes) {
    return std::nullopt;
  }
  std::optional<SmallVector<SizeProduct>> resultSizes =
      factorShapeVector(factorizer, fromElements, resultType);
  if (!resultSizes || !canCompareAsExact(*operandSizes, *resultSizes)) {
    return std::nullopt;
  }
  return groupReshapeDims(*operandSizes, *resultSizes);
}

// Returns the groups of a pure collapse or expansion that `reassociation`
// describes, from the higher-rank side to the lower-rank one.
SmallVector<ReshapeGroup>
getReshapeGroups(ArrayRef<ReassociationIndices> reassociation, bool collapses) {
  SmallVector<ReshapeGroup> groups;
  for (const ReassociationIndices &indices : reassociation) {
    int64_t size = indices.size();
    groups.push_back(collapses ? ReshapeGroup{size, 1} : ReshapeGroup{1, size});
  }
  return groups;
}

// Returns, for each group, the indices of its dimensions in the intermediate
// shape: the result dimensions of a group that does not collapse, otherwise
// its operand dimensions. Groups that `isMerged` selects keep their indices
// together; every other dimension forms its own group.
SmallVector<ReassociationIndices>
getIntermediateReassociation(ArrayRef<ReshapeGroup> groups,
                             llvm::function_ref<bool(ReshapeGroup)> isMerged) {
  SmallVector<ReassociationIndices> reassociation;
  int64_t next = 0;
  for (ReshapeGroup group : groups) {
    int64_t size =
        group.collapses() ? group.numOperandDims : group.numResultDims;
    if (isMerged(group)) {
      reassociation.push_back(llvm::to_vector(llvm::seq(next, next + size)));
      next += size;
      continue;
    }
    for (int64_t i = 0; i < size; ++i) {
      reassociation.push_back({next++});
    }
  }
  return reassociation;
}

struct DynamicReshapeOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicReshapeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    Value operand = adaptor.getOperand();
    std::optional<SmallVector<ReshapeGroup>> groups =
        getGroups(operand, adaptor.getOutputShape(), resultType);
    if (!groups) {
      rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, operand,
                                                     adaptor.getOutputShape());
      return success();
    }
    rewriter.replaceOp(op, reshapeByGroups(rewriter, op.getLoc(), operand,
                                           adaptor.getOutputShape(), resultType,
                                           *groups));
    return success();
  }

private:
  // Returns how the reshape groups dimensions. Element count determines a
  // single dynamic dimension of a collapse or an expansion unless a static
  // dimension is zero; more dynamic dimensions need a size proof.
  static std::optional<SmallVector<ReshapeGroup>>
  getGroups(Value operand, Value outputShape, RankedTensorType resultType) {
    auto operandType = cast<RankedTensorType>(operand.getType());
    bool collapses = operandType.getRank() > resultType.getRank();
    auto collapsedType = collapses ? resultType : operandType;
    if (collapsedType.getNumDynamicDims() <= 1 &&
        !llvm::is_contained(collapsedType.getShape(), 0)) {
      if (auto reassociation =
              getReassociationIndicesForReshape(operandType, resultType)) {
        return getReshapeGroups(*reassociation, collapses);
      }
    }
    if (llvm::is_contained(operandType.getShape(), 0) ||
        llvm::is_contained(resultType.getShape(), 0)) {
      return std::nullopt;
    }
    return getProvenReshapeGroups(operand, outputShape, resultType);
  }

  // Reshapes `operand` to `resultType` by expanding the groups that expand
  // into an intermediate shape, then collapsing the groups that collapse.
  static Value reshapeByGroups(OpBuilder &builder, Location loc, Value operand,
                               Value outputShape, RankedTensorType resultType,
                               ArrayRef<ReshapeGroup> groups) {
    // A rank-0 side has no groups: the other side's dimensions are all units,
    // which a reshape from a scalar expands and one to a scalar collapses.
    int64_t operandRank = cast<RankedTensorType>(operand.getType()).getRank();
    bool anyExpands = operandRank < resultType.getRank() ||
                      llvm::any_of(groups, std::mem_fn(&ReshapeGroup::expands));
    bool anyCollapses =
        operandRank > resultType.getRank() ||
        llvm::any_of(groups, std::mem_fn(&ReshapeGroup::collapses));
    Value reshaped = operand;
    if (anyExpands) {
      RankedTensorType intermediateType =
          anyCollapses ? getIntermediateType(operand, resultType, groups)
                       : resultType;
      reshaped = tensor::ExpandShapeOp::create(
          builder, loc, intermediateType, operand,
          getIntermediateReassociation(groups,
                                       std::mem_fn(&ReshapeGroup::expands)),
          getIntermediateSizes(builder, loc, operand, outputShape, resultType,
                               groups));
    }
    if (anyCollapses) {
      reshaped = tensor::CollapseShapeOp::create(
          builder, loc, resultType, reshaped,
          getIntermediateReassociation(groups,
                                       std::mem_fn(&ReshapeGroup::collapses)));
    }
    if (reshaped.getType() != resultType) {
      reshaped = tensor::CastOp::create(builder, loc, resultType, reshaped);
    }
    return reshaped;
  }

  // Returns the intermediate shape's type: the result dimensions of groups
  // that do not collapse and the operand dimensions of groups that do.
  static RankedTensorType getIntermediateType(Value operand,
                                              RankedTensorType resultType,
                                              ArrayRef<ReshapeGroup> groups) {
    auto operandType = cast<RankedTensorType>(operand.getType());
    SmallVector<int64_t> shape;
    int64_t operandDim = 0;
    int64_t resultDim = 0;
    for (ReshapeGroup group : groups) {
      if (group.collapses()) {
        llvm::append_range(shape, operandType.getShape().slice(
                                      operandDim, group.numOperandDims));
      } else {
        llvm::append_range(
            shape, resultType.getShape().slice(resultDim, group.numResultDims));
      }
      operandDim += group.numOperandDims;
      resultDim += group.numResultDims;
    }
    return RankedTensorType::get(shape, resultType.getElementType(),
                                 resultType.getEncoding());
  }

  // Returns result dimension `i`: static, or read from the shape vector.
  static OpFoldResult getResultSize(OpBuilder &builder, Location loc,
                                    Value outputShape,
                                    RankedTensorType resultType, int64_t i) {
    if (!resultType.isDynamicDim(i)) {
      return builder.getIndexAttr(resultType.getDimSize(i));
    }
    return extractIndex(builder, loc, outputShape, i);
  }

  // Returns the intermediate sizes, read from the reshape's shape vector for
  // result dimensions and from the operand for operand dimensions. Without
  // groups, from a scalar operand, every result dimension is intermediate.
  static SmallVector<OpFoldResult>
  getIntermediateSizes(OpBuilder &builder, Location loc, Value operand,
                       Value outputShape, RankedTensorType resultType,
                       ArrayRef<ReshapeGroup> groups) {
    SmallVector<OpFoldResult> sizes;
    int64_t operandDim = 0;
    int64_t resultDim = 0;
    for (ReshapeGroup group : groups) {
      if (group.collapses()) {
        for (int64_t i = 0; i < group.numOperandDims; ++i) {
          sizes.push_back(
              tensor::getMixedSize(builder, loc, operand, operandDim + i));
        }
      } else {
        for (int64_t i = resultDim; i < resultDim + group.numResultDims; ++i) {
          sizes.push_back(
              getResultSize(builder, loc, outputShape, resultType, i));
        }
      }
      operandDim += group.numOperandDims;
      resultDim += group.numResultDims;
    }
    for (int64_t i = resultDim; i < resultType.getRank(); ++i) {
      sizes.push_back(getResultSize(builder, loc, outputShape, resultType, i));
    }
    return sizes;
  }
};

struct DynamicPadGeometry {
  SmallVector<Value> resultDynamicDims;
  SmallVector<Value> lowPadding, highPadding, interiorPadding;
  SmallVector<OpFoldResult> insertSizes, insertOffsets, insertStrides;
  bool canUseInsert;
  Value hasCrop;
  Value nonempty;
};

// A strided insert must not contain a constant stride or intermediate shape
// outside the index range, even in a branch which will not execute. Unknown
// interior padding and empty sources use output-driven generation instead.
static bool canInsertDimension(int64_t size, Value interiorPadding,
                               int64_t indexLimit) {
  if (size == 0) {
    return false;
  }
  if (size == 1) {
    return true;
  }
  std::optional<int64_t> interior = getConstantIntValue(interiorPadding);
  if (!interior || *interior < 0) {
    return false;
  }
  if (*interior == 0) {
    return true;
  }
  if (ShapedType::isDynamic(size)) {
    return false;
  }
  auto extra = llvm::checkedMul(size - 1, *interior);
  auto dilatedSize = extra ? llvm::checkedAdd(size, *extra) : std::nullopt;
  return dilatedSize && *dilatedSize <= indexLimit;
}

// The dilated operand is an intermediate that even a valid pad may make larger
// than its result, so it must fit the narrowest index type.
static bool canUseStridedInsertion(RankedTensorType inputType,
                                   ArrayRef<Value> interiorPadding) {
  int64_t limit = APInt::getSignedMaxValue(kNarrowestIndexWidth).getSExtValue();
  return llvm::all_of(llvm::zip_equal(inputType.getShape(), interiorPadding),
                      [limit](auto dimension) {
                        return canInsertDimension(std::get<0>(dimension),
                                                  std::get<1>(dimension),
                                                  limit);
                      });
}

static DynamicPadGeometry
buildDynamicPadGeometry(OpBuilder &rewriter, Location loc, Value operand,
                        ArrayRef<Value> lowPadding, ArrayRef<Value> highPadding,
                        ArrayRef<Value> interiorPadding, bool isUnsignedPadding,
                        RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  DynamicPadGeometry geometry;
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value zeroI64 = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
  geometry.nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
  geometry.hasCrop = rewriter.createOrFold<arith::ConstantIntOp>(loc, 0, 1);
  geometry.lowPadding.assign(lowPadding.begin(), lowPadding.end());
  geometry.highPadding.assign(highPadding.begin(), highPadding.end());
  geometry.interiorPadding.assign(interiorPadding.begin(),
                                  interiorPadding.end());
  geometry.canUseInsert = canUseStridedInsertion(
      cast<RankedTensorType>(operand.getType()), interiorPadding);
  geometry.insertSizes = tensor::getMixedSizes(rewriter, loc, operand);
  for (int64_t i = 0; i < rank; ++i) {
    Value dim =
        getValueOrCreateConstantIndexOp(rewriter, loc, geometry.insertSizes[i]);
    Value low = lowPadding[i];
    Value high = highPadding[i];
    Value interior = interiorPadding[i];
    // A valid pad never has a negative result size, so isNegative is unused.
    auto padded = buildPaddedDimension(rewriter, loc, dim, low, high, interior,
                                       isUnsignedPadding);
    if (!isUnsignedPadding) {
      Value negativeLow = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, low, zeroI64);
      Value negativeHigh = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, high, zeroI64);
      Value cropped =
          rewriter.createOrFold<arith::OrIOp>(loc, negativeLow, negativeHigh);
      geometry.hasCrop =
          rewriter.createOrFold<arith::OrIOp>(loc, geometry.hasCrop, cropped);
    }

    if (resultType.isDynamicDim(i)) {
      geometry.resultDynamicDims.push_back(padded.size);
    }

    // These coordinates are used only on the nonnegative-padding insert path.
    Value offset = rewriter.createOrFold<arith::IndexCastUIOp>(
        loc, rewriter.getIndexType(), low);
    geometry.insertOffsets.push_back(offset);
    Value stride = rewriter.createOrFold<arith::IndexCastUIOp>(
        loc, rewriter.getIndexType(), interior);
    stride = rewriter.createOrFold<arith::AddIOp>(loc, stride, one);
    Value singleElement = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, dim, one);
    stride =
        rewriter.createOrFold<arith::SelectOp>(loc, singleElement, one, stride);
    geometry.insertStrides.push_back(stride);
    Value size = resultType.isDynamicDim(i)
                     ? padded.size
                     : rewriter.createOrFold<arith::ConstantIndexOp>(
                           loc, resultType.getDimSize(i));
    Value positive = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, size, zero);
    geometry.nonempty =
        rewriter.createOrFold<arith::AndIOp>(loc, geometry.nonempty, positive);
  }
  return geometry;
}

static Value createPaddedTensor(OpBuilder &rewriter, Location loc,
                                Value operand, Value paddingValue,
                                RankedTensorType resultType,
                                const DynamicPadGeometry &geometry) {
  Value empty = tensor::EmptyOp::create(rewriter, loc, resultType,
                                        geometry.resultDynamicDims);
  Value filled =
      linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
  return tensor::InsertSliceOp::create(
             rewriter, loc, operand, filled, geometry.insertOffsets,
             geometry.insertSizes, geometry.insertStrides)
      .getResult();
}

// Where one dimension of a pad reads its operand, in index values bounded by
// the result size, so that device code needs no 64-bit arithmetic. Output
// position d reads operand element `j - firstRead + firstSource` when
// d = firstAligned + j * stride for some j in [firstRead, endRead), and the
// padding value otherwise.
struct PadReadPhase {
  Value firstAligned;
  Value stride;
  Value firstRead;
  Value endRead;
  Value firstSource;
};

// Derives the read phase of one dimension on the host, exactly in i64, from
// the operand size `in`, the result size `result`, and the low and interior
// padding. Output position d is dilated position d - low, which holds operand
// element i when d - low = i * s with s = interior + 1, possibly 2^64 for an
// unsigned interior. With q = floor(low / s), the aligned positions are
// d = (low - q * s) + j * s and read element j - q, which is in range when
// q <= j < in + q. Positions and counts are clamped to the result size, and a
// stride larger than the result can align only its first position.
static PadReadPhase buildPadReadPhase(ImplicitLocOpBuilder &b, Value in,
                                      Value result, Value low, Value interior,
                                      bool isUnsignedPadding) {
  Type i64 = b.getI64Type();
  Value zero = b.create<arith::ConstantIntOp>(0, 64);
  Value one = b.create<arith::ConstantIntOp>(1, 64);
  Value in64 = b.createOrFold<arith::IndexCastUIOp>(i64, in);
  Value result64 = b.createOrFold<arith::IndexCastUIOp>(i64, result);
  // A wrapped stride denotes 2^64, which exceeds every unsigned magnitude.
  Value stride = b.createOrFold<arith::AddIOp>(interior, one);
  Value wrapped =
      b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::eq, stride, zero);
  Value divisor = b.createOrFold<arith::SelectOp>(wrapped, one, stride);

  // Split low into its sign and unsigned magnitude, and divide the magnitude.
  Value isNegative = b.create<arith::ConstantIntOp>(0, 1);
  Value magnitude = low;
  if (!isUnsignedPadding) {
    isNegative =
        b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::slt, low, zero);
    Value negated = b.createOrFold<arith::SubIOp>(zero, low);
    magnitude = b.createOrFold<arith::SelectOp>(isNegative, negated, low);
  }
  Value quotient = b.createOrFold<arith::DivUIOp>(magnitude, divisor);
  Value remainder = b.createOrFold<arith::RemUIOp>(magnitude, divisor);
  quotient = b.createOrFold<arith::SelectOp>(wrapped, zero, quotient);
  remainder = b.createOrFold<arith::SelectOp>(wrapped, magnitude, remainder);
  // For a negative low, floor(low / s) = -(quotient + (remainder != 0)) and
  // the first aligned position is s - remainder, or 0 for no remainder.
  Value inexact =
      b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::ne, remainder, zero);
  Value roundedUp = b.createOrFold<arith::AddIOp>(
      quotient, b.createOrFold<arith::ExtUIOp>(i64, inexact));
  Value floorMagnitude =
      b.createOrFold<arith::SelectOp>(isNegative, roundedUp, quotient);
  Value negativeAligned = b.createOrFold<arith::SelectOp>(
      inexact, b.createOrFold<arith::SubIOp>(stride, remainder), zero);
  Value aligned =
      b.createOrFold<arith::SelectOp>(isNegative, negativeAligned, remainder);

  // Clamp to the result size. Every value here is nonnegative, so the
  // comparisons are unsigned.
  auto minimum = [&b](Value lhs, Value rhs) {
    return b.createOrFold<arith::MinUIOp>(lhs, rhs);
  };
  Value firstAligned = minimum(aligned, result64);
  Value clampedStride = b.createOrFold<arith::SelectOp>(
      wrapped, result64, minimum(stride, result64));
  clampedStride = b.createOrFold<arith::MaxUIOp>(clampedStride, one);
  // Element j - q is read for q <= j < in + q, where q = floor(low / s).
  Value firstRead = b.createOrFold<arith::SelectOp>(
      isNegative, zero, minimum(floorMagnitude, result64));
  Value firstSource = b.createOrFold<arith::SelectOp>(
      isNegative, minimum(floorMagnitude, in64), zero);
  Value passesOperand = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::uge,
                                                      floorMagnitude, in64);
  Value negativeEnd = b.createOrFold<arith::SelectOp>(
      passesOperand, zero,
      minimum(b.createOrFold<arith::SubIOp>(in64, floorMagnitude), result64));
  Value pastResult = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::uge,
                                                   floorMagnitude, result64);
  Value positiveEnd = b.createOrFold<arith::SelectOp>(
      pastResult, result64,
      minimum(b.createOrFold<arith::AddIOp>(in64, floorMagnitude), result64));
  Value endRead =
      b.createOrFold<arith::SelectOp>(isNegative, negativeEnd, positiveEnd);

  auto toIndex = [&b](Value value) {
    return b.createOrFold<arith::IndexCastUIOp>(b.getIndexType(), value);
  };
  return {toIndex(firstAligned), toIndex(clampedStride), toIndex(firstRead),
          toIndex(endRead), toIndex(firstSource)};
}

// Reads one output element from the read phases, in index arithmetic.
static Value buildCroppedPadElement(OpBuilder &builder, Location loc,
                                    Value operand, Value paddingValue,
                                    ArrayRef<PadReadPhase> phases) {
  ImplicitLocOpBuilder b(loc, builder);
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value inBounds = b.create<arith::ConstantIntOp>(1, 1);
  SmallVector<Value> coordinates;
  for (auto [d, phase] : llvm::enumerate(phases)) {
    Value output = b.create<linalg::IndexOp>(d);
    Value afterFirst = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::uge, output, phase.firstAligned);
    Value offset = b.createOrFold<arith::SubIOp>(output, phase.firstAligned);
    offset = b.createOrFold<arith::SelectOp>(afterFirst, offset, zero);
    Value step = b.createOrFold<arith::DivUIOp>(offset, phase.stride);
    Value remainder = b.createOrFold<arith::RemUIOp>(offset, phase.stride);
    Value aligned = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                  remainder, zero);
    Value fromFirst = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::uge,
                                                    step, phase.firstRead);
    Value beforeEnd = b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::ult,
                                                    step, phase.endRead);
    Value valid = b.createOrFold<arith::AndIOp>(afterFirst, aligned);
    valid = b.createOrFold<arith::AndIOp>(valid, fromFirst);
    valid = b.createOrFold<arith::AndIOp>(valid, beforeEnd);
    inBounds = b.createOrFold<arith::AndIOp>(inBounds, valid);
    Value source = b.createOrFold<arith::SubIOp>(step, phase.firstRead);
    coordinates.push_back(
        b.createOrFold<arith::AddIOp>(source, phase.firstSource));
  }
  auto ifOp =
      b.create<scf::IfOp>(TypeRange{paddingValue.getType()}, inBounds, true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value element = b.create<tensor::ExtractOp>(operand, coordinates);
    b.create<scf::YieldOp>(element);
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    b.create<scf::YieldOp>(paddingValue);
  }
  return ifOp.getResult(0);
}

// Generate only the final padded shape, including crops across dilation gaps.
// The read phases are computed outside the generated loop, so the 64-bit
// arithmetic they need stays on the host.
static Value createOutputDrivenPad(OpBuilder &builder, Location loc,
                                   Value operand, Value paddingValue,
                                   RankedTensorType resultType,
                                   const DynamicPadGeometry &geometry,
                                   bool isUnsignedPadding) {
  ImplicitLocOpBuilder b(loc, builder);
  Value empty =
      b.create<tensor::EmptyOp>(resultType, geometry.resultDynamicDims);
  SmallVector<PadReadPhase> phases;
  for (int64_t d = 0; d < resultType.getRank(); ++d) {
    Value in = getValueOrCreateConstantIndexOp(b, loc, geometry.insertSizes[d]);
    Value result = b.createOrFold<tensor::DimOp>(empty, d);
    phases.push_back(buildPadReadPhase(b, in, result, geometry.lowPadding[d],
                                       geometry.interiorPadding[d],
                                       isUnsignedPadding));
  }
  return linalg::GenericOp::create(
             builder, loc, TypeRange{resultType}, ValueRange{},
             ValueRange{empty},
             SmallVector<AffineMap>{
                 builder.getMultiDimIdentityMap(resultType.getRank())},
             mlir::stablehlo::getNParallelLoopsAttrs(resultType.getRank()),
             [operand, paddingValue, &phases](OpBuilder &nested,
                                              Location nestedLoc, ValueRange) {
               Value element = buildCroppedPadElement(
                   nested, nestedLoc, operand, paddingValue, phases);
               linalg::YieldOp::create(nested, nestedLoc, element);
             })
      .getResult(0);
}

// Returns whether a possibly cropping pad has no interior padding, so that
// each output dimension is a contiguous window of the operand framed by
// padding. A pad known not to crop keeps the strided insertion. A slice of a
// static empty dimension has no valid static offset, so such an operand keeps
// the output-driven form.
static bool canUseSlicedPad(Value operand, const DynamicPadGeometry &geometry,
                            bool isUnsignedPadding) {
  if (llvm::is_contained(cast<RankedTensorType>(operand.getType()).getShape(),
                         0)) {
    return false;
  }
  return !isUnsignedPadding && !matchPattern(geometry.hasCrop, m_Zero()) &&
         llvm::all_of(geometry.interiorPadding, [](Value interior) {
           return matchPattern(interior, m_Zero());
         });
}

// Pads without interior padding as the kept window of the operand, padded on
// both sides. Negative edges crop, and a crop may pass the whole operand and
// remove padding from the other edge. Per dimension, with input size `in`:
//   cropLow = max(-low, 0), cropHigh = max(-high, 0)
//   kept = max(in - cropLow - cropHigh, 0)
//   offset = kept == 0 ? 0 : cropLow, which keeps an empty slice in bounds
//   result = in + low + high
//   padLow = min(max(low, 0), result), padHigh = result - padLow - kept
// For example, [a, b, c] with low = 5 and high = -6 keeps no element and pads
// two elements. The input is trusted to have a nonnegative result size.
static Value createSlicedPad(OpBuilder &builder, Location loc, Value operand,
                             Value paddingValue, RankedTensorType resultType,
                             const DynamicPadGeometry &geometry) {
  ImplicitLocOpBuilder b(loc, builder);
  Value zero = b.create<arith::ConstantIndexOp>(0);
  SmallVector<OpFoldResult> offsets, sizes, padLows, padHighs;
  for (auto [size, lowI64, highI64] : llvm::zip_equal(
           geometry.insertSizes, geometry.lowPadding, geometry.highPadding)) {
    Value in = getValueOrCreateConstantIndexOp(b, loc, size);
    Value low = b.createOrFold<arith::IndexCastOp>(b.getIndexType(), lowI64);
    Value high = b.createOrFold<arith::IndexCastOp>(b.getIndexType(), highI64);
    Value cropLow = b.createOrFold<arith::MaxSIOp>(
        b.createOrFold<arith::SubIOp>(zero, low), zero);
    Value cropHigh = b.createOrFold<arith::MaxSIOp>(
        b.createOrFold<arith::SubIOp>(zero, high), zero);
    Value kept = b.createOrFold<arith::SubIOp>(
        b.createOrFold<arith::SubIOp>(in, cropLow), cropHigh);
    kept = b.createOrFold<arith::MaxSIOp>(kept, zero);
    Value keepsNothing =
        b.createOrFold<arith::CmpIOp>(arith::CmpIPredicate::eq, kept, zero);
    Value offset = b.createOrFold<arith::SelectOp>(keepsNothing, zero, cropLow);
    Value result = b.createOrFold<arith::AddIOp>(
        b.createOrFold<arith::AddIOp>(in, low), high);
    Value padLow = b.createOrFold<arith::MinSIOp>(
        b.createOrFold<arith::MaxSIOp>(low, zero), result);
    Value padHigh = b.createOrFold<arith::SubIOp>(
        b.createOrFold<arith::SubIOp>(result, padLow), kept);
    offsets.push_back(getAsOpFoldResult(offset));
    sizes.push_back(getAsOpFoldResult(kept));
    padLows.push_back(getAsOpFoldResult(padLow));
    padHighs.push_back(getAsOpFoldResult(padHigh));
  }
  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  Value padded =
      b.create<tensor::ExtractSliceOp>(operand, offsets, sizes, strides);
  if (!llvm::all_of(llvm::concat<OpFoldResult>(padLows, padHighs),
                    isZeroInteger)) {
    padded = b.create<tensor::PadOp>(Type(), padded, padLows, padHighs,
                                     paddingValue);
  }
  if (padded.getType() != resultType) {
    padded = b.create<tensor::CastOp>(resultType, padded);
  }
  return padded;
}

static Value createNonemptyPad(OpBuilder &builder, Location loc, Value operand,
                               Value paddingValue, RankedTensorType resultType,
                               const DynamicPadGeometry &geometry,
                               bool isUnsignedPadding) {
  if (!geometry.canUseInsert || matchPattern(geometry.hasCrop, m_One())) {
    return createOutputDrivenPad(builder, loc, operand, paddingValue,
                                 resultType, geometry, isUnsignedPadding);
  }
  if (matchPattern(geometry.hasCrop, m_Zero())) {
    return createPaddedTensor(builder, loc, operand, paddingValue, resultType,
                              geometry);
  }
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{resultType},
                                geometry.hasCrop, true);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value cropped =
        createOutputDrivenPad(builder, loc, operand, paddingValue, resultType,
                              geometry, isUnsignedPadding);
    scf::YieldOp::create(builder, loc, cropped);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    Value inserted = createPaddedTensor(builder, loc, operand, paddingValue,
                                        resultType, geometry);
    scf::YieldOp::create(builder, loc, inserted);
  }
  return ifOp.getResult(0);
}

static Value createCroppedPad(OpBuilder &builder, Location loc, Value operand,
                              Value paddingValue, RankedTensorType resultType,
                              const DynamicPadGeometry &geometry,
                              bool isUnsignedPadding) {
  Value result;
  if (canUseSlicedPad(operand, geometry, isUnsignedPadding)) {
    // An empty result is a slice and pad of zero size, so it needs no guard.
    result = createSlicedPad(builder, loc, operand, paddingValue, resultType,
                             geometry);
  } else if (matchPattern(geometry.nonempty, m_Zero())) {
    result = tensor::EmptyOp::create(builder, loc, resultType,
                                     geometry.resultDynamicDims);
  } else if (matchPattern(geometry.nonempty, m_One())) {
    result = createNonemptyPad(builder, loc, operand, paddingValue, resultType,
                               geometry, isUnsignedPadding);
  } else {
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{resultType},
                                  geometry.nonempty, true);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value padded = createNonemptyPad(builder, loc, operand, paddingValue,
                                       resultType, geometry, isUnsignedPadding);
      scf::YieldOp::create(builder, loc, padded);
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value empty = tensor::EmptyOp::create(builder, loc, resultType,
                                            geometry.resultDynamicDims);
      scf::YieldOp::create(builder, loc, empty);
    }
    result = ifOp.getResult(0);
  }
  return IREE::Flow::TensorTieShapeOp::create(builder, loc, resultType, result,
                                              geometry.resultDynamicDims);
}

// Keep padding integers in i64 storage until the exact output extent and
// guarded source coordinates have been computed.
static SmallVector<Value> extractPaddingElements(OpBuilder &builder,
                                                 Location loc, Value padding,
                                                 bool isUnsignedPadding,
                                                 int64_t rank) {
  SmallVector<Value> elements;
  for (int64_t d = 0; d < rank; ++d) {
    Value index = arith::ConstantIndexOp::create(builder, loc, d);
    Value element =
        builder.createOrFold<tensor::ExtractOp>(loc, padding, index);
    if (element.getType().getIntOrFloatBitWidth() < 64) {
      if (isUnsignedPadding) {
        element = builder.createOrFold<arith::ExtUIOp>(
            loc, builder.getI64Type(), element);
      } else {
        element = builder.createOrFold<arith::ExtSIOp>(
            loc, builder.getI64Type(), element);
      }
    }
    elements.push_back(element);
  }
  return elements;
}

struct DynamicPadOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicPadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicPadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    auto operandType =
        dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
    if (!resultType || !operandType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    bool isUnsignedPadding = cast<ShapedType>(op.getEdgePaddingLow().getType())
                                 .getElementType()
                                 .isUnsignedInteger();
    auto lows =
        extractPaddingElements(rewriter, loc, adaptor.getEdgePaddingLow(),
                               isUnsignedPadding, resultType.getRank());
    auto highs =
        extractPaddingElements(rewriter, loc, adaptor.getEdgePaddingHigh(),
                               isUnsignedPadding, resultType.getRank());
    auto interiors =
        extractPaddingElements(rewriter, loc, adaptor.getInteriorPadding(),
                               isUnsignedPadding, resultType.getRank());
    auto geometry = buildDynamicPadGeometry(rewriter, loc, adaptor.getOperand(),
                                            lows, highs, interiors,
                                            isUnsignedPadding, resultType);
    rewriter.replaceOp(op, createCroppedPad(rewriter, loc, adaptor.getOperand(),
                                            paddingValue, resultType, geometry,
                                            isUnsignedPadding));
    return success();
  }
};

struct PadOpConversion final : OpConversionPattern<mlir::stablehlo::PadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    auto isNegative = [](int64_t v) { return v < 0; };
    if (llvm::any_of(op.getInteriorPadding(),
                     [](int64_t v) { return v != 0; }) ||
        llvm::any_of(op.getEdgePaddingLow(), isNegative) ||
        llvm::any_of(op.getEdgePaddingHigh(), isNegative)) {
      // tensor.pad cannot represent cropping or interior padding.
      Location loc = op.getLoc();
      auto toValues = [&](ArrayRef<int64_t> values) {
        return llvm::map_to_vector(values, [&](int64_t value) -> Value {
          return arith::ConstantIntOp::create(rewriter, loc, value, 64);
        });
      };
      DynamicPadGeometry geometry = buildDynamicPadGeometry(
          rewriter, loc, adaptor.getOperand(), toValues(op.getEdgePaddingLow()),
          toValues(op.getEdgePaddingHigh()), toValues(op.getInteriorPadding()),
          /*isUnsignedPadding=*/false, resultType);
      Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
          loc, adaptor.getPaddingValue());
      rewriter.replaceOp(op,
                         createCroppedPad(rewriter, loc, adaptor.getOperand(),
                                          paddingValue, resultType, geometry,
                                          /*isUnsignedPadding=*/false));
      return success();
    }
    if (resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "upstream handles nonnegative static padding");
    }
    auto toIndexAttr = [&](int64_t size) -> OpFoldResult {
      return rewriter.getIndexAttr(size);
    };
    Value padding = tensor::ExtractOp::create(
        rewriter, op.getLoc(), adaptor.getPaddingValue(), ValueRange{});
    // tensor.pad requires the inferred shape, even if the declared shape is
    // less precise. Preserve the declaration with a cast after padding.
    Value padded = tensor::PadOp::create(
        rewriter, op.getLoc(), Type{}, adaptor.getOperand(),
        llvm::map_to_vector(op.getEdgePaddingLow(), toIndexAttr),
        llvm::map_to_vector(op.getEdgePaddingHigh(), toIndexAttr), padding);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, padded);
    return success();
  }
};

struct ConvolutionWindow {
  SmallVector<int64_t> strides, lhsDilations, rhsDilations, padding;
};

ConvolutionWindow getConvolutionWindow(mlir::stablehlo::ConvolutionOp op) {
  int64_t spatialRank =
      op.getDimensionNumbers().getInputSpatialDimensions().size();
  ConvolutionWindow window{SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(spatialRank, 1),
                           SmallVector<int64_t>(2 * spatialRank, 0)};
  if (op.getWindowStrides()) {
    window.strides = llvm::to_vector(*op.getWindowStrides());
  }
  if (op.getLhsDilation()) {
    window.lhsDilations = llvm::to_vector(*op.getLhsDilation());
  }
  if (op.getRhsDilation()) {
    window.rhsDilations = llvm::to_vector(*op.getRhsDilation());
  }
  if (op.getPadding()) {
    window.padding = llvm::to_vector(op.getPadding()->getValues<int64_t>());
  }
  return window;
}

struct ConvolutionResultShape {
  SmallVector<Value> dynamicDims;
  Value nonempty;
};

ConvolutionResultShape
buildConvolutionResultShape(OpBuilder &rewriter,
                            mlir::stablehlo::ConvolutionOp op, Value input,
                            Value filter, RankedTensorType resultType,
                            const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  auto constant = [&](int64_t value) -> Value {
    return rewriter.createOrFold<arith::ConstantIndexOp>(loc, value);
  };
  Value zero = constant(0);
  Value one = constant(1);
  // Dilation by d turns a size n into (n - 1) * d + 1, except that an empty
  // input or kernel stays empty (StableHLO convolution constraint C25).
  auto dilate = [&](Value size, int64_t dilation) -> Value {
    if (dilation == 1) {
      return size;
    }
    Value empty = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, size, zero);
    Value span = rewriter.createOrFold<arith::SubIOp>(loc, size, one);
    span = rewriter.createOrFold<arith::MulIOp>(loc, span, constant(dilation));
    Value dilated = rewriter.createOrFold<arith::AddIOp>(loc, span, one);
    return rewriter.createOrFold<arith::SelectOp>(loc, empty, zero, dilated);
  };

  ConvolutionResultShape shape;
  shape.nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
  for (int64_t d = 0; d < rank; ++d) {
    Value size;
    if (!resultType.isDynamicDim(d)) {
      size = constant(resultType.getDimSize(d));
    } else if (d == 0) {
      size = rewriter.createOrFold<tensor::DimOp>(loc, input, 0);
      size = rewriter.createOrFold<arith::DivUIOp>(
          loc, size, constant(op.getBatchGroupCount()));
    } else if (d == rank - 1) {
      size = rewriter.createOrFold<tensor::DimOp>(loc, filter, rank - 1);
    } else {
      int64_t s = d - 1;
      Value inputSize = rewriter.createOrFold<tensor::DimOp>(loc, input, d);
      Value kernelSize = rewriter.createOrFold<tensor::DimOp>(loc, filter, s);
      Value padded = dilate(inputSize, window.lhsDilations[s]);
      padded = rewriter.createOrFold<arith::AddIOp>(
          loc, padded, constant(window.padding[2 * s]));
      padded = rewriter.createOrFold<arith::AddIOp>(
          loc, padded, constant(window.padding[2 * s + 1]));
      Value dilatedKernel = dilate(kernelSize, window.rhsDilations[s]);
      Value span =
          rewriter.createOrFold<arith::SubIOp>(loc, padded, dilatedKernel);
      // For a fitting window span is nonnegative, so truncation is floor.
      // The select discards this count when the window does not fit.
      Value count = rewriter.createOrFold<arith::DivSIOp>(
          loc, span, constant(window.strides[s]));
      count = rewriter.createOrFold<arith::AddIOp>(loc, count, one);
      Value fits = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, span, zero);
      Value hasInput = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, padded, zero);
      fits = rewriter.createOrFold<arith::AndIOp>(loc, fits, hasInput);
      size = rewriter.createOrFold<arith::SelectOp>(loc, fits, count, zero);
    }
    if (resultType.isDynamicDim(d)) {
      shape.dynamicDims.push_back(size);
    }
    Value positive = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, size, zero);
    shape.nonempty =
        rewriter.createOrFold<arith::AndIOp>(loc, shape.nonempty, positive);
  }
  return shape;
}

Value padConvolutionInput(OpBuilder &rewriter, Location loc, Value input,
                          const ConvolutionWindow &window) {
  int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
  if (llvm::any_of(window.padding, [](int64_t v) { return v != 0; }) ||
      llvm::any_of(window.lhsDilations, [](int64_t v) { return v != 1; })) {
    SmallVector<int64_t> lows(rank, 0), highs(rank, 0), interiors(rank, 0);
    for (int64_t i = 0; i < rank - 2; ++i) {
      lows[i + 1] = window.padding[2 * i];
      highs[i + 1] = window.padding[2 * i + 1];
      interiors[i + 1] = window.lhsDilations[i] - 1;
    }
    auto shapeType = RankedTensorType::get({rank}, rewriter.getI64Type());
    auto shapeConstant = [&](ArrayRef<int64_t> values) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, DenseIntElementsAttr::get(shapeType, values));
    };
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> paddedShape(inputType.getShape());
    for (int64_t i = 1; i < rank - 1; ++i) {
      paddedShape[i] = ShapedType::kDynamic;
    }
    Value scalarEmpty = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{}, inputType.getElementType());
    Value padValue =
        mlir::stablehlo::fillTensorWithZeros(rewriter, loc, scalarEmpty);
    return mlir::stablehlo::DynamicPadOp::create(
        rewriter, loc,
        RankedTensorType::get(paddedShape, inputType.getElementType()), input,
        padValue, shapeConstant(lows), shapeConstant(highs),
        shapeConstant(interiors));
  }
  return input;
}

Value splitConvolutionGroupDimension(OpBuilder &rewriter, Location loc,
                                     Value value, int64_t dimension,
                                     int64_t groups) {
  auto type = cast<RankedTensorType>(value.getType());
  int64_t rank = type.getRank();
  SmallVector<int64_t> shape(type.getShape());
  if (!ShapedType::isDynamic(shape[dimension])) {
    shape[dimension] /= groups;
  }
  shape.insert(shape.begin() + dimension, groups);
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t d = 0; d < rank; ++d) {
    if (d < dimension) {
      reassociation.push_back({d});
    } else if (d == dimension) {
      reassociation.push_back({d, d + 1});
    } else {
      reassociation.push_back({d + 1});
    }
  }
  return tensor::ExpandShapeOp::create(
      rewriter, loc, RankedTensorType::get(shape, type.getElementType()), value,
      reassociation);
}

Value createGroupedConvolution(OpBuilder &rewriter,
                               mlir::stablehlo::ConvolutionOp op, Value input,
                               Value filter, Value init,
                               RankedTensorType resultType,
                               const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  // Split the grouped dimension instead of unrolling one convolution per
  // group. This keeps the IR size independent of the number of channels.
  int64_t groups = std::max(op.getFeatureGroupCount(), op.getBatchGroupCount());
  input = splitConvolutionGroupDimension(
      rewriter, loc, input, op.getFeatureGroupCount() != 1 ? rank - 1 : 0,
      groups);
  filter =
      splitConvolutionGroupDimension(rewriter, loc, filter, rank - 1, groups);
  Value groupedInit =
      splitConvolutionGroupDimension(rewriter, loc, init, rank - 1, groups);
  // Parallel loops: batch, output spatial dimensions, group, output
  // channel within the group. Reduction loops: kernel spatial, channel.
  int64_t loops = 2 * rank;
  auto dim = [&](int64_t d) { return rewriter.getAffineDimExpr(d); };
  SmallVector<AffineExpr> inputMap, filterMap, outputMap;
  if (op.getBatchGroupCount() != 1) {
    inputMap.push_back(dim(rank - 1));
  }
  inputMap.push_back(dim(0));
  for (int64_t i = 0; i < rank - 2; ++i) {
    inputMap.push_back(dim(i + 1) * window.strides[i] +
                       dim(rank + 1 + i) * window.rhsDilations[i]);
    filterMap.push_back(dim(rank + 1 + i));
  }
  if (op.getFeatureGroupCount() != 1) {
    inputMap.push_back(dim(rank - 1));
  }
  inputMap.push_back(dim(loops - 1));
  filterMap.append({dim(loops - 1), dim(rank - 1), dim(rank)});
  for (int64_t i = 0; i <= rank; ++i) {
    outputMap.push_back(dim(i));
  }
  SmallVector<AffineMap> maps;
  for (auto &exprs : {inputMap, filterMap, outputMap}) {
    maps.push_back(AffineMap::get(loops, 0, exprs, rewriter.getContext()));
  }
  SmallVector<utils::IteratorType> iterators(loops,
                                             utils::IteratorType::reduction);
  std::fill_n(iterators.begin(), rank + 1, utils::IteratorType::parallel);
  Value convolved =
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{groupedInit.getType()},
          ValueRange{input, filter}, ValueRange{groupedInit}, maps, iterators,
          [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
            ImplicitLocOpBuilder nested(nestedLoc, builder);
            linalg::Conv2DOp::regionBuilder(nested, *nested.getInsertionBlock(),
                                            {}, {});
          },
          linalg::getPrunedAttributeList(op))
          .getResult(0);
  SmallVector<ReassociationIndices> collapse;
  for (int64_t i = 0; i < rank - 1; ++i) {
    collapse.push_back({i});
  }
  collapse.push_back({rank - 1, rank});
  return tensor::CollapseShapeOp::create(rewriter, loc, resultType, convolved,
                                         collapse);
}

Value createConvolution(OpBuilder &rewriter, mlir::stablehlo::ConvolutionOp op,
                        Value input, Value filter, Value init,
                        RankedTensorType resultType,
                        const ConvolutionWindow &window) {
  Location loc = op.getLoc();
  int64_t rank = resultType.getRank();
  bool grouped = op.getFeatureGroupCount() != 1 || op.getBatchGroupCount() != 1;
  auto strideAttr = rewriter.getI64TensorAttr(window.strides);
  auto dilationAttr = rewriter.getI64TensorAttr(window.rhsDilations);
  if (grouped) {
    return createGroupedConvolution(rewriter, op, input, filter, init,
                                    resultType, window);
  }
  if (rank == 3) {
    return linalg::Conv1DNwcWcfOp::create(
               rewriter, loc, resultType, ValueRange{input, filter},
               ValueRange{init}, strideAttr, dilationAttr,
               linalg::getPrunedAttributeList(op))
        .getResult(0);
  }
  if (rank == 4) {
    return linalg::Conv2DNhwcHwcfOp::create(
               rewriter, loc, resultType, ValueRange{input, filter},
               ValueRange{init}, strideAttr, dilationAttr,
               linalg::getPrunedAttributeList(op))
        .getResult(0);
  }
  return linalg::Conv3DNdhwcDhwcfOp::create(
             rewriter, loc, resultType, ValueRange{input, filter},
             ValueRange{init}, strideAttr, dilationAttr,
             linalg::getPrunedAttributeList(op))
      .getResult(0);
}

// The upstream named convolution lowering computes dynamic batch/channel sizes
// but requires static spatial results. Compute each spatial result size as
// StableHLO convolution constraint C25 defines it: the padded, dilated input
// size minus the dilated window size, divided by the stride, plus one, or zero
// when no window fits.
// Grouped convolutions additionally split the group dimensions for Linalg.
struct DynamicConvolutionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using Base::Base;

  // Returns whether the convolution uses the batch, spatial..., feature layout
  // for input and output and the spatial..., input feature, output feature
  // layout for the kernel, which the preprocessing pass produces.
  static bool hasCanonicalLayout(mlir::stablehlo::ConvDimensionNumbersAttr dims,
                                 int64_t rank) {
    if (dims.getInputBatchDimension() != 0 ||
        dims.getOutputBatchDimension() != 0 ||
        dims.getInputFeatureDimension() != rank - 1 ||
        dims.getOutputFeatureDimension() != rank - 1 ||
        dims.getKernelInputFeatureDimension() != rank - 2 ||
        dims.getKernelOutputFeatureDimension() != rank - 1) {
      return false;
    }
    auto spatialDims = llvm::seq<int64_t>(1, rank - 1);
    auto kernelSpatialDims = llvm::seq<int64_t>(0, rank - 2);
    return llvm::equal(dims.getInputSpatialDimensions(), spatialDims) &&
           llvm::equal(dims.getOutputSpatialDimensions(), spatialDims) &&
           llvm::equal(dims.getKernelSpatialDimensions(), kernelSpatialDims);
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked result");
    }
    int64_t rank = resultType.getRank();
    if (rank < 3 || rank > 5) {
      return rewriter.notifyMatchFailure(op,
                                         "expected 1D, 2D or 3D convolution");
    }
    auto dims = op.getDimensionNumbers();
    bool grouped =
        op.getFeatureGroupCount() != 1 || op.getBatchGroupCount() != 1;
    if ((!grouped || resultType.hasStaticShape()) &&
        !llvm::any_of(dims.getOutputSpatialDimensions(),
                      [&](int64_t i) { return resultType.isDynamicDim(i); })) {
      return rewriter.notifyMatchFailure(
          op, "upstream handles static spatial results");
    }
    if (auto reversal = op.getWindowReversal();
        reversal && llvm::is_contained(*reversal, true)) {
      return rewriter.notifyMatchFailure(
          op, "expected preprocessed window reversal");
    }
    if (!hasCanonicalLayout(dims, rank)) {
      return rewriter.notifyMatchFailure(
          op, "expected canonical dimension numbers");
    }

    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value filter = adaptor.getRhs();
    auto window = getConvolutionWindow(op);
    auto shape = buildConvolutionResultShape(rewriter, op, input, filter,
                                             resultType, window);
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resultType, shape.dynamicDims);
    Value init = mlir::stablehlo::fillTensorWithZeros(rewriter, loc, empty);

    // Avoid materializing a negatively-sized crop (or running a convolution)
    // when no window fits. The result shape above is still the C25 size.
    auto ifOp =
        scf::IfOp::create(rewriter, loc, TypeRange{resultType}, shape.nonempty,
                          /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      input = padConvolutionInput(rewriter, loc, input, window);
      Value result = createConvolution(rewriter, op, input, filter, init,
                                       resultType, window);
      scf::YieldOp::create(rewriter, loc, result);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, init);
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTieShapeOp>(
        op, resultType, ifOp.getResult(0), shape.dynamicDims);
    return success();
  }
};

struct WideIndexGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::GatherOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto startIndicesType = cast<ShapedType>(op.getStartIndices().getType());
    auto indexType = dyn_cast<IntegerType>(startIndicesType.getElementType());
    if (!indexType ||
        (!indexType.isUnsignedInteger() && indexType.getWidth() < 64)) {
      return rewriter.notifyMatchFailure(
          op, "start indices fit signed 32-bit conversion path");
    }

    auto sliceSizesType =
        RankedTensorType::get({static_cast<int64_t>(op.getSliceSizes().size())},
                              rewriter.getI64Type());
    auto sliceSizesAttr =
        DenseIntElementsAttr::get(sliceSizesType, op.getSliceSizes());
    Value sliceSizes = mlir::stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), sliceSizesAttr);
    // Preserve source types here so DynamicGatherOpConversion can still
    // inspect signedness before the type converter removes it.
    auto dynamicGather = mlir::stablehlo::DynamicGatherOp::create(
        rewriter, op.getLoc(), op.getType(), op.getOperand(),
        op.getStartIndices(), sliceSizes, op.getDimensionNumbersAttr(),
        op.getIndicesAreSortedAttr());
    rewriter.replaceOp(op, dynamicGather.getResult());
    return success();
  }
};

// Builds the element computation of a dynamic gather. Each result element
// reads the operand at the clamped start index, plus the batching index, plus
// the offset within the slice.
struct DynamicGatherBodyBuilder {
  Value operand;
  Value startIndices;
  mlir::stablehlo::GatherDimensionNumbersAttr dims;
  // The operand dimensions that offset dimensions index, in order.
  SmallVector<unsigned> remappedOffsetDims;
  // Index constants 0, 1, ... used as fixed indices.
  SmallVector<Value> constants;
  // Per operand dimension, the largest start index keeping the slice inside.
  SmallVector<Value> largestStartIndices;
  Type originalStartIndexElementType;
  int64_t resultRank;

  void build(OpBuilder &builder, Location loc) const {
    SmallVector<Value> resultIndices;
    SmallVector<Value> gatherIndex;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      Value index = linalg::IndexOp::create(builder, loc, dim);
      resultIndices.push_back(index);
      if (!llvm::is_contained(dims.getOffsetDims(), dim)) {
        gatherIndex.push_back(index);
      }
    }
    SmallVector<Value> startIndexByOperand =
        extractStartIndices(builder, loc, gatherIndex);
    SmallVector<Value> indexFromBatching = getBatchingIndices(gatherIndex);
    SmallVector<Value> combinedIndex;
    for (auto [operandDim, start] : llvm::enumerate(startIndexByOperand)) {
      Value startIndex = constants[0];
      if (start) {
        startIndex =
            clampGatherIndex(builder, loc, start, originalStartIndexElementType,
                             largestStartIndices[operandDim]);
      }
      Value offsetIndex = getOffsetIndex(operandDim, resultIndices);
      combinedIndex.push_back(builder.createOrFold<arith::AddIOp>(
          loc, builder.getIndexType(),
          builder.createOrFold<arith::AddIOp>(loc, builder.getIndexType(),
                                              startIndex,
                                              indexFromBatching[operandDim]),
          offsetIndex));
    }
    Value element =
        tensor::ExtractOp::create(builder, loc, operand, combinedIndex);
    linalg::YieldOp::create(builder, loc, element);
  }

  // Reads each start index component and places it at the operand dimension
  // the start index map assigns; other dimensions stay null.
  SmallVector<Value> extractStartIndices(OpBuilder &builder, Location loc,
                                         ArrayRef<Value> gatherIndex) const {
    int64_t operandRank = largestStartIndices.size();
    int64_t indicesRank =
        cast<RankedTensorType>(startIndices.getType()).getRank();
    int64_t indexVectorDim = dims.getIndexVectorDim();
    SmallVector<Value> startIndexByOperand(operandRank);
    for (auto [i, operandDim] : llvm::enumerate(dims.getStartIndexMap())) {
      SmallVector<Value> indices(gatherIndex);
      if (indexVectorDim != indicesRank) {
        indices.insert(indices.begin() + indexVectorDim, constants[i]);
      }
      startIndexByOperand[operandDim] =
          tensor::ExtractOp::create(builder, loc, startIndices, indices);
    }
    return startIndexByOperand;
  }

  // Returns, per operand dimension, the batch index it takes from the start
  // indices, or zero.
  SmallVector<Value> getBatchingIndices(ArrayRef<Value> gatherIndex) const {
    int64_t indexVectorDim = dims.getIndexVectorDim();
    SmallVector<Value> indexFromBatching(largestStartIndices.size(),
                                         constants[0]);
    for (auto [operandDim, indicesDim] :
         llvm::zip_equal(dims.getOperandBatchingDims(),
                         dims.getStartIndicesBatchingDims())) {
      indexFromBatching[operandDim] =
          gatherIndex[indicesDim - (indicesDim < indexVectorDim ? 0 : 1)];
    }
    return indexFromBatching;
  }

  // Returns the offset within the slice along `operandDim`, or zero for a
  // collapsed or batching dimension.
  Value getOffsetIndex(int64_t operandDim,
                       ArrayRef<Value> resultIndices) const {
    auto offset = llvm::find(remappedOffsetDims, operandDim);
    if (offset == remappedOffsetDims.end()) {
      return constants[0];
    }
    return resultIndices[dims.getOffsetDims()[offset -
                                              remappedOffsetDims.begin()]];
  }
};

struct DynamicGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicGatherOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto startIndicesType =
        dyn_cast<RankedTensorType>(adaptor.getStartIndices().getType());
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!operandType || !startIndicesType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    }
    auto dims = op.getDimensionNumbers();
    SmallVector<unsigned> remappedOffsetDims =
        getOffsetOperandDims(dims, operandType.getRank());
    if (remappedOffsetDims.size() != dims.getOffsetDims().size()) {
      return rewriter.notifyMatchFailure(op, "inconsistent offset dimensions");
    }

    int64_t operandRank = operandType.getRank();
    int64_t resultRank = resultType.getRank();
    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      constants.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));
    }
    Value empty = mlir::stablehlo::getEmptyTensorFor(rewriter, loc, resultType,
                                                     op, adaptor.getOperands());
    DynamicGatherBodyBuilder body{
        operand,
        adaptor.getStartIndices(),
        dims,
        remappedOffsetDims,
        constants,
        getLargestStartIndices(rewriter, loc, operand, empty,
                               remappedOffsetDims, dims.getOffsetDims(),
                               /*one=*/constants[1]),
        cast<ShapedType>(op.getStartIndices().getType()).getElementType(),
        resultRank};
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/empty,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        mlir::stablehlo::getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
          body.build(builder, nestedLoc);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }

private:
  // Returns the operand dimensions that are neither collapsed nor batching;
  // the result's offset dimensions index them in order.
  static SmallVector<unsigned>
  getOffsetOperandDims(mlir::stablehlo::GatherDimensionNumbersAttr dims,
                       int64_t operandRank) {
    SmallVector<unsigned> offsetOperandDims;
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      if (!llvm::is_contained(dims.getCollapsedSliceDims(), dim) &&
          !llvm::is_contained(dims.getOperandBatchingDims(), dim)) {
        offsetOperandDims.push_back(static_cast<unsigned>(dim));
      }
    }
    return offsetOperandDims;
  }

  // Returns operand extent minus slice extent for each operand dimension. A
  // slice is one element wide except along offset dimensions, where it takes
  // the result extent.
  static SmallVector<Value>
  getLargestStartIndices(OpBuilder &builder, Location loc, Value operand,
                         Value result, ArrayRef<unsigned> remappedOffsetDims,
                         ArrayRef<int64_t> offsetDims, Value one) {
    int64_t operandRank = cast<RankedTensorType>(operand.getType()).getRank();
    SmallVector<Value> operandDims(operandRank);
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      operandDims[dim] = builder.createOrFold<tensor::DimOp>(loc, operand, dim);
    }
    SmallVector<Value> sliceSizes(operandRank, one);
    for (auto [operandDim, resultDim] :
         llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      sliceSizes[operandDim] =
          builder.createOrFold<tensor::DimOp>(loc, result, resultDim);
    }
    SmallVector<Value> largestStartIndices(operandRank);
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      largestStartIndices[dim] = builder.createOrFold<arith::SubIOp>(
          loc, operandDims[dim], sliceSizes[dim]);
    }
    return largestStartIndices;
  }
};

struct DynamicBroadcastInDimOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using Base::Base;

  // Shape computation legalization assembles dynamic dimension vectors from
  // scalar values with tensor.from_elements before this conversion runs.
  static Value getShapeElement(Value shape, int64_t index) {
    auto fromElements = shape.getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements || index < 0 ||
        index >= static_cast<int64_t>(fromElements.getNumOperands())) {
      return {};
    }
    return fromElements.getOperand(index);
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value operand = adaptor.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!operandType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    }

    SmallVector<std::optional<bool>> expanding =
        getDeclaredExpansion(op, operandType);
    bool inferredNonexpanding = inferNonexpandingDims(op, operand, expanding);
    if (llvm::all_of(expanding, [](auto value) { return value.has_value(); })) {
      if (!inferredNonexpanding) {
        return rewriter.notifyMatchFailure(op, "expansion behavior is known");
      }
      // Record the proof so the upstream pattern emits an affine-map generic.
      replaceWithKnownExpansion(op, adaptor, expanding, rewriter);
      return success();
    }
    rewriter.replaceOp(op, createRuntimeExpansionBroadcast(
                               op, adaptor, resultType, expanding, rewriter));
    return success();
  }

private:
  // Returns, per operand dimension, whether it is known to expand: static
  // sizes and the op's declared expanding and nonexpanding dimensions.
  static SmallVector<std::optional<bool>>
  getDeclaredExpansion(mlir::stablehlo::DynamicBroadcastInDimOp op,
                       RankedTensorType operandType) {
    SmallVector<std::optional<bool>> expanding(operandType.getRank());
    for (auto [i, size] : llvm::enumerate(operandType.getShape())) {
      if (ShapedType::isStatic(size)) {
        expanding[i] = size == 1;
      }
    }
    for (int64_t i :
         op.getKnownExpandingDimensions().value_or(ArrayRef<int64_t>{})) {
      expanding[i] = true;
    }
    for (int64_t i :
         op.getKnownNonexpandingDimensions().value_or(ArrayRef<int64_t>{})) {
      expanding[i] = false;
    }
    return expanding;
  }

  // Marks unknown dimensions nonexpanding where the operand size provably
  // equals the requested output size. Returns whether any was inferred.
  static bool
  inferNonexpandingDims(mlir::stablehlo::DynamicBroadcastInDimOp op,
                        Value operand,
                        MutableArrayRef<std::optional<bool>> expanding) {
    // One factorizer bounds every proof for this broadcast.
    SizeFactorizer factorizer;
    bool inferredNonexpanding = false;
    for (auto [operandDim, resultDim] :
         llvm::enumerate(op.getBroadcastDimensions())) {
      if (expanding[operandDim].has_value()) {
        continue;
      }
      Value outputElement =
          getShapeElement(op.getOutputDimensions(), resultDim);
      if (!outputElement) {
        continue;
      }
      std::optional<SizeProduct> sourceSize =
          factorizer.factorDim(operand, operandDim);
      std::optional<SizeProduct> outputSize =
          factorizer.factorSize(outputElement);
      // A wrapped operand size may be one where the output size is not.
      if (sourceSize && outputSize && !sourceSize->mayWrap &&
          equalSizes(*sourceSize, *outputSize)) {
        expanding[operandDim] = false;
        inferredNonexpanding = true;
      }
    }
    return inferredNonexpanding;
  }

  // Replaces `op` with a broadcast that declares every dynamic dimension's
  // expansion, which the upstream pattern lowers with affine maps.
  static void
  replaceWithKnownExpansion(mlir::stablehlo::DynamicBroadcastInDimOp op,
                            OpAdaptor adaptor,
                            ArrayRef<std::optional<bool>> expanding,
                            ConversionPatternRewriter &rewriter) {
    auto operandType = cast<RankedTensorType>(adaptor.getOperand().getType());
    SmallVector<int64_t> knownExpanding;
    SmallVector<int64_t> knownNonexpanding;
    for (auto [i, value] : llvm::enumerate(expanding)) {
      if (operandType.isDynamicDim(i)) {
        (*value ? knownExpanding : knownNonexpanding).push_back(i);
      }
    }
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
        op, op.getType(), adaptor.getOperand(), adaptor.getOutputDimensions(),
        rewriter.getDenseI64ArrayAttr(op.getBroadcastDimensions()),
        rewriter.getDenseI64ArrayAttr(knownExpanding),
        rewriter.getDenseI64ArrayAttr(knownNonexpanding));
  }

  // Builds a gather-style broadcast that selects index zero for operand
  // dimensions whose expansion is only decided at runtime by a size of one.
  static Value createRuntimeExpansionBroadcast(
      mlir::stablehlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      RankedTensorType resultType, ArrayRef<std::optional<bool>> expanding,
      ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    Value empty = mlir::stablehlo::getEmptyTensorFor(rewriter, loc, resultType,
                                                     op, adaptor.getOperands());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> isExpanding(expanding.size());
    for (auto [i, known] : llvm::enumerate(expanding)) {
      if (known.has_value()) {
        continue;
      }
      Value dim = rewriter.createOrFold<tensor::DimOp>(loc, operand, i);
      isExpanding[i] = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, dim, one);
    }

    int64_t resultRank = resultType.getRank();
    auto generic = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/empty,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        mlir::stablehlo::getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
          SmallVector<Value> operandIndex;
          for (auto [operandDim, resultDim] :
               llvm::enumerate(op.getBroadcastDimensions())) {
            Value resultIndex =
                linalg::IndexOp::create(builder, nestedLoc, resultDim);
            if (expanding[operandDim].has_value()) {
              operandIndex.push_back(
                  expanding[operandDim].value() ? zero : resultIndex);
              continue;
            }
            operandIndex.push_back(arith::SelectOp::create(
                builder, nestedLoc, isExpanding[operandDim], zero,
                resultIndex));
          }
          Value element = tensor::ExtractOp::create(builder, nestedLoc, operand,
                                                    operandIndex);
          linalg::YieldOp::create(builder, nestedLoc, element);
        },
        linalg::getPrunedAttributeList(op));
    return generic.getResult(0);
  }
};

// Returns the reducer for one result when it combines that result's init and
// input values directly, matching the reducer shape expected by linalg.
Operation *getReduceWindowReductionOp(mlir::stablehlo::ReduceWindowOp op,
                                      int64_t resultIndex) {
  auto returnOp =
      cast<mlir::stablehlo::ReturnOp>(op.getBody().front().getTerminator());
  Operation *reduceOp = returnOp.getResults()[resultIndex].getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2) {
    return nullptr;
  }
  auto lhs = dyn_cast<BlockArgument>(reduceOp->getOperand(0));
  auto rhs = dyn_cast<BlockArgument>(reduceOp->getOperand(1));
  if (!lhs || !rhs) {
    return nullptr;
  }
  int64_t accumulatorArg = resultIndex;
  int64_t inputArg = resultIndex + op.getInputs().size();
  if (lhs.getArgNumber() == accumulatorArg && rhs.getArgNumber() == inputArg) {
    return reduceOp;
  }
  if (lhs.getArgNumber() == inputArg && rhs.getArgNumber() == accumulatorArg &&
      reduceOp->hasTrait<OpTrait::IsCommutative>()) {
    return reduceOp;
  }
  return nullptr;
}

// Returns whether optional per-dimension values are absent or all one.
bool isAbsentOrAllOnes(std::optional<ArrayRef<int64_t>> values) {
  return !values ||
         llvm::all_of(*values, [](int64_t value) { return value == 1; });
}

// Returns whether the window neither spans nor strides over the batch and
// channel dimensions, the first and last.
bool keepsBatchAndChannel(mlir::stablehlo::ReduceWindowOp op) {
  ArrayRef<int64_t> window = op.getWindowDimensions();
  int64_t lastDim = window.size() - 1;
  if (window[0] != 1 || window[lastDim] != 1) {
    return false;
  }
  std::optional<ArrayRef<int64_t>> strides = op.getWindowStrides();
  return !strides || ((*strides)[0] == 1 && (*strides)[lastDim] == 1);
}

// Returns whether input `index` has the f32 static spatial shape and min, max
// or add reducer that the upstream pooling lowering handles. Upstream reads
// batch and channel through tensor.dim; only spatial dimensions enter its
// output-shape formula.
bool isUpstreamPoolingInput(mlir::stablehlo::ReduceWindowOp op, int64_t index) {
  auto inputType = cast<ShapedType>(op.getInputs()[index].getType());
  if (!inputType.getElementType().isF32()) {
    return false;
  }
  ArrayRef<int64_t> window = op.getWindowDimensions();
  for (int64_t dim = 1, lastDim = inputType.getRank() - 1; dim < lastDim;
       ++dim) {
    if (inputType.isDynamicDim(dim) ||
        inputType.getDimSize(dim) < window[dim]) {
      return false;
    }
  }
  Operation *reducer = getReduceWindowReductionOp(op, index);
  return reducer && isa<mlir::stablehlo::MinOp, mlir::stablehlo::MaxOp,
                        mlir::stablehlo::AddOp>(reducer);
}

// Returns whether the upstream pooling lowering, ReduceWindowOpConversion in
// StableHLO's StablehloToLinalgReduce.cpp, handles `op`. It derives the
// result shape from the window formula of the StableHLO reduce_window
// specification (constraint C15), which this checks it can evaluate: 2-D or
// 3-D pooling without padding or dilation over static spatial dimensions.
bool matchesUpstreamPooling(mlir::stablehlo::ReduceWindowOp op) {
  int64_t rank = cast<ShapedType>(op.getResultTypes()[0]).getRank();
  bool hasPoolingRank = rank == 4 || rank == 5;
  bool hasNoPadding =
      !op.getPadding() || mlir::stablehlo::isSplatValue(*op.getPadding(), 0);
  if (!hasPoolingRank || !hasNoPadding ||
      !isAbsentOrAllOnes(op.getBaseDilations()) ||
      !isAbsentOrAllOnes(op.getWindowDilations()) ||
      !keepsBatchAndChannel(op)) {
    return false;
  }
  return llvm::all_of(
      llvm::seq<int64_t>(op.getInputs().size()),
      [&](int64_t index) { return isUpstreamPoolingInput(op, index); });
}

// The window attributes of a reduce_window with their defaults filled in.
struct ReduceWindowGeometry {
  SmallVector<int64_t> windowDimensions;
  // Low and high padding, interleaved per dimension.
  SmallVector<int64_t> padding;
  SmallVector<int64_t> baseDilations;
  SmallVector<int64_t> windowStrides;
  SmallVector<int64_t> windowDilations;

  explicit ReduceWindowGeometry(mlir::stablehlo::ReduceWindowOp op)
      : windowDimensions(op.getWindowDimensions()) {
    int64_t rank = windowDimensions.size();
    auto orOnes = [&](std::optional<ArrayRef<int64_t>> values) {
      return values ? SmallVector<int64_t>(*values)
                    : SmallVector<int64_t>(rank, 1);
    };
    padding = op.getPadding()
                  ? llvm::to_vector(op.getPadding()->getValues<int64_t>())
                  : SmallVector<int64_t>(2 * rank, 0);
    baseDilations = orOnes(op.getBaseDilations());
    windowStrides = orOnes(op.getWindowStrides());
    windowDilations = orOnes(op.getWindowDilations());
  }

  int64_t getRank() const { return windowDimensions.size(); }

  bool needsPadding() const {
    return llvm::any_of(padding, [](int64_t value) { return value != 0; }) ||
           llvm::any_of(baseDilations,
                        [](int64_t value) { return value != 1; });
  }

  int64_t getDilatedWindowSize(int64_t dim) const {
    return (windowDimensions[dim] - 1) * windowDilations[dim] + 1;
  }

  // Returns the source, window and result maps over result dimensions followed
  // by the non-unit window dimensions, whose sizes fill `windowShape`.
  SmallVector<AffineMap>
  getIndexingMaps(MLIRContext *context,
                  SmallVectorImpl<int64_t> &windowShape) const {
    int64_t rank = getRank();
    SmallVector<AffineExpr> sourceExprs, windowExprs, resultExprs;
    for (int64_t dim = 0; dim < rank; ++dim) {
      AffineExpr sourceExpr = getAffineDimExpr(dim, context);
      if (windowStrides[dim] != 1) {
        sourceExpr = sourceExpr * windowStrides[dim];
      }
      if (windowDimensions[dim] != 1) {
        windowShape.push_back(windowDimensions[dim]);
        AffineExpr windowExpr =
            getAffineDimExpr(rank + windowExprs.size(), context);
        windowExprs.push_back(windowExpr);
        if (windowDilations[dim] != 1) {
          windowExpr = windowExpr * windowDilations[dim];
        }
        sourceExpr = sourceExpr + windowExpr;
      }
      sourceExprs.push_back(sourceExpr);
      resultExprs.push_back(getAffineDimExpr(dim, context));
    }
    if (rank == 0) {
      return SmallVector<AffineMap>(3, AffineMap::get(context));
    }
    return AffineMap::inferFromExprList({sourceExprs, windowExprs, resultExprs},
                                        context);
  }
};

struct DynamicReduceWindowOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (matchesUpstreamPooling(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "upstream pooling lowering applies");
    }
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }
    if (llvm::all_of(resultTypes, [](Type type) {
          return cast<ShapedType>(type).hasStaticShape();
        })) {
      return rewriter.notifyMatchFailure(op, "static result shape");
    }
    if (op.getInputs().size() != op.getResults().size()) {
      return rewriter.notifyMatchFailure(op, "expected one result per input");
    }

    Location loc = op.getLoc();
    ReduceWindowGeometry geometry(op);
    SmallVector<int64_t> windowShape;
    SmallVector<AffineMap> maps =
        geometry.getIndexingMaps(op->getContext(), windowShape);
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.append(op.getInputs().size(), maps[0]);
    indexingMaps.push_back(maps[1]);
    indexingMaps.append(op.getResults().size(), maps[2]);

    SmallVector<Value> inputs = llvm::to_vector(adaptor.getInputs());
    SmallVector<Value> initValues = llvm::to_vector(adaptor.getInitValues());
    if (geometry.needsPadding()) {
      padInputs(rewriter, loc, geometry, inputs, initValues);
    }
    SmallVector<Value> seeds;
    for (auto [input, initValue, resultType] :
         llvm::zip(inputs, initValues, resultTypes)) {
      seeds.push_back(createSeed(rewriter, loc, geometry, input, initValue,
                                 cast<RankedTensorType>(resultType)));
    }
    Type windowElementType =
        cast<ShapedType>(inputs.front().getType()).getElementType();
    inputs.push_back(
        tensor::EmptyOp::create(rewriter, loc, windowShape, windowElementType));
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultTypes, inputs, seeds, indexingMaps,
        mlir::stablehlo::getParallelAndReductionIterators(
            geometry.getRank() + windowShape.size(), windowShape.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));
    moveReducerInto(op, generic, ArrayRef<Value>(inputs).drop_back(),
                    windowElementType, resultTypes, rewriter);
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }

private:
  // Pads and base-dilates each input with its init value, as reduce_window
  // defines the padded input.
  static void padInputs(ConversionPatternRewriter &rewriter, Location loc,
                        const ReduceWindowGeometry &geometry,
                        MutableArrayRef<Value> inputs,
                        ArrayRef<Value> initValues) {
    int64_t rank = geometry.getRank();
    SmallVector<int64_t> lows(rank), highs(rank), interiors(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      lows[dim] = geometry.padding[2 * dim];
      highs[dim] = geometry.padding[2 * dim + 1];
      interiors[dim] = geometry.baseDilations[dim] - 1;
    }
    for (auto [index, initValue] : llvm::enumerate(initValues)) {
      inputs[index] = mlir::stablehlo::PadOp::create(
          rewriter, loc, inputs[index], initValue, lows, highs, interiors);
    }
  }

  // Returns a result tensor filled with `initValue`. Each dynamic extent is
  // the number of windows that fit the padded input, or zero when none does.
  static Value createSeed(ConversionPatternRewriter &rewriter, Location loc,
                          const ReduceWindowGeometry &geometry, Value input,
                          Value initValue, RankedTensorType resultType) {
    auto constantIndex = [&](int64_t value) {
      return arith::ConstantIndexOp::create(rewriter, loc, value).getResult();
    };
    SmallVector<Value> dynamicDims;
    for (int64_t dim = 0; dim < geometry.getRank(); ++dim) {
      if (!resultType.isDynamicDim(dim)) {
        continue;
      }
      Value paddedExtent = tensor::DimOp::create(rewriter, loc, input, dim);
      Value span = arith::SubIOp::create(
          rewriter, loc, paddedExtent,
          constantIndex(geometry.getDilatedWindowSize(dim)));
      Value nonnegative = rewriter.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, span, constantIndex(0));
      Value extent = arith::DivSIOp::create(
          rewriter, loc, span, constantIndex(geometry.windowStrides[dim]));
      extent = arith::AddIOp::create(rewriter, loc, extent, constantIndex(1));
      dynamicDims.push_back(rewriter.createOrFold<arith::SelectOp>(
          loc, nonnegative, extent, constantIndex(0)));
    }
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resultType, dynamicDims);
    Value scalarInit = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
    return linalg::FillOp::create(rewriter, loc, scalarInit, empty).result();
  }

  // Moves the reducer into `generic`, reordering its arguments: StableHLO
  // orders accumulator arguments first and input arguments second;
  // linalg.generic orders input scalars before output accumulator scalars.
  void moveReducerInto(mlir::stablehlo::ReduceWindowOp op,
                       linalg::GenericOp generic, ArrayRef<Value> inputs,
                       Type windowElementType, ArrayRef<Type> resultTypes,
                       ConversionPatternRewriter &rewriter) const {
    Region &region = generic.getRegion();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());
    TypeConverter::SignatureConversion signatureConversion(
        op.getInputs().size() + op.getResults().size());
    for (auto [index, type] : llvm::enumerate(resultTypes)) {
      signatureConversion.addInputs(op.getInputs().size() + index,
                                    cast<ShapedType>(type).getElementType());
    }
    signatureConversion.addInputs(windowElementType);
    for (auto [index, input] : llvm::enumerate(inputs)) {
      signatureConversion.addInputs(
          index, cast<ShapedType>(input.getType()).getElementType());
    }
    rewriter.applySignatureConversion(&region.front(), signatureConversion,
                                      getTypeConverter());
  }
};

// The upstream slice lowerings cast start indices to index before clamping
// them as signed values, which misreads unsigned and wider-than-index ones.
bool needsOwnStartClamp(TypeRange startTypes) {
  return llvm::any_of(startTypes, [&](Type type) {
    auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
    return intType &&
           (intType.isUnsigned() || intType.getWidth() > kNarrowestIndexWidth);
  });
}

// Clamps each scalar start index to [0, operand extent - slice extent].
SmallVector<OpFoldResult> clampStartIndices(OpBuilder &builder, Location loc,
                                            ValueRange starts,
                                            TypeRange originalTypes,
                                            Value operand,
                                            ArrayRef<OpFoldResult> sliceSizes) {
  SmallVector<OpFoldResult> offsets;
  for (auto [dim, start, originalType] :
       llvm::enumerate(starts, originalTypes)) {
    Value raw = tensor::ExtractOp::create(builder, loc, start, ValueRange{});
    Value operandSize = getValueOrCreateConstantIndexOp(
        builder, loc, tensor::getMixedSize(builder, loc, operand, dim));
    Value sliceSize =
        getValueOrCreateConstantIndexOp(builder, loc, sliceSizes[dim]);
    Value upperBound =
        builder.createOrFold<arith::SubIOp>(loc, operandSize, sliceSize);
    offsets.push_back(clampGatherIndex(
        builder, loc, raw, getElementTypeOrSelf(originalType), upperBound));
  }
  return offsets;
}

struct DynamicSliceOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsOwnStartClamp(op.getStartIndices().getTypes())) {
      return rewriter.notifyMatchFailure(op, "upstream clamps these indices");
    }
    auto resultType = dyn_cast_if_present<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked result");
    }
    Location loc = op.getLoc();
    SmallVector<OpFoldResult> sizes =
        getAsIndexOpFoldResult(rewriter.getContext(), op.getSliceSizes());
    SmallVector<OpFoldResult> offsets = clampStartIndices(
        rewriter, loc, adaptor.getStartIndices(),
        op.getStartIndices().getTypes(), adaptor.getOperand(), sizes);
    SmallVector<OpFoldResult> strides(sizes.size(), rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, adaptor.getOperand(), offsets, sizes, strides);
    return success();
  }
};

// Also takes runtime operand and update extents, which the upstream lowering
// rejects; insert_slice keeps the update in place for bufferization.
struct DynamicUpdateSliceOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operandType = cast<RankedTensorType>(adaptor.getOperand().getType());
    auto updateType = cast<RankedTensorType>(adaptor.getUpdate().getType());
    if (operandType.hasStaticShape() && updateType.hasStaticShape() &&
        !needsOwnStartClamp(op.getStartIndices().getTypes())) {
      return rewriter.notifyMatchFailure(op, "upstream lowers this form");
    }
    auto resultType = dyn_cast_if_present<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked result");
    }
    Location loc = op.getLoc();
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, adaptor.getUpdate());
    SmallVector<OpFoldResult> offsets = clampStartIndices(
        rewriter, loc, adaptor.getStartIndices(),
        op.getStartIndices().getTypes(), adaptor.getOperand(), sizes);
    SmallVector<OpFoldResult> strides(sizes.size(), rewriter.getIndexAttr(1));
    Value inserted = tensor::InsertSliceOp::create(
        rewriter, loc, adaptor.getUpdate(), adaptor.getOperand(), offsets,
        sizes, strides);
    // The result type may be more static than the operand.
    if (inserted.getType() != resultType) {
      inserted = tensor::CastOp::create(rewriter, loc, resultType, inserted);
    }
    rewriter.replaceOp(op, inserted);
    return success();
  }
};

// The upstream pointwise lowering reifies a dynamic result through an
// interface reduce_precision lacks; the result has its operand's shape.
struct DynamicReducePrecisionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ReducePrecisionOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReducePrecisionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast_if_present<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!resultType || resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a dynamic ranked result");
    }
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    Value empty = tensor::EmptyOp::create(
        rewriter, loc, tensor::getMixedSizes(rewriter, loc, operand),
        resultType.getElementType());
    int64_t rank = resultType.getRank();
    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, empty.getType(), operand, empty, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = mlir::stablehlo::StablehloOpToStdScalarOp::mapOp(
              op, resultType.getElementType(), args.take_front(), &b);
          linalg::YieldOp::create(b, nestedLoc, result);
        });
    // The verifier only requires compatible shapes, so the declared result
    // may be refined differently from the operand.
    Value result = generic.getResult(0);
    if (result.getType() != resultType) {
      result = tensor::CastOp::create(rewriter, loc, resultType, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Prefer these lowerings over the upstream patterns for the same ops.
  patterns
      ->add<DynamicReshapeOpConversion, PadOpConversion, DynamicPadOpConversion,
            DynamicConvolutionOpConversion, WideIndexGatherOpConversion,
            DynamicGatherOpConversion, DynamicBroadcastInDimOpConversion,
            DynamicReduceWindowOpConversion, DynamicReducePrecisionOpConversion,
            DynamicSliceOpConversion, DynamicUpdateSliceOpConversion>(
          typeConverter, context, PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
