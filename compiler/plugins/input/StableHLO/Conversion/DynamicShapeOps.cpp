// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include <optional>

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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

Value clampGatherIndex(OpBuilder &builder, Location loc, Value rawIndex,
                       Type originalElementType, Value upperBound,
                       unsigned indexWidth) {
  auto rawType = cast<IntegerType>(rawIndex.getType());
  bool isUnsigned = originalElementType.isUnsignedInteger();
  unsigned rawWidth = rawType.getWidth();

  if (rawWidth >= indexWidth) {
    Value zero =
        arith::ConstantOp::create(builder, loc, IntegerAttr::get(rawType, 0));
    Value rawUpperBound;
    if (isUnsigned) {
      rawUpperBound =
          arith::IndexCastUIOp::create(builder, loc, rawType, upperBound);
    } else {
      rawUpperBound =
          arith::IndexCastOp::create(builder, loc, rawType, upperBound);
    }
    Value nonnegative;
    if (isUnsigned) {
      nonnegative = arith::MaxUIOp::create(builder, loc, rawIndex, zero);
    } else {
      nonnegative = arith::MaxSIOp::create(builder, loc, rawIndex, zero);
    }
    Value clamped;
    if (isUnsigned) {
      clamped =
          arith::MinUIOp::create(builder, loc, nonnegative, rawUpperBound);
      return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                          clamped);
    }
    clamped = arith::MinSIOp::create(builder, loc, nonnegative, rawUpperBound);
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      clamped);
  }

  Value index;
  if (isUnsigned) {
    index = arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                         rawIndex);
  } else {
    index = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                       rawIndex);
  }
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value nonnegative = arith::MaxSIOp::create(builder, loc, index, zero);
  Value clamped = arith::MinSIOp::create(builder, loc, nonnegative, upperBound);
  return clamped;
}

// Bounds the walk from a size to the ops that define it, the ops one proof
// visits, and the symbols one product holds.
constexpr int kMaxFactorDepth = 64;
constexpr int kMaxFactorVisits = 256;
constexpr size_t kMaxFactorSymbols = 64;

// The work left to one shape proof.
struct FactorBudget {
  int visits = 0;
};

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

// A size as a constant times a product of opaque sizes, plus an offset.
// `mayWrap` marks a product or sum taken in fewer than 64 bits, read as exact
// without proof. A size with an offset names itself as `whole`, the opaque
// symbol it becomes in a product. A constant size has no offset.
struct SizeProduct {
  int64_t constant = 1;
  SmallVector<SizeSymbol> symbols;
  bool mayWrap = false;
  int64_t offset = 0;
  std::optional<SizeSymbol> whole;

  bool isOne() const { return constant == 1 && symbols.empty() && offset == 0; }
  bool isNonzero() const {
    if (symbols.empty()) {
      return constant != 0;
    }
    bool positive = llvm::all_of(
        symbols, [](const SizeSymbol &symbol) { return symbol.positive; });
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
  if (!constant ||
      lhs.symbols.size() + rhs.symbols.size() > kMaxFactorSymbols) {
    return std::nullopt;
  }
  SizeProduct product{*constant, lhs.symbols, lhs.mayWrap || rhs.mayWrap};
  llvm::append_range(product.symbols, rhs.symbols);
  return product;
}

// Returns whether every symbol of `part` is in `whole`, counting repeats.
bool containsSymbols(ArrayRef<SizeSymbol> part, ArrayRef<SizeSymbol> whole) {
  SmallVector<SizeSymbol> remaining(whole);
  for (const auto &symbol : part) {
    auto it = llvm::find(remaining, symbol);
    if (it == remaining.end()) {
      return false;
    }
    remaining.erase(it);
  }
  return true;
}

bool equalSizes(const SizeProduct &lhs, const SizeProduct &rhs) {
  return lhs.constant == rhs.constant && lhs.offset == rhs.offset &&
         lhs.symbols.size() == rhs.symbols.size() &&
         containsSymbols(lhs.symbols, rhs.symbols);
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
    const SizeSymbol &symbol = size.symbols.front();
    std::optional<int64_t> largest;
    if (size.symbols.size() == 1 && symbol.positive && symbol.umax &&
        *symbol.umax <= static_cast<uint64_t>(INT64_MAX) && size.constant > 0) {
      std::optional<int64_t> scaled =
          llvm::checkedMul(size.constant, static_cast<int64_t>(*symbol.umax));
      largest = scaled ? llvm::checkedAdd(*scaled, size.offset) : std::nullopt;
    }
    if (largest && size.constant + size.offset >= 0 &&
        *largest < (int64_t{1} << (width - 1))) {
      return size;
    }
    return SizeProduct{1, {{truncated, -1}}};
  }
  if (size.constant == 1 && size.symbols.size() == 1) {
    SizeSymbol symbol = size.symbols.front();
    if (symbol.width == 0 || symbol.width > width) {
      symbol.width = width;
    }
    return SizeProduct{1, {symbol}};
  }
  return SizeProduct{1, {{truncated, -1}}};
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

std::optional<SizeProduct> factorDim(Value tensor, int64_t dim, int depth,
                                     FactorBudget &budget);

// Factors an index or integer size through products, casts, assumptions and
// 0-d tensor round trips. A narrowing cast keeps its width. A wrapped i32
// product would break the reshape's element count or the broadcast's operand
// size, so products are read as exact.
std::optional<SizeProduct> factorSize(Value value, int depth,
                                      FactorBudget &budget) {
  if (++budget.visits > kMaxFactorVisits) {
    return std::nullopt;
  }
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) {
    return SizeProduct{constant.getSExtValue(), {}};
  }
  if (depth > 0) {
    if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>()) {
      std::optional<SizeProduct> input =
          factorSize(indexCast.getIn(), depth - 1, budget);
      std::optional<unsigned> width = getTruncationWidth(indexCast.getType());
      if (!input || !width) {
        return input;
      }
      return truncateSize(*input, *width, value);
    }
    // Element `i` of a 0-d or 1-d size vector, as shape reification builds.
    if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
      std::optional<int64_t> index = 0;
      if (extract.getIndices().size() == 1) {
        index = getConstantIntValue(extract.getIndices().front());
      } else if (!extract.getIndices().empty()) {
        index = std::nullopt;
      }
      // Casts of the whole vector apply to the element, innermost first.
      Value source = extract.getTensor();
      SmallVector<unsigned> widths;
      while (auto castOp = source.getDefiningOp<arith::IndexCastOp>()) {
        if (std::optional<unsigned> width =
                getTruncationWidth(castOp.getType())) {
          widths.push_back(*width);
        }
        source = castOp.getIn();
      }
      std::optional<SizeProduct> element;
      Operation *vector = source.getDefiningOp();
      if (auto fromElements =
              dyn_cast_if_present<tensor::FromElementsOp>(vector);
          fromElements && index &&
          *index < static_cast<int64_t>(fromElements.getNumOperands())) {
        element =
            factorSize(fromElements.getOperand(*index), depth - 1, budget);
      } else if (auto shapeOf = dyn_cast_if_present<shape::ShapeOfOp>(vector);
                 shapeOf && index && extract.getIndices().size() == 1) {
        element = factorDim(shapeOf.getArg(), *index, depth - 1, budget);
      }
      if (element) {
        for (unsigned width : llvm::reverse(widths)) {
          element = truncateSize(*element, width, value);
        }
        return element;
      }
    }
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
      std::optional<SizeProduct> lhs =
          factorSize(mul.getLhs(), depth - 1, budget);
      std::optional<SizeProduct> rhs =
          factorSize(mul.getRhs(), depth - 1, budget);
      std::optional<SizeProduct> product =
          lhs && rhs ? multiplySizes(*lhs, *rhs) : std::nullopt;
      if (product && getTruncationWidth(mul.getType())) {
        product->mayWrap = true;
      }
      return product;
    }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      std::optional<SizeProduct> lhs =
          factorSize(add.getLhs(), depth - 1, budget);
      std::optional<SizeProduct> rhs =
          factorSize(add.getRhs(), depth - 1, budget);
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      if (std::optional<SizeProduct> sum =
              addSizes(*lhs, *rhs, SizeSymbol{value, -1})) {
        if (getTruncationWidth(add.getType())) {
          sum->mayWrap = true;
        }
        return sum;
      }
    }
    if (auto extend = value.getDefiningOp<arith::ExtSIOp>()) {
      return factorSize(extend.getIn(), depth - 1, budget);
    }
    if (auto truncate = value.getDefiningOp<arith::TruncIOp>()) {
      std::optional<SizeProduct> input =
          factorSize(truncate.getIn(), depth - 1, budget);
      std::optional<unsigned> width = getTruncationWidth(truncate.getType());
      if (!input || !width) {
        return input;
      }
      return truncateSize(*input, *width, value);
    }
    if (auto dimOp = value.getDefiningOp<tensor::DimOp>()) {
      if (std::optional<int64_t> index = dimOp.getConstantIndex()) {
        return factorDim(dimOp.getSource(), *index, depth - 1, budget);
      }
    }
  }
  return SizeProduct{1, {{value, -1}}};
}

// Factors dimension `dim` of `tensor` through the ops that carry its size.
std::optional<SizeProduct> factorDim(Value tensor, int64_t dim, int depth,
                                     FactorBudget &budget) {
  if (++budget.visits > kMaxFactorVisits) {
    return std::nullopt;
  }
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(dim)) {
    return SizeProduct{type.getDimSize(dim), {}};
  }
  Operation *op = tensor.getDefiningOp();
  if (depth > 0 && op) {
    auto getDynamicIndex = [&](RankedTensorType shapedType) {
      return llvm::count_if(shapedType.getShape().take_front(dim),
                            ShapedType::isDynamic);
    };
    if (auto tie = dyn_cast<IREE::Flow::TensorTieShapeOp>(op)) {
      return factorSize(tie.getDynamicDims()[getDynamicIndex(type)], depth - 1,
                        budget);
    }
    if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
      return factorSize(empty.getDynamicSize(dim), depth - 1, budget);
    }
    if (auto castOp = dyn_cast<tensor::CastOp>(op)) {
      return factorDim(castOp.getSource(), dim, depth - 1, budget);
    }
    // StableHLO operations may not be converted yet.
    if (op->hasTrait<OpTrait::SameOperandsAndResultShape>() &&
        op->getNumOperands() > 0) {
      return factorDim(op->getOperand(0), dim, depth - 1, budget);
    }
    if (auto transpose = dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
      return factorDim(transpose.getOperand(), transpose.getPermutation()[dim],
                       depth - 1, budget);
    }
    if (auto dot = dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      // Results are batch, then free lhs, then free rhs dimensions.
      auto numbers = dot.getDotDimensionNumbers();
      ArrayRef<int64_t> lhsBatch = numbers.getLhsBatchingDimensions();
      if (dim < static_cast<int64_t>(lhsBatch.size())) {
        return factorDim(dot.getLhs(), lhsBatch[dim], depth - 1, budget);
      }
      int64_t index = lhsBatch.size();
      for (auto [operand, batch, contracting] :
           {std::make_tuple(dot.getLhs(), lhsBatch,
                            numbers.getLhsContractingDimensions()),
            std::make_tuple(dot.getRhs(), numbers.getRhsBatchingDimensions(),
                            numbers.getRhsContractingDimensions())}) {
        int64_t rank = cast<RankedTensorType>(operand.getType()).getRank();
        for (int64_t i = 0; i < rank; ++i) {
          if (llvm::is_contained(batch, i) ||
              llvm::is_contained(contracting, i)) {
            continue;
          }
          if (index++ == dim) {
            return factorDim(operand, i, depth - 1, budget);
          }
        }
      }
    }
    if (auto reduce = dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
      // Results keep the inputs' dimensions that are not reduced.
      Value input =
          reduce.getInputs()[cast<OpResult>(tensor).getResultNumber()];
      int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
      int64_t index = 0;
      for (int64_t i = 0; i < rank; ++i) {
        if (llvm::is_contained(reduce.getDimensions(), i)) {
          continue;
        }
        if (index++ == dim) {
          return factorDim(input, i, depth - 1, budget);
        }
      }
    }
    if (auto dot = dyn_cast<mlir::stablehlo::DotOp>(op)) {
      // Results are lhs but its last, then rhs but its first dimension.
      int64_t lhsFree =
          cast<RankedTensorType>(dot.getLhs().getType()).getRank() - 1;
      return dim < lhsFree ? factorDim(dot.getLhs(), dim, depth - 1, budget)
                           : factorDim(dot.getRhs(), dim - lhsFree + 1,
                                       depth - 1, budget);
    }
    Value shape;
    if (auto reshape = dyn_cast<mlir::stablehlo::DynamicReshapeOp>(op)) {
      shape = reshape.getOutputShape();
    } else if (auto reshape = dyn_cast<tensor::ReshapeOp>(op)) {
      shape = reshape.getShape();
    } else if (auto broadcast =
                   dyn_cast<mlir::stablehlo::DynamicBroadcastInDimOp>(op)) {
      shape = broadcast.getOutputDimensions();
    }
    if (auto fromElements =
            shape ? shape.getDefiningOp<tensor::FromElementsOp>() : nullptr;
        fromElements &&
        dim < static_cast<int64_t>(fromElements.getNumOperands())) {
      return factorSize(fromElements.getOperand(dim), depth - 1, budget);
    }
    if (auto destination = dyn_cast<DestinationStyleOpInterface>(op)) {
      return factorDim(
          destination.getTiedOpOperand(cast<OpResult>(tensor))->get(), dim,
          depth - 1, budget);
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      SizeProduct product;
      SmallVector<ReassociationIndices> groups =
          collapse.getReassociationIndices();
      for (int64_t source : groups[dim]) {
        std::optional<SizeProduct> factor =
            factorDim(collapse.getSrc(), source, depth - 1, budget);
        if (factor) {
          factor = multiplySizes(product, *factor);
        }
        if (!factor) {
          return std::nullopt;
        }
        product = *factor;
      }
      return product;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      OpFoldResult size = expand.getMixedOutputShape()[dim];
      if (auto value = dyn_cast<Value>(size)) {
        return factorSize(value, depth - 1, budget);
      }
    }
  }
  return SizeProduct{1, {{tensor, dim}}};
}

// Groups the dimensions of `wide` contiguously so that each group's product
// is the matching size of `narrow`; trailing ones join the last group.
std::optional<SmallVector<ReassociationIndices>>
groupContiguousSizes(ArrayRef<SizeProduct> wide, ArrayRef<SizeProduct> narrow) {
  SmallVector<ReassociationIndices> groups;
  size_t next = 0;
  for (const SizeProduct &target : narrow) {
    ReassociationIndices group;
    SizeProduct product;
    do {
      if (next == wide.size()) {
        return std::nullopt;
      }
      std::optional<SizeProduct> extended = multiplySizes(product, wide[next]);
      if (!extended || !dividesSize(*extended, target)) {
        return std::nullopt;
      }
      product = *extended;
      group.push_back(next++);
    } while (!equalSizes(product, target));
    groups.push_back(group);
  }
  for (; next < wide.size(); ++next) {
    if (groups.empty() || !wide[next].isOne()) {
      return std::nullopt;
    }
    groups.back().push_back(next);
  }
  return groups;
}

// Proves a reassociation for reshaping `operand` to `shape`, a vector of
// sizes assembled with tensor.from_elements.
std::optional<SmallVector<ReassociationIndices>>
getProvenReassociation(Value operand, Value shape,
                       RankedTensorType resultType) {
  auto fromElements = shape.getDefiningOp<tensor::FromElementsOp>();
  auto operandType = cast<RankedTensorType>(operand.getType());
  if (!fromElements ||
      static_cast<int64_t>(fromElements.getNumOperands()) !=
          resultType.getRank() ||
      operandType.getRank() == resultType.getRank()) {
    return std::nullopt;
  }
  FactorBudget budget;
  SmallVector<SizeProduct> operandSizes;
  for (int64_t i = 0; i < operandType.getRank(); ++i) {
    std::optional<SizeProduct> size =
        factorDim(operand, i, kMaxFactorDepth, budget);
    if (!size) {
      return std::nullopt;
    }
    operandSizes.push_back(*size);
  }
  SmallVector<SizeProduct> resultSizes;
  for (auto [i, element] : llvm::enumerate(fromElements.getOperands())) {
    std::optional<SizeProduct> size =
        factorSize(element, kMaxFactorDepth, budget);
    if (!size) {
      return std::nullopt;
    }
    // A static result size must agree with the shape vector.
    if (!resultType.isDynamicDim(i) &&
        !equalSizes(*size, SizeProduct{resultType.getDimSize(i), {}})) {
      return std::nullopt;
    }
    resultSizes.push_back(*size);
  }
  // A wrapped product equals the exact one only while the element count rules
  // it out: an empty tensor reshapes to any sizes with a zero.
  auto mayWrap = [](const SizeProduct &size) { return size.mayWrap; };
  if ((llvm::any_of(operandSizes, mayWrap) ||
       llvm::any_of(resultSizes, mayWrap)) &&
      !llvm::all_of(operandSizes,
                    [](const SizeProduct &size) { return size.isNonzero(); })) {
    return std::nullopt;
  }
  if (operandType.getRank() > resultType.getRank()) {
    return groupContiguousSizes(operandSizes, resultSizes);
  }
  return groupContiguousSizes(resultSizes, operandSizes);
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
    auto operandType = cast<RankedTensorType>(adaptor.getOperand().getType());
    bool isCollapse = operandType.getRank() > resultType.getRank();
    auto collapsedType = isCollapse ? resultType : operandType;
    std::optional<SmallVector<ReassociationIndices>> reassociation;
    // Element count determines a single dynamic dimension unless a static
    // dimension is zero. Multiple dynamic dimensions need shape analysis.
    if (collapsedType.getNumDynamicDims() <= 1 &&
        !llvm::is_contained(collapsedType.getShape(), 0)) {
      reassociation =
          getReassociationIndicesForReshape(operandType, resultType);
    }
    if (!reassociation && !llvm::is_contained(operandType.getShape(), 0) &&
        !llvm::is_contained(resultType.getShape(), 0)) {
      reassociation = getProvenReassociation(
          adaptor.getOperand(), adaptor.getOutputShape(), resultType);
    }
    if (reassociation) {
      if (isCollapse) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, resultType, adaptor.getOperand(), *reassociation);
      } else {
        SmallVector<OpFoldResult> outputShape;
        for (auto [i, size] : llvm::enumerate(resultType.getShape())) {
          outputShape.push_back(
              ShapedType::isDynamic(size)
                  ? OpFoldResult(extractIndex(rewriter, op.getLoc(),
                                              adaptor.getOutputShape(), i))
                  : OpFoldResult(rewriter.getIndexAttr(size)));
        }
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, resultType, adaptor.getOperand(), *reassociation, outputShape);
      }
      return success();
    }
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getOperand(), adaptor.getOutputShape());
    return success();
  }
};

struct DynamicPadGeometry {
  SmallVector<Value> scratchDynamicDims, resultDynamicDims;
  SmallVector<OpFoldResult> insertSizes, insertOffsets, insertStrides;
  SmallVector<OpFoldResult> extractOffsets, extractSizes;
  Value nonempty;
};

DynamicPadGeometry computeDynamicPadGeometry(
    OpBuilder &rewriter, Location loc, Value operand,
    ArrayRef<OpFoldResult> lowPadding, ArrayRef<OpFoldResult> highPadding,
    ArrayRef<OpFoldResult> interiorPadding, RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  DynamicPadGeometry geometry;
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  geometry.nonempty = rewriter.createOrFold<arith::ConstantIntOp>(loc, 1, 1);
  geometry.insertSizes = tensor::getMixedSizes(rewriter, loc, operand);
  for (int64_t i = 0; i < rank; ++i) {
    Value dim =
        getValueOrCreateConstantIndexOp(rewriter, loc, geometry.insertSizes[i]);
    Value low = getValueOrCreateConstantIndexOp(rewriter, loc, lowPadding[i]);
    Value high = getValueOrCreateConstantIndexOp(rewriter, loc, highPadding[i]);
    Value interior =
        getValueOrCreateConstantIndexOp(rewriter, loc, interiorPadding[i]);

    Value lowPos = arith::MaxSIOp::create(rewriter, loc, low, zero);
    Value highPos = arith::MaxSIOp::create(rewriter, loc, high, zero);
    Value lowNeg = arith::MaxSIOp::create(
        rewriter, loc, arith::SubIOp::create(rewriter, loc, zero, low), zero);

    Value dimMinusOne = arith::SubIOp::create(rewriter, loc, dim, one);
    Value clampedDimMinusOne =
        arith::MaxSIOp::create(rewriter, loc, dimMinusOne, zero);
    Value interiorTotal =
        arith::MulIOp::create(rewriter, loc, clampedDimMinusOne, interior);
    Value dimAndInterior =
        arith::AddIOp::create(rewriter, loc, dim, interiorTotal);

    // The scratch tensor's size depends on lowPos/highPos, which are
    // runtime values even where the result type's dim is static, so every
    // dim of the scratch tensor is dynamic.
    Value dimAndLowPos =
        arith::AddIOp::create(rewriter, loc, dimAndInterior, lowPos);
    Value filledDim =
        arith::AddIOp::create(rewriter, loc, dimAndLowPos, highPos);
    geometry.scratchDynamicDims.push_back(filledDim);

    if (resultType.isDynamicDim(i)) {
      Value dimAndLow =
          arith::AddIOp::create(rewriter, loc, dimAndInterior, low);
      Value resultDim = arith::AddIOp::create(rewriter, loc, dimAndLow, high);
      geometry.extractSizes.push_back(resultDim);
      geometry.resultDynamicDims.push_back(resultDim);
    } else {
      geometry.extractSizes.push_back(
          rewriter.getIndexAttr(resultType.getDimSize(i)));
    }

    geometry.insertOffsets.push_back(lowPos);
    geometry.insertStrides.push_back(
        arith::AddIOp::create(rewriter, loc, interior, one).getResult());
    geometry.extractOffsets.push_back(lowNeg);
    Value size = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                 geometry.extractSizes.back());
    Value positive = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, size, zero);
    geometry.nonempty =
        rewriter.createOrFold<arith::AndIOp>(loc, geometry.nonempty, positive);
  }
  return geometry;
}

Value createPaddedTensor(OpBuilder &rewriter, Location loc, Value operand,
                         Value paddingValue, RankedTensorType resultType,
                         const DynamicPadGeometry &geometry) {
  int64_t rank = resultType.getRank();
  SmallVector<int64_t> scratchShape(rank, ShapedType::kDynamic);
  Value empty = tensor::EmptyOp::create(rewriter, loc, scratchShape,
                                        resultType.getElementType(),
                                        geometry.scratchDynamicDims);
  Value filled =
      linalg::FillOp::create(rewriter, loc, paddingValue, empty).result();
  Value inserted = tensor::InsertSliceOp::create(
                       rewriter, loc, operand, filled, geometry.insertOffsets,
                       geometry.insertSizes, geometry.insertStrides)
                       .getResult();
  // Negative edge padding crops, which the insert cannot express.
  SmallVector<OpFoldResult> extractStrides(rank, rewriter.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(rewriter, loc, resultType, inserted,
                                        geometry.extractOffsets,
                                        geometry.extractSizes, extractStrides);
}

Value createCroppedPad(OpBuilder &rewriter, Location loc, Value operand,
                       Value paddingValue, RankedTensorType resultType,
                       const DynamicPadGeometry &geometry) {
  // Empty crops need no scratch buffer. In particular, runtime convolution
  // padding can remove an arbitrarily large dilated input entirely.
  auto ifOp =
      scf::IfOp::create(rewriter, loc, TypeRange{resultType}, geometry.nonempty,
                        /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value result = createPaddedTensor(rewriter, loc, operand, paddingValue,
                                      resultType, geometry);
    scf::YieldOp::create(rewriter, loc, result);
    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    Value resultEmpty = tensor::EmptyOp::create(rewriter, loc, resultType,
                                                geometry.resultDynamicDims);
    scf::YieldOp::create(rewriter, loc, resultEmpty);
  }
  return IREE::Flow::TensorTieShapeOp::create(
      rewriter, loc, resultType, ifOp.getResult(0), geometry.resultDynamicDims);
}

// tensor.pad has no interior padding, and whether the interior amounts are
// zero is unknown here, so the strided insert covers every case.
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

    auto extractAll = [&](Value padding) {
      return llvm::map_to_vector(
          llvm::seq<int64_t>(resultType.getRank()), [&](int64_t i) {
            return OpFoldResult(extractIndex(rewriter, loc, padding, i));
          });
    };
    auto geometry = computeDynamicPadGeometry(
        rewriter, loc, adaptor.getOperand(),
        extractAll(adaptor.getEdgePaddingLow()),
        extractAll(adaptor.getEdgePaddingHigh()),
        extractAll(adaptor.getInteriorPadding()), resultType);
    rewriter.replaceOp(op,
                       createCroppedPad(rewriter, loc, adaptor.getOperand(),
                                        paddingValue, resultType, geometry));
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
    if (!resultType || resultType.hasStaticShape()) {
      return failure();
    }
    auto isNegative = [](int64_t v) { return v < 0; };
    if (llvm::any_of(op.getInteriorPadding(),
                     [](int64_t v) { return v != 0; }) ||
        llvm::any_of(op.getEdgePaddingLow(), isNegative) ||
        llvm::any_of(op.getEdgePaddingHigh(), isNegative)) {
      // tensor.pad takes neither, so pad nonnegatively and crop.
      Location loc = op.getLoc();
      auto toAttrs = [&](ArrayRef<int64_t> values) {
        return getAsIndexOpFoldResult(rewriter.getContext(), values);
      };
      DynamicPadGeometry geometry = computeDynamicPadGeometry(
          rewriter, loc, adaptor.getOperand(), toAttrs(op.getEdgePaddingLow()),
          toAttrs(op.getEdgePaddingHigh()), toAttrs(op.getInteriorPadding()),
          resultType);
      Value paddingValue = rewriter.createOrFold<tensor::ExtractOp>(
          loc, adaptor.getPaddingValue());
      rewriter.replaceOp(op,
                         createCroppedPad(rewriter, loc, adaptor.getOperand(),
                                          paddingValue, resultType, geometry));
      return success();
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
computeConvolutionResultShape(OpBuilder &rewriter,
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
  // An empty input/kernel stays empty under dilation (StableHLO C25).
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
// but requires static spatial results. Compute those sizes from StableHLO C25.
// Grouped convolutions additionally split the group dimensions for Linalg.
struct DynamicConvolutionOpConversion final
    : OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using Base::Base;

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
    // The preprocessing pass puts convolutions into this layout.
    if (dims.getInputBatchDimension() != 0 ||
        dims.getOutputBatchDimension() != 0 ||
        dims.getInputFeatureDimension() != rank - 1 ||
        dims.getOutputFeatureDimension() != rank - 1 ||
        dims.getKernelInputFeatureDimension() != rank - 2 ||
        dims.getKernelOutputFeatureDimension() != rank - 1) {
      return rewriter.notifyMatchFailure(
          op, "expected canonical dimension numbers");
    }
    for (int64_t i = 0; i < rank - 2; ++i) {
      if (dims.getInputSpatialDimensions()[i] != i + 1 ||
          dims.getOutputSpatialDimensions()[i] != i + 1 ||
          dims.getKernelSpatialDimensions()[i] != i) {
        return rewriter.notifyMatchFailure(
            op, "expected canonical spatial dimensions");
      }
    }

    Location loc = op.getLoc();
    Value input = adaptor.getLhs();
    Value filter = adaptor.getRhs();
    auto window = getConvolutionWindow(op);
    auto shape = computeConvolutionResultShape(rewriter, op, input, filter,
                                               resultType, window);
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, resultType, shape.dynamicDims);
    Value init = mlir::stablehlo::fillTensorWithZeros(rewriter, loc, empty);

    // Avoid materializing a negatively-sized crop (or running a convolution)
    // when no window fits. The shape calculation above still follows C25.
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
  using Base = OpConversionPattern<mlir::stablehlo::GatherOp>;
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

struct DynamicGatherOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicGatherOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = adaptor.getOperand();
    Value startIndices = adaptor.getStartIndices();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto startIndicesType = dyn_cast<RankedTensorType>(startIndices.getType());
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!operandType || !startIndicesType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    }

    int64_t operandRank = operandType.getRank();
    int64_t resultRank = resultType.getRank();
    auto dims = op.getDimensionNumbers();
    int64_t indexVectorDim = dims.getIndexVectorDim();
    ArrayRef<int64_t> offsetDims = dims.getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims = dims.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims = dims.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        dims.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap = dims.getStartIndexMap();

    auto isCollapsedOrBatching = [&](int64_t dim) {
      return llvm::is_contained(collapsedSliceDims, dim) ||
             llvm::is_contained(operandBatchingDims, dim);
    };
    SmallVector<unsigned> remappedOffsetDims;
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      if (!isCollapsedOrBatching(dim)) {
        remappedOffsetDims.push_back(static_cast<unsigned>(dim));
      }
    }
    if (remappedOffsetDims.size() != offsetDims.size()) {
      return rewriter.notifyMatchFailure(op, "inconsistent offset dimensions");
    }

    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      constants.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));
    }

    Value empty = mlir::stablehlo::getEmptyTensorFor(rewriter, loc, resultType,
                                                     op, adaptor.getOperands());
    SmallVector<Value> operandDims(operandRank);
    SmallVector<Value> sliceSizes(operandRank, constants[1]);
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      operandDims[dim] =
          rewriter.createOrFold<tensor::DimOp>(loc, operand, dim);
    }
    for (auto [operandDim, resultDim] :
         llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      sliceSizes[operandDim] =
          rewriter.createOrFold<tensor::DimOp>(loc, empty, resultDim);
    }
    SmallVector<Value> largestStartIndices(operandRank);
    for (int64_t dim = 0; dim < operandRank; ++dim) {
      largestStartIndices[dim] = rewriter.createOrFold<arith::SubIOp>(
          loc, operandDims[dim], sliceSizes[dim]);
    }

    unsigned indexWidth = mlir::DataLayout::closest(op)
                              .getTypeSizeInBits(rewriter.getIndexType())
                              .getFixedValue();
    Type originalStartIndicesElementType =
        cast<ShapedType>(op.getStartIndices().getType()).getElementType();
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputs=*/empty,
        SmallVector<AffineMap>{rewriter.getMultiDimIdentityMap(resultRank)},
        mlir::stablehlo::getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &builder, Location nestedLoc, ValueRange) {
          SmallVector<Value> resultIndices;
          SmallVector<Value> gatherIndex;
          for (int64_t dim = 0; dim < resultRank; ++dim) {
            Value index = linalg::IndexOp::create(builder, nestedLoc, dim);
            resultIndices.push_back(index);
            if (!llvm::is_contained(offsetDims, dim)) {
              gatherIndex.push_back(index);
            }
          }

          SmallVector<Value> startIndexByOperand(operandRank);
          for (size_t i = 0, e = startIndexMap.size(); i < e; ++i) {
            SmallVector<Value> indices(gatherIndex);
            if (indexVectorDim != startIndicesType.getRank()) {
              indices.insert(indices.begin() + indexVectorDim, constants[i]);
            }
            Value index = tensor::ExtractOp::create(builder, nestedLoc,
                                                    startIndices, indices);
            startIndexByOperand[startIndexMap[i]] = index;
          }

          SmallVector<Value> indexFromBatching(operandRank, constants[0]);
          for (auto [operandDim, indicesDim] :
               llvm::zip_equal(operandBatchingDims, startIndicesBatchingDims)) {
            indexFromBatching[operandDim] =
                gatherIndex[indicesDim - (indicesDim < indexVectorDim ? 0 : 1)];
          }

          SmallVector<Value> combinedIndex;
          for (int64_t operandDim = 0; operandDim < operandRank; ++operandDim) {
            Value startIndex = constants[0];
            if (startIndexByOperand[operandDim]) {
              startIndex = clampGatherIndex(
                  builder, nestedLoc, startIndexByOperand[operandDim],
                  originalStartIndicesElementType,
                  largestStartIndices[operandDim], indexWidth);
            }
            Value offsetIndex = constants[0];
            auto offset = llvm::find(remappedOffsetDims, operandDim);
            if (offset != remappedOffsetDims.end()) {
              offsetIndex =
                  resultIndices[offsetDims[offset -
                                           remappedOffsetDims.begin()]];
            }
            combinedIndex.push_back(builder.createOrFold<arith::AddIOp>(
                nestedLoc, builder.getIndexType(),
                builder.createOrFold<arith::AddIOp>(
                    nestedLoc, builder.getIndexType(), startIndex,
                    indexFromBatching[operandDim]),
                offsetIndex));
          }

          Value element = tensor::ExtractOp::create(builder, nestedLoc, operand,
                                                    combinedIndex);
          linalg::YieldOp::create(builder, nestedLoc, element);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
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

    ArrayRef<int64_t> broadcastDims = op.getBroadcastDimensions();
    SmallVector<std::optional<bool>> expanding(operandType.getRank());
    for (auto [i, size] : llvm::enumerate(operandType.getShape())) {
      if (!ShapedType::isDynamic(size)) {
        expanding[i] = size == 1;
      }
    }
    if (auto dims = op.getKnownExpandingDimensions()) {
      for (int64_t i : *dims) {
        expanding[i] = true;
      }
    }
    if (auto dims = op.getKnownNonexpandingDimensions()) {
      for (int64_t i : *dims) {
        expanding[i] = false;
      }
    }
    // One budget bounds every proof for this broadcast.
    FactorBudget budget;
    bool inferredNonexpanding = false;
    for (auto [operandDim, resultDim] : llvm::enumerate(broadcastDims)) {
      if (expanding[operandDim].has_value()) {
        continue;
      }
      Value outputElement =
          getShapeElement(op.getOutputDimensions(), resultDim);
      if (!outputElement) {
        continue;
      }
      std::optional<SizeProduct> sourceSize =
          factorDim(operand, operandDim, kMaxFactorDepth, budget);
      std::optional<SizeProduct> outputSize =
          factorSize(outputElement, kMaxFactorDepth, budget);
      // A wrapped operand size may be one where the output size is not.
      if (sourceSize && outputSize && !sourceSize->mayWrap &&
          equalSizes(*sourceSize, *outputSize)) {
        expanding[operandDim] = false;
        inferredNonexpanding = true;
      }
    }
    if (llvm::all_of(expanding, [](auto value) { return value.has_value(); })) {
      if (!inferredNonexpanding) {
        return rewriter.notifyMatchFailure(op, "expansion behavior is known");
      }
      // Record the proof so the upstream pattern emits an affine-map generic.
      SmallVector<int64_t> knownExpanding;
      SmallVector<int64_t> knownNonexpanding;
      for (auto [i, value] : llvm::enumerate(expanding)) {
        if (ShapedType::isDynamic(operandType.getDimSize(i))) {
          (*value ? knownExpanding : knownNonexpanding).push_back(i);
        }
      }
      rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
          op, op.getType(), operand, adaptor.getOutputDimensions(),
          rewriter.getDenseI64ArrayAttr(broadcastDims),
          rewriter.getDenseI64ArrayAttr(knownExpanding),
          rewriter.getDenseI64ArrayAttr(knownNonexpanding));
      return success();
    }

    Location loc = op.getLoc();
    Value empty = mlir::stablehlo::getEmptyTensorFor(rewriter, loc, resultType,
                                                     op, adaptor.getOperands());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> isExpanding(operandType.getRank());
    for (int64_t i = 0, e = operandType.getRank(); i < e; ++i) {
      if (expanding[i].has_value()) {
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
          for (auto [operandDim, resultDim] : llvm::enumerate(broadcastDims)) {
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
    rewriter.replaceOp(op, generic.getResults());
    return success();
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

// Defer only pooling cases whose upstream shape reification matches C15.
bool matchesUpstreamPooling(mlir::stablehlo::ReduceWindowOp op) {
  int64_t rank = cast<ShapedType>(op.getResultTypes()[0]).getRank();
  if ((rank != 4 && rank != 5) ||
      (op.getPadding() &&
       !mlir::stablehlo::isSplatValue(*op.getPadding(), 0))) {
    return false;
  }
  if (auto dilations = op.getBaseDilations();
      dilations &&
      !llvm::all_of(*dilations, [](int64_t value) { return value == 1; })) {
    return false;
  }
  if (auto dilations = op.getWindowDilations();
      dilations &&
      !llvm::all_of(*dilations, [](int64_t value) { return value == 1; })) {
    return false;
  }
  int64_t lastDim = rank - 1;
  if (op.getWindowDimensions()[0] != 1 ||
      op.getWindowDimensions()[lastDim] != 1) {
    return false;
  }
  if (auto strides = op.getWindowStrides();
      strides && (strides.value()[0] != 1 || strides.value()[lastDim] != 1)) {
    return false;
  }
  for (auto [index, input] : llvm::enumerate(op.getInputs())) {
    // Upstream reads batch and channel through tensor.dim; only spatial dims
    // enter its output-shape formula.
    auto inputType = cast<ShapedType>(input.getType());
    if (!inputType.getElementType().isF32()) {
      return false;
    }
    for (int64_t dim = 1; dim < lastDim; ++dim) {
      if (inputType.isDynamicDim(dim) ||
          inputType.getDimSize(dim) < op.getWindowDimensions()[dim]) {
        return false;
      }
    }
    Operation *reducer = getReduceWindowReductionOp(op, index);
    if (!reducer || !isa<mlir::stablehlo::MinOp, mlir::stablehlo::MaxOp,
                         mlir::stablehlo::AddOp>(*reducer)) {
      return false;
    }
  }
  return true;
}

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

    MLIRContext *context = op->getContext();
    Location loc = op.getLoc();
    int64_t rank = op.getWindowDimensions().size();
    SmallVector<int64_t> windowDimensions(op.getWindowDimensions());
    SmallVector<int64_t> padding(2 * rank, 0);
    if (op.getPadding()) {
      padding = llvm::to_vector(op.getPadding()->getValues<int64_t>());
    }
    SmallVector<int64_t> baseDilations(rank, 1);
    if (op.getBaseDilations()) {
      baseDilations = llvm::to_vector(*op.getBaseDilations());
    }
    SmallVector<int64_t> windowStrides(rank, 1);
    if (op.getWindowStrides()) {
      windowStrides = llvm::to_vector(*op.getWindowStrides());
    }
    SmallVector<int64_t> windowDilations(rank, 1);
    if (op.getWindowDilations()) {
      windowDilations = llvm::to_vector(*op.getWindowDilations());
    }

    SmallVector<AffineExpr> sourceExprs, windowExprs, resultExprs;
    SmallVector<int64_t> filteredWindowShape;
    int64_t windowDim = 0;
    for (int64_t dim = 0; dim < rank; ++dim) {
      AffineExpr sourceExpr = getAffineDimExpr(dim, context);
      if (windowStrides[dim] != 1) {
        sourceExpr = sourceExpr * windowStrides[dim];
      }
      if (windowDimensions[dim] != 1) {
        filteredWindowShape.push_back(windowDimensions[dim]);
        AffineExpr windowExpr = getAffineDimExpr(rank + windowDim, context);
        windowExprs.push_back(windowExpr);
        if (windowDilations[dim] != 1) {
          windowExpr = windowExpr * windowDilations[dim];
        }
        sourceExpr = sourceExpr + windowExpr;
        ++windowDim;
      }
      sourceExprs.push_back(sourceExpr);
      resultExprs.push_back(getAffineDimExpr(dim, context));
    }
    SmallVector<AffineMap> maps(3, AffineMap::get(context));
    if (rank > 0) {
      maps = AffineMap::inferFromExprList(
          {sourceExprs, windowExprs, resultExprs}, context);
    }
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.append(op.getInputs().size(), maps[0]);
    indexingMaps.push_back(maps[1]);
    indexingMaps.append(op.getResults().size(), maps[2]);

    SmallVector<Value> inputs = llvm::to_vector(adaptor.getInputs());
    SmallVector<Value> initValues = llvm::to_vector(adaptor.getInitValues());
    if (llvm::any_of(padding, [](int64_t value) { return value != 0; }) ||
        llvm::any_of(baseDilations, [](int64_t value) { return value != 1; })) {
      SmallVector<int64_t> lows(rank), highs(rank), interiors(rank);
      for (int64_t dim = 0; dim < rank; ++dim) {
        lows[dim] = padding[2 * dim];
        highs[dim] = padding[2 * dim + 1];
        interiors[dim] = baseDilations[dim] - 1;
      }
      for (auto [index, initValue] : llvm::enumerate(initValues)) {
        inputs[index] = mlir::stablehlo::PadOp::create(
            rewriter, loc, inputs[index], initValue, lows, highs, interiors);
      }
    }

    auto constantIndex = [&](int64_t value) {
      return arith::ConstantIndexOp::create(rewriter, loc, value).getResult();
    };
    SmallVector<Value> seeds;
    for (auto [input, initValue, resultType] :
         llvm::zip(inputs, initValues, resultTypes)) {
      auto tensorType = cast<RankedTensorType>(resultType);
      SmallVector<Value> dynamicDims;
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (!tensorType.isDynamicDim(dim)) {
          continue;
        }
        Value paddedExtent = tensor::DimOp::create(rewriter, loc, input, dim);
        int64_t dilatedWindow =
            (windowDimensions[dim] - 1) * windowDilations[dim] + 1;
        Value span = arith::SubIOp::create(rewriter, loc, paddedExtent,
                                           constantIndex(dilatedWindow));
        Value nonnegative = rewriter.createOrFold<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, span, constantIndex(0));
        Value extent = arith::DivSIOp::create(
            rewriter, loc, span, constantIndex(windowStrides[dim]));
        extent = arith::AddIOp::create(rewriter, loc, extent, constantIndex(1));
        dynamicDims.push_back(rewriter.createOrFold<arith::SelectOp>(
            loc, nonnegative, extent, constantIndex(0)));
      }
      Value empty =
          tensor::EmptyOp::create(rewriter, loc, tensorType, dynamicDims);
      Value scalarInit =
          rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      seeds.push_back(
          linalg::FillOp::create(rewriter, loc, scalarInit, empty).result());
    }
    Type windowElementType =
        cast<ShapedType>(inputs.front().getType()).getElementType();
    inputs.push_back(tensor::EmptyOp::create(rewriter, loc, filteredWindowShape,
                                             windowElementType));
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultTypes, inputs, seeds, indexingMaps,
        mlir::stablehlo::getParallelAndReductionIterators(
            rank + filteredWindowShape.size(), filteredWindowShape.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    Region &region = generic.getRegion();
    rewriter.cloneRegionBefore(op.getBody(), region, region.end());
    TypeConverter::SignatureConversion signatureConversion(
        op.getInputs().size() + op.getResults().size());
    // StableHLO orders accumulator arguments first and input arguments second;
    // linalg.generic orders input scalars before output accumulator scalars.
    for (auto [index, type] : llvm::enumerate(resultTypes)) {
      signatureConversion.addInputs(op.getInputs().size() + index,
                                    cast<ShapedType>(type).getElementType());
    }
    signatureConversion.addInputs(windowElementType);
    for (auto [index, input] :
         llvm::enumerate(ArrayRef<Value>(inputs).drop_back())) {
      signatureConversion.addInputs(
          index, cast<ShapedType>(input.getType()).getElementType());
    }
    rewriter.applySignatureConversion(&region.front(), signatureConversion,
                                      getTypeConverter());
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  patterns
      ->add<DynamicReshapeOpConversion, PadOpConversion, DynamicPadOpConversion,
            DynamicConvolutionOpConversion, WideIndexGatherOpConversion,
            DynamicGatherOpConversion, DynamicBroadcastInDimOpConversion,
            DynamicReduceWindowOpConversion>(typeConverter, context,
                                             PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
