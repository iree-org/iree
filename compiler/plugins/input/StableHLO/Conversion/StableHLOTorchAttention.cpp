// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOTorchAttention.h"

#include <cmath>

#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Yields `score`, or -inf where the key position is after the query position,
// as torch's tril(ones(L, S)). Score index `queryIndex` is the query position
// and the next one the key position.
void buildCausalScore(OpBuilder &builder, Location loc, Value score,
                      int64_t queryIndex) {
  Value query = IREE::LinalgExt::IndexOp::create(builder, loc, queryIndex);
  Value key = IREE::LinalgExt::IndexOp::create(builder, loc, queryIndex + 1);
  Value hidden = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ugt,
                                       key, query);
  auto scoreType = cast<FloatType>(score.getType());
  Value negInf = arith::ConstantOp::create(
      builder, loc,
      builder.getFloatAttr(scoreType,
                           APFloat::getInf(scoreType.getFloatSemantics(),
                                           /*Negative=*/true)));
  Value masked = arith::SelectOp::create(builder, loc, hidden, negInf, score);
  IREE::LinalgExt::YieldOp::create(builder, loc, masked);
}

// Zeroes the rows of `attention` whose mask hides every key, as torch's
// _safe_softmax does; the attention op would divide by an empty sum there.
// `maskBatchDims` are the batch dims of `attention` that the mask keeps.
Value zeroHiddenRows(PatternRewriter &rewriter, Location loc, Value mask,
                     Value attention, ArrayRef<int64_t> maskBatchDims) {
  MLIRContext *context = rewriter.getContext();
  auto maskType = cast<RankedTensorType>(mask.getType());
  int64_t maskRank = maskType.getRank();
  SmallVector<OpFoldResult> rowSizes;
  for (int64_t i = 0; i < maskRank - 1; ++i) {
    rowSizes.push_back(tensor::getMixedSize(rewriter, loc, mask, i));
  }
  Value hidden =
      tensor::EmptyOp::create(rewriter, loc, rowSizes, rewriter.getI1Type());
  Value none = arith::ConstantIntOp::create(rewriter, loc, 0, 1);
  hidden = linalg::FillOp::create(rewriter, loc, none, hidden).getResult(0);
  AffineMap maskMap = rewriter.getMultiDimIdentityMap(maskRank);
  SmallVector<utils::IteratorType> iterators(maskRank - 1,
                                             utils::IteratorType::parallel);
  iterators.push_back(utils::IteratorType::reduction);
  auto floatType = dyn_cast<FloatType>(maskType.getElementType());
  Value visible =
      linalg::GenericOp::create(
          rewriter, loc, hidden.getType(), mask, hidden,
          ArrayRef<AffineMap>{maskMap, maskMap.dropResult(maskRank - 1)},
          iterators,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value seen = args[0];
            if (floatType) {
              // A NaN entry is no -inf, so its row stays visible.
              Value negativeInfinity = arith::ConstantOp::create(
                  b, nestedLoc,
                  b.getFloatAttr(floatType,
                                 APFloat::getInf(floatType.getFloatSemantics(),
                                                 /*Negative=*/true)));
              seen =
                  arith::CmpFOp::create(b, nestedLoc, arith::CmpFPredicate::UNE,
                                        args[0], negativeInfinity);
            }
            Value any = arith::OrIOp::create(b, nestedLoc, seen, args[1]);
            linalg::YieldOp::create(b, nestedLoc, any);
          })
          .getResult(0);

  auto resultType = cast<RankedTensorType>(attention.getType());
  int64_t rank = resultType.getRank();
  SmallVector<AffineExpr> rowExprs;
  for (int64_t i : maskBatchDims) {
    rowExprs.push_back(rewriter.getAffineDimExpr(i));
  }
  rowExprs.push_back(rewriter.getAffineDimExpr(rank - 2));
  AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  Value init = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, attention),
      resultType.getElementType());
  return linalg::GenericOp::create(
             rewriter, loc, resultType, ValueRange{attention, visible}, init,
             ArrayRef<AffineMap>{identity,
                                 AffineMap::get(rank, 0, rowExprs, context),
                                 identity},
             SmallVector<utils::IteratorType>(rank,
                                              utils::IteratorType::parallel),
             [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
               Value zero = arith::ConstantOp::create(
                   b, nestedLoc, b.getZeroAttr(args[0].getType()));
               Value selected = arith::SelectOp::create(b, nestedLoc, args[1],
                                                        args[0], zero);
               linalg::YieldOp::create(b, nestedLoc, selected);
             })
      .getResult(0);
}

// Returns true if |mask| is a constant with a visible key in every row; NaN
// counts as visible, as in zeroHiddenRows.
bool hasVisibleKeyInEveryRow(Value mask) {
  DenseElementsAttr values;
  if (!matchPattern(mask, m_Constant(&values))) {
    return false;
  }
  int64_t rowLength = values.getType().getShape().back();
  if (rowLength == 0) {
    return false;
  }
  SmallVector<bool> visible;
  if (values.getElementType().isInteger(1)) {
    visible = llvm::to_vector(values.getValues<bool>());
  } else {
    for (const APFloat &value : values.getValues<APFloat>()) {
      visible.push_back(!(value.isInfinity() && value.isNegative()));
    }
  }
  for (size_t row = 0; row < visible.size(); row += rowLength) {
    if (!llvm::is_contained(ArrayRef(visible).slice(row, rowLength), true)) {
      return false;
    }
  }
  return true;
}

LogicalResult rewriteAttention(mlir::stablehlo::CompositeOp op,
                               PatternRewriter &rewriter) {
  auto attrs = op->getAttrOfType<DictionaryAttr>("composite_attributes");
  if (!attrs || !op->getDiscardableAttrs().empty()) {
    return failure();
  }
  for (NamedAttribute attr : attrs) {
    if (!llvm::is_contained({"is_causal", "enable_gqa", "scale"},
                            attr.getName().strref())) {
      return failure();
    }
  }
  auto isCausal = attrs.getAs<BoolAttr>("is_causal");
  auto enableGqa = attrs.getAs<BoolAttr>("enable_gqa");
  auto scaleAttr = attrs.getAs<FloatAttr>("scale");
  if (!isCausal || !enableGqa || (attrs.contains("scale") && !scaleAttr)) {
    return failure();
  }

  size_t numInputs = op.getInputs().size();
  if ((numInputs != 3 && numInputs != 4) || op->getNumResults() != 1) {
    return failure();
  }
  auto queryType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
  auto keyType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
  auto valueType = dyn_cast<RankedTensorType>(op.getInputs()[2].getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!queryType || !keyType || !valueType || !resultType) {
    return failure();
  }
  auto elementType = dyn_cast<FloatType>(queryType.getElementType());
  int64_t rank = queryType.getRank();
  // The attention op accumulates in f32, which would narrow f64.
  if (!elementType || elementType.getWidth() > 32 || rank < 2 ||
      keyType.getElementType() != elementType ||
      valueType.getElementType() != elementType ||
      resultType.getElementType() != elementType || keyType.getRank() != rank ||
      valueType.getRank() != rank || resultType.getRank() != rank) {
    return failure();
  }
  ArrayRef<int64_t> q = queryType.getShape(), k = keyType.getShape(),
                    v = valueType.getShape(), r = resultType.getShape();
  int64_t numBatches = rank - 2;
  ArrayRef<int64_t> batch = q.take_front(numBatches);
  // Grouped-query attention shares each key and value head among `groupSize`
  // consecutive query heads, as torch's repeat_interleave. Run-time head
  // counts may differ, so they prove no grouping.
  int64_t groupSize = 1;
  if (enableGqa.getValue() && numBatches > 0 &&
      (ShapedType::isDynamic(q[numBatches - 1]) ||
       ShapedType::isDynamic(k[numBatches - 1]))) {
    return failure();
  }
  if (enableGqa.getValue() && numBatches > 0 &&
      q[numBatches - 1] != k[numBatches - 1]) {
    int64_t queryHeads = q[numBatches - 1], keyHeads = k[numBatches - 1];
    if (ShapedType::isDynamic(queryHeads) || ShapedType::isDynamic(keyHeads) ||
        keyHeads <= 0 || queryHeads % keyHeads != 0) {
      return failure();
    }
    groupSize = queryHeads / keyHeads;
  }
  // Key and value heads are the query heads divided into groups.
  SmallVector<int64_t> keyBatch(batch);
  if (groupSize > 1) {
    keyBatch.back() /= groupSize;
  }
  if (k.take_front(numBatches) != ArrayRef(keyBatch) ||
      v.take_front(numBatches) != ArrayRef(keyBatch) ||
      r.take_front(numBatches) != batch || q[rank - 1] != k[rank - 1] ||
      k[rank - 2] != v[rank - 2] || r[rank - 2] != q[rank - 2] ||
      r[rank - 1] != v[rank - 1]) {
    return failure();
  }
  int64_t l = q[rank - 2], s = k[rank - 2], e = q[rank - 1];
  // Every check precedes the first created op: a declined rewrite leaves the
  // IR unchanged.
  double scaleValue;
  if (scaleAttr) {
    scaleValue = scaleAttr.getValueAsDouble();
  } else if (!ShapedType::isDynamic(e)) {
    scaleValue = 1.0 / std::sqrt(static_cast<double>(e));
  } else {
    return failure();
  }

  Location loc = op.getLoc();
  Value mask;
  // Batch dims the mask keeps; a unit mask dim broadcasts over its batch dim.
  SmallVector<int64_t> maskBatchDims;
  if (numInputs == 4) {
    mask = op.getInputs()[3];
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (isCausal.getValue() || !maskType ||
        (!maskType.getElementType().isInteger(1) &&
         maskType.getElementType() != elementType)) {
      return failure();
    }
    ArrayRef<int64_t> maskShape = maskType.getShape();
    if (maskShape.size() < 2 || maskShape.take_back(2) != ArrayRef{l, s} ||
        (maskShape.size() != 2 &&
         maskShape.size() != static_cast<size_t>(numBatches) + 2)) {
      return failure();
    }
    for (int64_t i = 0; i < numBatches && maskShape.size() != 2; ++i) {
      if (maskShape[i] == batch[i]) {
        maskBatchDims.push_back(i);
      } else if (maskShape[i] != 1) {
        return failure();
      }
    }
    // Collapse each broadcast unit dim into the dim after it.
    if (maskShape.size() != 2 &&
        maskBatchDims.size() != static_cast<size_t>(numBatches)) {
      SmallVector<ReassociationIndices> groups;
      ReassociationIndices pending;
      for (int64_t dim = 0; dim < static_cast<int64_t>(maskShape.size());
           ++dim) {
        pending.push_back(dim);
        if (dim >= numBatches || llvm::is_contained(maskBatchDims, dim)) {
          groups.push_back(pending);
          pending.clear();
        }
      }
      mask = tensor::CollapseShapeOp::create(rewriter, loc, mask, groups);
    }
  }

  Value scale = arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(elementType, scaleValue));

  // Grouped queries split their head dim into (key heads, groupSize); so does
  // a mask that keeps the head dim. `mask` stays row-shaped for zeroHiddenRows.
  int64_t head = numBatches - 1;
  Value query = op.getInputs()[0];
  Value attentionMask = mask;
  RankedTensorType attentionType = resultType;
  auto splitHead = [&](ArrayRef<int64_t> shape, int64_t dim) {
    SmallVector<int64_t> split(shape);
    split[dim] /= groupSize;
    split.insert(split.begin() + dim + 1, groupSize);
    return split;
  };
  auto headReassociation = [&](int64_t splitRank, int64_t dim) {
    SmallVector<ReassociationIndices> reassociation;
    for (int64_t i = 0; i < splitRank; ++i) {
      if (i == dim + 1) {
        reassociation.back().push_back(i);
      } else {
        reassociation.push_back({i});
      }
    }
    return reassociation;
  };
  auto expandHead = [&](Value value, int64_t dim) -> Value {
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape = splitHead(type.getShape(), dim);
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, value);
    sizes[dim] = rewriter.getIndexAttr(shape[dim]);
    sizes.insert(sizes.begin() + dim + 1, rewriter.getIndexAttr(groupSize));
    return tensor::ExpandShapeOp::create(
        rewriter, loc, type.clone(shape), value,
        headReassociation(shape.size(), dim), sizes);
  };
  if (groupSize > 1) {
    query = expandHead(query, head);
    if (auto *kept = llvm::find(maskBatchDims, head);
        mask && kept != maskBatchDims.end()) {
      attentionMask = expandHead(mask, kept - maskBatchDims.begin());
    }
    attentionType = resultType.clone(splitHead(resultType.getShape(), head));
  }

  int64_t attentionRank = attentionType.getRank();
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < attentionRank - 1; ++i) {
    if (attentionType.isDynamicDim(i)) {
      dynamicSizes.push_back(tensor::DimOp::create(rewriter, loc, query, i));
    }
  }
  if (attentionType.isDynamicDim(attentionRank - 1)) {
    dynamicSizes.push_back(
        tensor::DimOp::create(rewriter, loc, op.getInputs()[2], rank - 1));
  }
  Value init = tensor::EmptyOp::create(rewriter, loc, attentionType.getShape(),
                                       elementType, dynamicSizes);
  SmallVector<AffineMap> maps =
      IREE::LinalgExt::getStandardAttentionIndexingMaps(
          rewriter.getContext(), numBatches, static_cast<bool>(mask),
          /*maskHasBatch=*/false);
  if (groupSize > 1) {
    // Over (batch..., h, g, l, n, d, s): query head h * groupSize + g reads key
    // and value head h.
    MLIRContext *context = rewriter.getContext();
    int64_t g = head + 1, row = head + 2, n = head + 3, d = head + 4,
            key = head + 5, domain = head + 6;
    auto map = [&](ArrayRef<int64_t> dims) {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0; i < head; ++i) {
        exprs.push_back(getAffineDimExpr(i, context));
      }
      for (int64_t dim : dims) {
        exprs.push_back(getAffineDimExpr(dim, context));
      }
      return AffineMap::get(domain, 0, exprs, context);
    };
    maps = {map({head, g, row, d}), map({head, key, d}), map({head, key, n}),
            AffineMap::get(domain, 0, {}, context)};
    if (mask) {
      SmallVector<AffineExpr> maskExprs;
      for (int64_t i : maskBatchDims) {
        maskExprs.push_back(getAffineDimExpr(i, context));
        if (i == head) {
          maskExprs.push_back(getAffineDimExpr(g, context));
        }
      }
      maskExprs.push_back(getAffineDimExpr(row, context));
      maskExprs.push_back(getAffineDimExpr(key, context));
      maps.push_back(AffineMap::get(domain, 0, maskExprs, context));
    }
    maps.push_back(map({head, g, row, n}));
  } else if (mask) {
    // The mask is indexed by its kept batch dims, the query and the key.
    SmallVector<AffineExpr> maskExprs;
    for (int64_t i : maskBatchDims) {
      maskExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    maskExprs.push_back(rewriter.getAffineDimExpr(numBatches));
    maskExprs.push_back(rewriter.getAffineDimExpr(numBatches + 3));
    maps[4] =
        AffineMap::get(numBatches + 4, 0, maskExprs, rewriter.getContext());
  }
  auto attention = IREE::LinalgExt::AttentionOp::create(
      rewriter, loc, attentionType, query, op.getInputs()[1], op.getInputs()[2],
      scale, init, rewriter.getAffineMapArrayAttr(maps),
      mask ? std::optional<Value>(attentionMask) : std::nullopt);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *block = rewriter.createBlock(&attention.getRegion());
    Value score = block->addArgument(rewriter.getF32Type(), loc);
    if (!mask && isCausal.getValue()) {
      // The group index precedes the query position in grouped scores.
      buildCausalScore(rewriter, loc, score,
                       groupSize > 1 ? numBatches + 1 : numBatches);
    } else {
      IREE::LinalgExt::YieldOp::create(rewriter, loc, score);
    }
  }
  Value result = attention.getResult(0);
  if (groupSize > 1) {
    result =
        tensor::CollapseShapeOp::create(rewriter, loc, resultType, result,
                                        headReassociation(attentionRank, head));
  }
  if (numInputs == 4 && !hasVisibleKeyInEveryRow(op.getInputs()[3])) {
    result = zeroHiddenRows(rewriter, loc, mask, result, maskBatchDims);
  }
  rewriter.replaceOp(op, result);
  return success();
}

} // namespace

LogicalResult populateTorchAttentionTargetRegistry(TargetRegistry &registry) {
  return registry.addComposite(
      "torch.scaled_dot_product_attention", 1, rewriteAttention,
      [](DialectRegistry &dialects) {
        dialects.insert<mlir::stablehlo::StablehloDialect, arith::ArithDialect,
                        linalg::LinalgDialect, tensor::TensorDialect,
                        IREE::LinalgExt::IREELinalgExtDialect>();
      });
}

} // namespace mlir::iree_compiler::stablehlo
