// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using transform_ext::StructuredOpMatcher;

namespace mlir::iree_compiler::IREE::Flow {

namespace {

//===----------------------------------------------------------------------===//
// Generic to Named Op Conversions
//===----------------------------------------------------------------------===//

// Method to match a transpose operation on the two most minor dimensions of the
// specified rank.
static bool matchInner2DTranspose(linalg::LinalgOp genericOp, unsigned rank) {
  // Only makes sense for minimum rank 2.
  if (rank < 2) {
    return false;
  }
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  // Check only for ops of the specified rank.
  if (genericOp.getNumLoops() != rank ||
      genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }
  // Check for transpose map.
  SmallVector<AffineExpr> exprList(rank);
  MLIRContext *context = genericOp.getContext();
  bindDimsList(context, MutableArrayRef{exprList});
  SmallVector<AffineExpr> transposeExprList(exprList);
  std::swap(transposeExprList[rank - 1], transposeExprList[rank - 2]);
  SmallVector<AffineMap> expectedMaps = {
      AffineMap::get(rank, 0, exprList, context),
      AffineMap::get(rank, 0, transposeExprList, context)};
  if (genericOp.getIndexingMapsArray() != expectedMaps) {
    return false;
  }

  Block *body = genericOp.getBlock();
  if (!llvm::hasSingleElement(*body)) {
    return false;
  }
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  auto blockArg = yieldOp.getOperand(0).dyn_cast<BlockArgument>();
  if (!blockArg || blockArg.getOwner() != body ||
      blockArg.getArgNumber() != 0) {
    return false;
  }
  return true;
}

// Method to match a linalg.matmul(a, linalg.transpose(b)). Returns `b` on
// success.
std::optional<Value> matchATransposeBMatmul(linalg::LinalgOp matmulOp) {
  if (!isa<linalg::MatmulOp>(matmulOp.getOperation())) {
    return std::nullopt;
  }
  auto rhs = matmulOp.getDpsInputOperand(1);
  auto genericOp = rhs->get().getDefiningOp<linalg::GenericOp>();
  if (genericOp && matchInner2DTranspose(genericOp, 2)) {
    return genericOp.getDpsInputOperand(0)->get();
  }
  return std::nullopt;
}

// Method to match a linalg.batch_matmul(a, linalg.transpose(b)). Returns `b` on
// success.
std::optional<Value> matchATransposeBBatchMatmul(linalg::LinalgOp bmmOp) {
  if (!isa<linalg::BatchMatmulOp>(bmmOp.getOperation())) {
    return std::nullopt;
  }
  auto rhs = bmmOp.getDpsInputOperand(1);
  auto genericOp = rhs->get().getDefiningOp<linalg::GenericOp>();
  if (genericOp && matchInner2DTranspose(genericOp, 3)) {
    return genericOp.getDpsInputOperand(0)->get();
  }
  return std::nullopt;
}

// Method to match a linalg.generic op representing a linalg.fill op. Returns
// the fill value (input operand to linalg.fill) on success.
std::optional<Value> matchGenericFill(linalg::LinalgOp linalgOp) {
  if (isa<linalg::GenericOp>(linalgOp.getOperation()) &&
      linalgOp.getNumDpsInputs() == 0 && linalgOp.getNumDpsInits() == 1 &&
      linalgOp.getNumParallelLoops() == linalgOp.getNumLoops() &&
      linalgOp.getIndexingMapsArray()[0].isIdentity()) {
    // Check that the op body is only a linalg.yield op.
    Value yieldOperand;
    for (Operation &bodyOp : linalgOp.getBlock()->getOperations()) {
      if (isa<linalg::YieldOp>(bodyOp)) {
        yieldOperand = bodyOp.getOperand(0);
      } else {
        return std::nullopt;
      }
    }
    // Check that the operand of the linalg.yield op is not an argument of the
    // linalg.generic basic block
    for (Value blockArg : linalgOp.getBlock()->getArguments()) {
      if (yieldOperand == blockArg) {
        return std::nullopt;
      }
    }
    return yieldOperand;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Slice Raising
//===----------------------------------------------------------------------===//

/// Matches a linalg.generic operation reading data from a tensor `source` using
/// tensor.extract, and raises the `source` tensor to an input of the linalg
/// operation.
static FailureOr<linalg::GenericOp>
raiseTensorExtractToInput(linalg::GenericOp linalgOp, RewriterBase &rewriter) {
  if (!linalgOp.hasTensorSemantics()) {
    return failure();
  }
  if (!isElementwise(linalgOp)) {
    return failure();
  }
  if (!llvm::hasSingleElement(linalgOp.getResults())) {
    return failure();
  }

  // Find a tensor.extract op in the linalgOp body.
  auto extractOps = linalgOp.getBody()->getOps<tensor::ExtractOp>();
  if (!llvm::hasSingleElement(extractOps)) {
    return failure();
  }
  tensor::ExtractOp extractOp = *extractOps.begin();
  auto resultType = dyn_cast<TensorType>(linalgOp.getResult(0).getType());
  if (!resultType) {
    return failure();
  }

  ArrayRef<int64_t> sourceShape = extractOp.getTensor().getType().getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();

  // Raise the tensor.extract op to an input.
  SmallVector<AffineExpr> exprs;
  for (auto [idx, indexValue] : llvm::enumerate(extractOp.getIndices())) {
    // For raising, the indexing value must be one of the following:
    //    1. A constant value.
    //    2. A linalg.index.

    // 1. Indexing value is a constant.
    APInt constantIndex;
    if (matchPattern(indexValue, m_ConstantInt(&constantIndex))) {
      // Restrict to cases where the constant is 0. This is because handling
      // constants other than 0 in indexing map, may cause problems in the
      // lowering pipeline later.
      if (constantIndex.getLimitedValue() != 0)
        return failure();
      exprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
      continue;
    }
    // 2. The indexing value is a linalg.index.
    if (auto indexOp = indexValue.getDefiningOp<linalg::IndexOp>()) {
      // Make sure that for this index, the size of the input and output
      // match and are not dynamic. We need this to maintain the op to be
      // elementwise.
      // TODO: This restriction can be relaxed by adding a extract_slice op
      // on the `source` tensor. This is not same as raising the whole
      // operation to an extract_slice, as there can be permutations and
      // projections involved.
      if (ShapedType::isDynamic(sourceShape[idx]) ||
          ShapedType::isDynamic(resultShape[indexOp.getDim()]) ||
          sourceShape[idx] != resultShape[indexOp.getDim()]) {
        return failure();
      }
      exprs.push_back(
          getAffineDimExpr(indexOp.getDim(), rewriter.getContext()));
      continue;
    }
    return failure();
  }
  AffineMap indexingMap = AffineMap::get(
      /*dimCount=*/linalgOp.getNumLoops(),
      /*symbolCount=*/0, exprs, rewriter.getContext());

  // Replace the linalgOp with a new linalgOp where the source tensor is
  // an input with the indexing map.
  SmallVector<Value> newInputs = linalgOp.getInputs();
  newInputs.insert(newInputs.begin(), extractOp.getTensor());
  SmallVector<Attribute> newIndexingMaps;
  newIndexingMaps.push_back(AffineMapAttr::get(indexingMap));
  for (AffineMap map : linalgOp.getIndexingMapsArray()) {
    newIndexingMaps.push_back(AffineMapAttr::get(map));
  }

  auto bodyBuilder = [&](OpBuilder &builder, Location loc, ValueRange args) {
    // Create an IR mapping from old block arguements to new ones.
    IRMapping mapper;
    ArrayRef<BlockArgument> oldArgs = linalgOp.getBody()->getArguments();
    // Map i^th old argument to (i + 1)^th new argument.
    for (unsigned i = 0; i < oldArgs.size(); ++i) {
      mapper.map(oldArgs[i], args[i + 1]);
    }
    // Clone the body of the linalgOp.
    for (Operation &op : linalgOp.getBody()->getOperations()) {
      // Replace the extractOp with the first block argument.
      if (&op == extractOp) {
        mapper.map(op.getResult(0), args[0]);
      } else {
        builder.clone(op, mapper);
      }
    }
  };

  linalg::GenericOp newLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
      linalgOp.getOutputs(),
      ArrayAttr::get(linalgOp->getContext(), newIndexingMaps),
      linalgOp.getIteratorTypesAttr(), linalgOp.getDocAttr(),
      linalgOp.getLibraryCallAttr(), bodyBuilder);

  return newLinalgOp;
}

/// Given a linalg.generic operation, and input/output tensors with their
/// indexing maps, tries to raise the operation to a tensor.extract_slice
/// operation. The tensor.extract_slice produced can be rank reducing.
static FailureOr<tensor::ExtractSliceOp>
tryRaiseToExtractSlice(AffineMap inputIndexingMap, AffineMap outputIndexingMap,
                       Value input, Value output, linalg::GenericOp linalgOp,
                       RewriterBase &rewriter) {
  // Output shape must be smaller than input shape.
  if (outputIndexingMap.getNumResults() >= inputIndexingMap.getNumResults()) {
    return failure();
  }
  // Output map should be identity.
  if (!outputIndexingMap.isIdentity()) {
    return failure();
  }

  auto outType = dyn_cast<RankedTensorType>(output.getType());
  if (!outType) {
    return failure();
  }
  ArrayRef<int64_t> outShape = outType.getShape();
  int64_t outRank = outShape.size();

  // Try to match each output dimension to an input dimension, in order.
  // If we find a constant access, we assume that dimension is supposed to be
  // rank reduced.
  // TODO: Support cases where the constant access matches the output dimension.
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  IntegerAttr zero = rewriter.getI64IntegerAttr(0);
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  unsigned currOutDim = 0;
  for (auto [idx, expr] : llvm::enumerate(inputIndexingMap.getResults())) {
    // Assume that the constant access is a rank reducing access.
    if (isa<AffineConstantExpr>(expr)) {
      IntegerAttr constIdx =
          rewriter.getI64IntegerAttr(cast<AffineConstantExpr>(expr).getValue());
      offsets.push_back(constIdx);
      sizes.push_back(one);
      continue;
    }
    // Check if the input dimension matches the current output dimension.
    if (currOutDim < outRank &&
        expr == outputIndexingMap.getResult(currOutDim)) {
      offsets.push_back(zero);
      // Get the dim size from the output tensor.
      if (ShapedType::isDynamic(outShape[currOutDim])) {
        auto dim = rewriter.create<tensor::DimOp>(linalgOp.getLoc(), output,
                                                  currOutDim);
        sizes.push_back(dim.getResult());
      } else {
        sizes.push_back(rewriter.getI64IntegerAttr(outShape[currOutDim]));
      }
      ++currOutDim;
      continue;
    }
    // Unknown access, fail.
    return failure();
  }

  // All output dimensions did not match an input dimension.
  if (currOutDim != outputIndexingMap.getNumResults()) {
    return failure();
  }

  // We only support dim expr or a constant expr on the input map, so strides
  // will always be 1.
  SmallVector<OpFoldResult> strides(inputIndexingMap.getNumResults(), one);

  return rewriter.create<tensor::ExtractSliceOp>(
      linalgOp.getLoc(), outType, input, offsets, sizes, strides);
}

/// Matches a linalg.generic operation with a single input and init output
/// tensor, and tries to raise it to a view-like operation on the input tensor.
static FailureOr<Operation *> tryRaiseToView(linalg::GenericOp linalgOp,
                                             RewriterBase &rewriter) {
  if (!linalgOp.hasTensorSemantics()) {
    return failure();
  }

  // Assume there is only 1 input, and 1 init tensor.
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1) {
    return failure();
  }
  OpOperand *inputOperand = linalgOp.getDpsInputOperand(0);
  OpOperand *outputOperand = linalgOp.getDpsInitOperand(0);

  // Check if linalg.yield yields a block arguement.
  auto yieldOp = dyn_cast<linalg::YieldOp>(linalgOp.getBody()->getTerminator());
  if (!yieldOp) {
    return failure();
  }
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  if (!blockArg) {
    return failure();
  }
  // Check if the block argument is an argument of the linalgOp.
  if (blockArg.getOwner() != linalgOp.getBody()) {
    return failure();
  }
  // Check that the block arguement corresponds to the input.
  if (blockArg.getArgNumber() != 0) {
    return failure();
  }

  Value input = inputOperand->get();
  Value output = outputOperand->get();
  AffineMap inputIndexingMap = linalgOp.getMatchingIndexingMap(inputOperand);
  AffineMap outputIndexingMap = linalgOp.getMatchingIndexingMap(outputOperand);

  // Try raising to tensor.collapse_shape.
  return tryRaiseToExtractSlice(inputIndexingMap, outputIndexingMap, input,
                                output, linalgOp, rewriter);
}

//===----------------------------------------------------------------------===//
// Partial negation and inner slice reverse
//===----------------------------------------------------------------------===//

static bool isUnaryNegate(linalg::GenericOp op) {
  auto block = op.getBlock();
  if (block->getNumArguments() != 2) {
    return false;
  }

  Operation *terminator = block->getTerminator();
  if (terminator->getNumOperands() != 1) {
    return false;
  }

  auto yielded = terminator->getOperand(0).getDefiningOp<arith::NegFOp>();
  if (!yielded) {
    return false;
  }

  if (yielded->getOperand(0) != block->getArgument(0)) {
    return false;
  }

  return true;
}

// This aims to match a partial negation and reverse of a tensor and rewrite
// it as a single linalg generic. For example, this will match sequences like
// the following:
//
//   %slice_0 = tensor.extract_slice %0[0, ..., 0] [..., 64] [1, ..., 1]
//                           : tensor<...x128xf16> to tensor<...x64xf16>
//   %slice_1 = tensor.extract_slice %0[0, ..., 64] [..., 64] [1, ..., 1]
//                            : tensor<...x128xf16> to tensor<...x64xf16>
//   %3 = linalg.generic {
//         indexing_maps = [affine_map<(d0, ..., dn-1) -> (d0, ..., dn-1)>,
//                          affine_map<(d0, ..., dn-1) -> (d0, ..., dn-1)>],
//         iterator_types = [n-1 x "parallel"]}
//      ins(%slice_1 : tensor<...x64xf16>) outs(%2 : tensor<...x64xf16>) {
//   ^bb0(%in: f16, %out: f16):
//     %5 = arith.negf %in : f16
//     linalg.yield %5 : f16
//   } -> tensor<...x64xf16>
//   %in_slice_0 = tensor.insert_slice %3 into
//                    %1[0, ..., 0] [..., 64] [1, ..., 1]
//                    : tensor<...x64xf16> into tensor<...x128xf16>
//   %in_slice_1 = tensor.insert_slice %slice_0 into
//                    %in_slice_0[0, ..., 64] [..., 64] [1, ..., 1]
//                    : tensor<...x64xf16> into tensor<...x128xf16>
//
// Where the input tensor is broken down along the inner most dimension, then
// the bottom elements are negated and the tensor is reconstructed, reversing
// the order of the slices. This is then rewritten as
//
//   %expanded = tensor.expand_shape %0 [[0], ..., [n-1, n]]
//                      : tensor<...x128xf16> into tensor<...x2x64xf16>
//   %2 = linalg.generic {
//         indexing_maps = [
//                     affine_map<(d0, ..., dn-1, dn) -> (d0, ..., dn-1, dn)>],
//         iterator_types = [n x "parallel"]
//      } outs(%output : tensor<...x2x64xf16>) {
//   ^bb0(%out: f16):
//     %i0 = linalg.index 0 : index
//     ...
//     %in-1 = linalg.index n-1 : index
//     %in   = linalg.index n : index
//     %rev = arith.subi %c1, %in-1 : index
//     %extracted = tensor.extract %expanded[%i0, ..., %rev, %in]
//                                         : tensor<...x2x64xf16>
//     %neg = arith.negf %extracted : f16
//     %cmp = arith.cmpi eq, %in-1, %c1 : index
//     %sel = arith.select %cmp, %neg, %extracted : f16
//     linalg.yield %sel : f16
//   } -> tensor<...x2x64xf16>
//   %collapsed = tensor.collapse_shape %2 [[0], ..., [n-1, n]]
//              : tensor<...x2x64xf16> into tensor<...x128xf16>
static std::optional<Value>
matchCatNegateAndSlice(tensor::InsertSliceOp insertOp) {
  /// First match against the desired op chain.
  auto topHalf = insertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!topHalf) {
    return std::nullopt;
  }
  auto dest = insertOp.getDest().getDefiningOp<tensor::InsertSliceOp>();
  if (!dest) {
    return std::nullopt;
  }

  /// TODO: This could be extended to other unary (or in general, elementwise
  /// operations) if the need arises.
  auto negate = dest.getSource().getDefiningOp<linalg::GenericOp>();
  if (!negate || !isUnaryNegate(negate)) {
    return std::nullopt;
  }

  auto bottomHalf =
      negate.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>();
  if (!bottomHalf) {
    return std::nullopt;
  }
  Value source = topHalf.getSource();
  if (source != bottomHalf.getSource()) {
    return std::nullopt;
  }

  /// Require that the overall operation isn't changing the tensor shape.
  if (source.getType() != dest.getType()) {
    return std::nullopt;
  }

  auto sourceType = source.getType().dyn_cast<RankedTensorType>();
  if (!sourceType || sourceType.getRank() == 0) {
    return std::nullopt;
  }

  /// Here we are verifying that the slice sequence inserts and extracts in the
  /// same order. In other words, the offsets, sizes, and strides for the nth
  /// extract slice match the nth insert slice.
  /// TODO: We could relax this condition if we see cases where the elementwise
  /// unary operation happens on a different slices of the tensor, but for now
  /// don't engineer for cases that don't exist.
  auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
  if (!insertOp.isSameAs(bottomHalf, isSame) ||
      !dest.isSameAs(topHalf, isSame)) {
    return std::nullopt;
  }

  /// All slices need unit stride and the initial slice should have a zero
  /// offset.
  if (!insertOp.hasUnitStride() || !dest.hasUnitStride() ||
      !dest.hasZeroOffset()) {
    return std::nullopt;
  }

  /// Verify all slice sizes are the same.
  for (auto [bottom, top] :
       llvm::zip_equal(insertOp.getMixedSizes(), dest.getMixedSizes())) {
    if (bottom != top) {
      return std::nullopt;
    }
  }

  /// Verify that the final InsertSlice has zero offsets for all except the
  /// inner most dim.
  SmallVector<OpFoldResult> insertOffsets = insertOp.getMixedOffsets();
  for (int i = 0, e = insertOffsets.size() - 1; i < e; ++i) {
    if (!isConstantIntValue(insertOffsets[i], 0)) {
      return std::nullopt;
    }
  }

  /// Finally verify that the inner most offset == the inner most slice size,
  /// and because we verified all slice sizes are the same, this necessarily
  /// is slicing exactly half of the inner most dimension.
  if (insertOffsets.back() != insertOp.getMixedSizes().back()) {
    return std::nullopt;
  }

  return source;
}

// This aims to match the exact same logical operation, but instead where the
// final concatenation is actually represented as a concatenation.
//
//   %slice_0 = tensor.extract_slice %0[0, ..., 0] [..., 64] [1, ..., 1]
//                           : tensor<...x128xf16> to tensor<...x64xf16>
//   %slice_1 = tensor.extract_slice %0[0, ..., 64] [..., 64] [1, ..., 1]
//                            : tensor<...x128xf16> to tensor<...x64xf16>
//   %3 = linalg.generic {
//         indexing_maps = [affine_map<(d0, ..., dn-1) -> (d0, ..., dn-1)>,
//                          affine_map<(d0, ..., dn-1) -> (d0, ..., dn-1)>],
//         iterator_types = [n-1 x "parallel"]}
//      ins(%slice_1 : tensor<...x64xf16>) outs(%2 : tensor<...x64xf16>) {
//   ^bb0(%in: f16, %out: f16):
//     %5 = arith.negf %in : f16
//     linalg.yield %5 : f16
//   } -> tensor<...x64xf16>
//   %in_slice_0 = tensor.concat %3, %slice_0 : ... -> tensor<...x128xf16>
//
// With the exact same rewritten IR.
static std::optional<Value> matchCatNegateAndSlice(tensor::ConcatOp concatOp) {
  /// First match against the desired op chain.
  ValueRange inputs = concatOp.getInputs();
  if (inputs.size() != 2) {
    return std::nullopt;
  }

  Value left = inputs[0];
  Value right = inputs[1];
  if (left.getType() != right.getType()) {
    return std::nullopt;
  }

  // The concatenation must happen along the inner most dimension.
  auto inputType = cast<RankedTensorType>(left.getType());
  if (concatOp.getDim() != inputType.getRank() - 1) {
    return std::nullopt;
  }

  /// TODO: This could be extended to other unary (or in general, elementwise
  /// operations) if the need arises.
  auto negate = left.getDefiningOp<linalg::GenericOp>();
  if (!negate || !isUnaryNegate(negate)) {
    return std::nullopt;
  }

  auto topHalf = right.getDefiningOp<tensor::ExtractSliceOp>();
  if (!topHalf) {
    return std::nullopt;
  }

  auto bottomHalf =
      negate.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>();
  if (!bottomHalf) {
    return std::nullopt;
  }
  Value source = topHalf.getSource();
  if (source != bottomHalf.getSource()) {
    return std::nullopt;
  }

  /// Require that the overall operation isn't changing the tensor shape.
  if (source.getType() != concatOp.getType()) {
    return std::nullopt;
  }
  return source;
}

static Value createCatNegateAndSlice(RewriterBase &rewriter, Value outTensor,
                                     Value source) {
  Location loc = source.getLoc();
  /// The matcher checks that this cast is valid.
  auto sourceType = cast<RankedTensorType>(source.getType());

  SmallVector<int64_t> targetShape(sourceType.getShape());
  SmallVector<ReassociationIndices> reassoc;
  for (int i = 0, e = targetShape.size(); i < e; i++) {
    reassoc.push_back(ReassociationIndices{i});
  }
  reassoc.back().push_back(targetShape.size());

  /// Note that because we've verified the pair of slices is exactly half
  /// of the whole inner most extent, the sliceSize is necessarily
  /// divisible by 2.
  int64_t sliceSize = targetShape.back();
  targetShape[targetShape.size() - 1] = 2;
  targetShape.push_back(ShapedType::isDynamic(sliceSize) ? sliceSize
                                                         : sliceSize / 2);
  Type expandedType =
      RankedTensorType::get(targetShape, sourceType.getElementType());
  Value expanded = rewriter.create<tensor::ExpandShapeOp>(loc, expandedType,
                                                          source, reassoc);

  Value expandedOutTensor = rewriter.create<tensor::ExpandShapeOp>(
      loc, expandedType, outTensor, reassoc);

  SmallVector<AffineMap> indexingMaps = {
      rewriter.getMultiDimIdentityMap(targetShape.size())};
  SmallVector<utils::IteratorType> iteratorTypes(targetShape.size(),
                                                 utils::IteratorType::parallel);

  auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange args) {
    SmallVector<Value> extractionIndices;
    for (size_t i = 0, e = targetShape.size(); i < e; ++i) {
      extractionIndices.push_back(b.create<linalg::IndexOp>(loc, i));
    }

    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    // Take the reverse of the second to last iterator. Because we statically
    // guaranteed it to be 2 it just becomes `1 - iters[-2]`.
    Value reverseSplitIdx = rewriter.create<arith::SubIOp>(
        loc, c1, extractionIndices[targetShape.size() - 2]);
    extractionIndices[targetShape.size() - 2] = reverseSplitIdx;

    // Extract the value from input tensor and negate the top half of the result
    // slice (lower half of the input slice).
    Value inputVal =
        b.create<tensor::ExtractOp>(loc, expanded, extractionIndices);
    Value maybeNegate = b.create<arith::NegFOp>(loc, inputVal);

    Value isEqual = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            reverseSplitIdx, c1);
    Value select =
        rewriter.create<arith::SelectOp>(loc, isEqual, maybeNegate, inputVal);
    b.create<linalg::YieldOp>(loc, select);
  };

  Value result =
      rewriter
          .create<linalg::GenericOp>(loc, expandedOutTensor.getType(),
                                     ValueRange(), expandedOutTensor,
                                     indexingMaps, iteratorTypes, bodyBuilder)
          .getResult(0);

  return rewriter.create<tensor::CollapseShapeOp>(loc, outTensor.getType(),
                                                  result, reassoc);
}

static Value rewriteCatNegateAndSlice(RewriterBase &rewriter,
                                      tensor::InsertSliceOp sliceOp,
                                      Value source) {
  rewriter.setInsertionPoint(sliceOp);
  Value outTensor =
      sliceOp.getDest().getDefiningOp<tensor::InsertSliceOp>().getDest();
  return createCatNegateAndSlice(rewriter, outTensor, source);
}

static Value rewriteCatNegateAndSlice(RewriterBase &rewriter,
                                      tensor::ConcatOp concatOp, Value source) {
  rewriter.setInsertionPoint(concatOp);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value outTensor = rewriter.create<tensor::EmptyOp>(
      source.getLoc(), tensor::getMixedSizes(rewriter, source.getLoc(), source),
      elemType);
  return createCatNegateAndSlice(rewriter, outTensor, source);
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RaiseSpecialOpsPass : public RaiseSpecialOpsBase<RaiseSpecialOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    getOperation()->walk([&](linalg::GenericOp op) {
      linalg::GenericOp linalgOp = op;

      OpBuilder::InsertionGuard guard(rewriter);

      // Try raising to tensor.export and create an intermediate linalg.generic.
      rewriter.setInsertionPoint(op);
      FailureOr<linalg::GenericOp> maybeNewOp =
          raiseTensorExtractToInput(linalgOp, rewriter);
      if (succeeded(maybeNewOp)) {
        linalgOp = *maybeNewOp;
      }

      // Try raising to a view-like operation. Replace if the op raising was
      // successful.
      rewriter.setInsertionPoint(op);
      FailureOr<Operation *> maybeRaisedView =
          tryRaiseToView(linalgOp, rewriter);
      if (succeeded(maybeRaisedView)) {
        rewriter.replaceOp(op, *maybeRaisedView);
      }
    });

    SmallVector<std::pair<linalg::LinalgOp, Value>> softmaxRoots;
    SmallVector<std::pair<linalg::MatmulOp, Value>> transposeMatmulRoots;
    SmallVector<std::pair<linalg::BatchMatmulOp, Value>>
        transposeBatchMatmulRoots;
    SmallVector<std::pair<linalg::GenericOp, Value>> genericFills;
    getOperation()->walk([&](linalg::LinalgOp op) {
      {
        transform_ext::MatcherContext matcherContext;
        transform_ext::StructuredOpMatcher *maxReduction;
        transform_ext::StructuredOpMatcher *softmaxroot;
        makeSoftmaxMatcher(matcherContext, maxReduction, softmaxroot);
        if (matchPattern(op, *softmaxroot)) {
          Value src = maxReduction->getCaptured()->getOperand(0);
          softmaxRoots.push_back(std::make_pair(op, src));
        }
        if (std::optional<Value> newRhs = matchATransposeBMatmul(op)) {
          transposeMatmulRoots.push_back(std::make_pair(
              cast<linalg::MatmulOp>(op.getOperation()), newRhs.value()));
        }
        if (std::optional<Value> newRhs = matchATransposeBBatchMatmul(op)) {
          transposeBatchMatmulRoots.push_back(std::make_pair(
              cast<linalg::BatchMatmulOp>(op.getOperation()), newRhs.value()));
        }
        if (std::optional<Value> fillInput = matchGenericFill(op)) {
          genericFills.push_back(
              std::make_pair(cast<linalg::GenericOp>(op), fillInput.value()));
        }
      }
    });

    SmallVector<std::pair<tensor::InsertSliceOp, Value>>
        catAsInsertNegateAndSliceRoots;
    getOperation()->walk([&](tensor::InsertSliceOp op) {
      if (std::optional<Value> catAsInsertNegateAndSliceRoot =
              matchCatNegateAndSlice(op)) {
        catAsInsertNegateAndSliceRoots.push_back(
            std::make_pair(op, catAsInsertNegateAndSliceRoot.value()));
      }
    });
    SmallVector<std::pair<tensor::ConcatOp, Value>> catNegateAndSliceRoots;
    getOperation()->walk([&](tensor::ConcatOp op) {
      if (std::optional<Value> catNegateAndSliceRoot =
              matchCatNegateAndSlice(op)) {
        catNegateAndSliceRoots.push_back(
            std::make_pair(op, catNegateAndSliceRoot.value()));
      }
    });

    for (auto [softmaxOp, src] : softmaxRoots) {
      rewriter.setInsertionPoint(softmaxOp);
      rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
          softmaxOp, softmaxOp->getResultTypes(), src,
          softmaxOp.getDpsInitOperand(0)->get(), softmaxOp.getNumLoops() - 1);
    }

    for (auto [matmulOp, newRhs] : transposeMatmulRoots) {
      Value lhs = matmulOp.getDpsInputOperand(0)->get();
      Value init = matmulOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(matmulOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(matmulOp);
      rewriter.replaceOpWithNewOp<linalg::MatmulTransposeBOp>(
          matmulOp, ValueRange{lhs, newRhs}, ValueRange{init}, attrs);
    }
    for (auto [bmmOp, newRhs] : transposeBatchMatmulRoots) {
      Value lhs = bmmOp.getDpsInputOperand(0)->get();
      Value init = bmmOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(bmmOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(bmmOp);
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulTransposeBOp>(
          bmmOp, ValueRange{lhs, newRhs}, ValueRange{init}, attrs);
    }
    for (auto [genericOp, fillInput] : genericFills) {
      Value init = genericOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(genericOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(genericOp);
      rewriter.replaceOpWithNewOp<linalg::FillOp>(
          genericOp, ValueRange{fillInput}, ValueRange{init}, attrs);
    }
    for (auto [sliceOp, input] : catAsInsertNegateAndSliceRoots) {
      Value res = rewriteCatNegateAndSlice(rewriter, sliceOp, input);
      rewriter.replaceOp(sliceOp, res);
    }
    for (auto [concatOp, input] : catNegateAndSliceRoots) {
      Value res = rewriteCatNegateAndSlice(rewriter, concatOp, input);
      rewriter.replaceOp(concatOp, res);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseSpecialOps() {
  return std::make_unique<RaiseSpecialOpsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
