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

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

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
      if (sourceShape[idx] == ShapedType::kDynamic ||
          resultShape[indexOp.getDim()] == ShapedType::kDynamic ||
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
    if (expr.isa<AffineConstantExpr>()) {
      IntegerAttr constIdx = rewriter.getI64IntegerAttr(
          expr.cast<AffineConstantExpr>().getValue());
      offsets.push_back(constIdx);
      sizes.push_back(one);
      continue;
    }
    // Check if the input dimension matches the current output dimension.
    if (currOutDim < outRank &&
        expr == outputIndexingMap.getResult(currOutDim)) {
      offsets.push_back(zero);
      // Get the dim size from the output tensor.
      if (outShape[currOutDim] == ShapedType::kDynamic) {
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

    for (std::pair<linalg::LinalgOp, Value> softmax : softmaxRoots) {
      linalg::LinalgOp op = softmax.first;
      Value src = softmax.second;
      rewriter.setInsertionPoint(softmax.first);
      rewriter.replaceOpWithNewOp<IREE::LinalgExt::SoftmaxOp>(
          op, src, op.getDpsInitOperand(0)->get(), op.getNumLoops() - 1);
    }

    for (std::pair<linalg::MatmulOp, Value> aTransposeBMatmul :
         transposeMatmulRoots) {
      auto matmulOp = aTransposeBMatmul.first;
      Value lhs = matmulOp.getDpsInputOperand(0)->get();
      auto newRhs = aTransposeBMatmul.second;
      Value init = matmulOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(matmulOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(matmulOp);
      rewriter.replaceOpWithNewOp<linalg::MatmulTransposeBOp>(
          matmulOp, ValueRange{lhs, newRhs}, ValueRange{init}, attrs);
    }
    for (std::pair<linalg::BatchMatmulOp, Value> aTransposeBBatchMatmul :
         transposeBatchMatmulRoots) {
      auto bmmOp = aTransposeBBatchMatmul.first;
      Value lhs = bmmOp.getDpsInputOperand(0)->get();
      auto newRhs = aTransposeBBatchMatmul.second;
      Value init = bmmOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(bmmOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(bmmOp);
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulTransposeBOp>(
          bmmOp, ValueRange{lhs, newRhs}, ValueRange{init}, attrs);
    }
    for (std::pair<linalg::GenericOp, Value> genericFill : genericFills) {
      auto genericOp = genericFill.first;
      Value fillInput = genericFill.second;
      Value init = genericOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(genericOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(genericOp);
      rewriter.replaceOpWithNewOp<linalg::FillOp>(
          genericOp, ValueRange{fillInput}, ValueRange{init}, attrs);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseSpecialOps() {
  return std::make_unique<RaiseSpecialOpsPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
