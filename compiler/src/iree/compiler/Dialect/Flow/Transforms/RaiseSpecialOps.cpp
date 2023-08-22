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

// Method to match a transpose operation.
static bool match2DTranspose(linalg::LinalgOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  // Check only for 2D ops.
  if (genericOp.getNumLoops() != 2 ||
      genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }
  // Check for transpose map.
  AffineExpr d0, d1;
  MLIRContext *context = genericOp.getContext();
  bindDims(context, d0, d1);
  SmallVector<AffineMap> expectedMaps = {
      AffineMap::get(2, 0, {d0, d1}, context),
      AffineMap::get(2, 0, {d1, d0}, context)};
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
  if (genericOp && match2DTranspose(genericOp)) {
    return genericOp.getDpsInputOperand(0)->get();
  }
  return std::nullopt;
}

/// Matches a linalg.generic operation reading data from a tensor `source` using
/// tensor.extract, and raises the `source` tensor to an input of the linalg
/// operation.
static LogicalResult raiseTensorExtractToInput(linalg::GenericOp linalgOp,
                                               RewriterBase &rewriter) {
  if (!linalgOp.hasTensorSemantics())
    return failure();
  if (!isElementwise(linalgOp))
    return failure();

  // Find a tensor.extract op in the linalgOp body.
  tensor::ExtractOp extractOp;
  linalgOp.walk([&](tensor::ExtractOp op) {
    if (extractOp)
      return WalkResult::interrupt();
    extractOp = op;
    return WalkResult::advance();
  });

  if (!extractOp)
    return failure();

  // Raise the tensor.extract op to an input.
  SmallVector<AffineExpr> exprs;
  for (Value indexValue : extractOp.getIndices()) {
    // For raising, the indexing value must be one of the following:
    //    1. A constant value.
    //    2. A linalg.index.

    // 1. Indexing value is a constant.
    APInt constantIndex;
    if (matchPattern(indexValue, m_ConstantInt(&constantIndex))) {
      exprs.push_back(getAffineConstantExpr(constantIndex.getLimitedValue(),
                                            rewriter.getContext()));
      continue;
    }
    // 2. The indexing value is a linalg.index.
    if (auto indexOp = indexValue.getDefiningOp<linalg::IndexOp>()) {
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
  for (AffineMap map : linalgOp.getIndexingMapsArray())
    newIndexingMaps.push_back(AffineMapAttr::get(map));

  auto bodyBuilder = [&](OpBuilder &builder, Location loc, ValueRange args) {
    // Create an IR mapping from old block arguements to new ones.
    IRMapping mapper;
    ArrayRef<BlockArgument> oldArgs = linalgOp.getBody()->getArguments();
    // Map i^th old argument to (i + 1)^th new argument.
    for (unsigned i = 0; i < oldArgs.size(); ++i)
      mapper.map(oldArgs[i], args[i + 1]);
    // Clone the body of the linalgOp.
    for (Operation &op : linalgOp.getBody()->getOperations()) {
      // Replace the extractOp with the first block argument.
      if (&op == extractOp)
        mapper.map(op.getResult(0), args[0]);
      else
        builder.clone(op, mapper);
    }
  };

  linalg::GenericOp newLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
      linalgOp.getOutputs(),
      ArrayAttr::get(linalgOp->getContext(), newIndexingMaps),
      linalgOp.getIteratorTypesAttr(), linalgOp.getDocAttr(),
      linalgOp.getLibraryCallAttr(), bodyBuilder);

  rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

  return success();
}

/// Given a linalg.generic operation, and input/output tensors with their
/// indexing maps, tries to raise the operation to a tensor.extract_slice
/// operation. The tensor.extract_slice produced can be rank reducing.
static LogicalResult tryRaiseToExtractSlice(AffineMap inputIndexingMap,
                                            AffineMap outputIndexingMap,
                                            Value input, Value output,
                                            linalg::GenericOp linalgOp,
                                            RewriterBase &rewriter) {
  // Output shape must be smaller than input shape.
  if (outputIndexingMap.getNumResults() >= inputIndexingMap.getNumResults())
    return failure();
  // Output map should be identity.
  if (!outputIndexingMap.isIdentity())
    return failure();

  auto outType = dyn_cast<RankedTensorType>(output.getType());
  if (!outType)
    return failure();
  ArrayRef<int64_t> outShape = outType.getShape();

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
    // Check if the input dimension matches the current output dimension.
    if (expr == outputIndexingMap.getResult(currOutDim)) {
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
    // Assume that the constant access is a rank reducing access.
    if (expr.isa<AffineConstantExpr>() &&
        expr.cast<AffineConstantExpr>().getValue() == 0) {
      offsets.push_back(zero);
      sizes.push_back(one);
      continue;
    }
    // Unknown access, fail.
    return failure();
  }

  // All output dimensions did not match an input dimension.
  if (currOutDim != outputIndexingMap.getNumResults())
    return failure();

  // We only support dim expr or a constant expr on the input map, so strides
  // will always be 1.
  SmallVector<OpFoldResult> strides(inputIndexingMap.getNumResults(), one);

  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(linalgOp, outType, input,
                                                      offsets, sizes, strides);
  return success();
}

/// Matches a linalg.generic operation with a single input and init output
/// tensor, and tries to raise it to a view-like operation on the input tensor.
static LogicalResult tryRaiseToView(linalg::GenericOp linalgOp,
                                    RewriterBase &rewriter) {
  if (!linalgOp.hasTensorSemantics())
    return failure();

  // Assume there is only 1 input, and 1 init tensor.
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return failure();
  OpOperand *inputOperand = linalgOp.getDpsInputOperand(0);
  OpOperand *outputOperand = linalgOp.getDpsInitOperand(0);

  // Check if linalg.yield yields a block arguement.
  auto yieldOp = dyn_cast<linalg::YieldOp>(linalgOp.getBody()->getTerminator());
  if (!yieldOp)
    return failure();
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  if (!blockArg)
    return failure();
  // Check if the block argument is an argument of the linalgOp.
  if (blockArg.getOwner() != linalgOp.getBody())
    return failure();
  // Check that the block arguement corresponds to the input.
  if (blockArg.getArgNumber() != 0)
    return failure();

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
    SmallVector<std::pair<linalg::LinalgOp, Value>> softmaxRoots;
    SmallVector<std::pair<linalg::MatmulOp, Value>> transposeMatmulRoots;
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
      }
    });

    IRRewriter rewriter(&getContext());

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

    // Raise tensor.export.
    getOperation()->walk([&](linalg::GenericOp op) {
      rewriter.setInsertionPoint(op);
      (void)raiseTensorExtractToInput(op, rewriter);
    });

    // Raise linalg.generic view-like ops.
    getOperation()->walk([&](linalg::GenericOp op) {
      rewriter.setInsertionPoint(op);
      (void)tryRaiseToView(op, rewriter);
    });
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
