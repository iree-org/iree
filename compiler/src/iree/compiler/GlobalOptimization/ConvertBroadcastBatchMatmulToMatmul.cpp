// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTBROADCASTBATCHMATMULTOMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

/// Check if an operation broadcasts only on the batch dimension (dim 0).
/// Handles both linalg.broadcast and linalg.generic (after generalization).
static bool isBatchDimBroadcast(Operation *op) {
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    ArrayRef<int64_t> dims = broadcastOp.getDimensions();
    return dims.size() == 1 && dims[0] == 0;
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return false;
  }

  // Check generic is a broadcast: 1 input, 1 output, all parallel iterators
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }

  auto iterTypes = genericOp.getIteratorTypesArray();
  if (!llvm::all_of(iterTypes, [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      })) {
    return false;
  }

  SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
  AffineMap inputMap = maps[0];
  AffineMap outputMap = maps[1];

  unsigned numDims = outputMap.getNumDims();
  if (!outputMap.isIdentity()) {
    return false;
  }

  if (!inputMap.isProjectedPermutation()) {
    return false;
  }

  // Check that the input map is exactly (d0, d1, ..., dN) -> (d1, d2, ..., dN)
  // i.e., only dimension 0 is broadcast (missing) and the remaining dimensions
  // are in order (not permuted). A permuted map like (d0,d1,d2) -> (d2,d1)
  // represents a fused broadcast+transpose, which this pattern cannot handle.
  if (inputMap.getNumResults() != numDims - 1) {
    return false;
  }
  for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inputMap.getResult(i));
    if (!dimExpr || dimExpr.getPosition() != i + 1) {
      return false;
    }
  }

  // Check body simply yields the input argument
  Block *body = genericOp.getBody();
  if (body->getNumArguments() != 2) {
    return false;
  }
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1) {
    return false;
  }
  if (yieldOp.getOperand(0) != body->getArgument(0)) {
    return false;
  }

  return true;
}

static Value getOriginalWeight(Operation *op) {
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    return broadcastOp.getInput();
  }
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    return genericOp.getDpsInputs()[0];
  }
  return nullptr;
}

static int64_t computeCollapsedDim(int64_t dim0, int64_t dim1) {
  if (dim0 == ShapedType::kDynamic || dim1 == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return dim0 * dim1;
}

static Value createCollapseShape(OpBuilder &builder, Location loc, Value input,
                                 RankedTensorType inputType) {
  SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}};
  int64_t collapsedDim0 =
      computeCollapsedDim(inputType.getDimSize(0), inputType.getDimSize(1));
  auto collapsedType = RankedTensorType::get(
      {collapsedDim0, inputType.getDimSize(2)}, inputType.getElementType());
  return tensor::CollapseShapeOp::create(builder, loc, collapsedType, input,
                                         reassoc);
}

static Value createExpandShape(OpBuilder &builder, Location loc, Value input,
                               RankedTensorType outputType, Value originalAct,
                               Value originalOut) {
  SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}};

  // Build output shape - extract dynamic dimensions from original tensors
  SmallVector<OpFoldResult> outputShape;
  for (int64_t i = 0; i < 3; ++i) {
    if (outputType.isDynamicDim(i)) {
      // For batch (i=0) and M (i=1), get from original activation
      // For N (i=2), get from original output
      Value source = (i < 2) ? originalAct : originalOut;
      Value idx = arith::ConstantIndexOp::create(builder, loc, i);
      Value dimVal = tensor::DimOp::create(builder, loc, source, idx);
      outputShape.push_back(dimVal);
    } else {
      outputShape.push_back(builder.getIndexAttr(outputType.getDimSize(i)));
    }
  }

  return tensor::ExpandShapeOp::create(builder, loc, outputType, input, reassoc,
                                       outputShape);
}

/// Pattern to convert broadcast + batch_matmul to collapse_shape + matmul +
/// expand_shape. This handles BatchMatmulOp and BatchMatmulTransposeBOp
/// (which share the same underlying op but with different indexing maps).
/// BatchMatmulTransposeAOp is not supported because the collapse would
/// produce mismatched shapes (LHS collapses batch*K but output collapses
/// batch*M).
class ConvertBroadcastBatchMatmulToMatmul
    : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp batchMatmul,
                                PatternRewriter &rewriter) const override {
    // Check RHS comes from broadcast
    Value rhs = batchMatmul.getDpsInputOperand(1)->get();
    Operation *rhsDefOp = rhs.getDefiningOp();
    if (!rhsDefOp) {
      return failure();
    }

    if (!isBatchDimBroadcast(rhsDefOp)) {
      return failure();
    }

    Value weight = getOriginalWeight(rhsDefOp);
    if (!weight) {
      return failure();
    }

    // Get LHS activation
    Value act = batchMatmul.getDpsInputOperand(0)->get();
    auto actType = dyn_cast<RankedTensorType>(act.getType());
    if (!actType) {
      return failure();
    }

    // Get result type
    Value out = batchMatmul.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<RankedTensorType>(out.getType());
    if (!outType) {
      return failure();
    }

    // Skip transpose_a variant - the collapse would produce mismatched shapes:
    // LHS [batch, K, M] collapses to [batch*K, M] but
    // Out [batch, M, N] collapses to [batch*M, N]
    // These don't match, so the transformation is invalid.
    if (isa<linalg::BatchMatmulTransposeAOp>(batchMatmul.getOperation())) {
      return failure();
    }

    Location loc = batchMatmul.getLoc();

    // Collapse activation: [batch, dim1, dim2] -> [batch*dim1, dim2]
    Value collapsedAct = createCollapseShape(rewriter, loc, act, actType);

    // Collapse output init tensor
    Value collapsedOut = createCollapseShape(rewriter, loc, out, outType);

    // Create matmul - check if this is a transpose variant
    auto collapsedOutType = cast<RankedTensorType>(collapsedOut.getType());
    Value matmulResult;

    if (isa<linalg::BatchMatmulTransposeBOp>(batchMatmul.getOperation())) {
      matmulResult = linalg::MatmulTransposeBOp::create(
                         rewriter, loc, collapsedOutType,
                         ValueRange{collapsedAct, weight}, collapsedOut)
                         .getResult(0);
    } else {
      matmulResult = linalg::MatmulOp::create(rewriter, loc, collapsedOutType,
                                              ValueRange{collapsedAct, weight},
                                              collapsedOut)
                         .getResult(0);
    }

    // Expand result back to 3D: [batch*dim1, dim2] -> [batch, dim1, dim2]
    Value expandedResult =
        createExpandShape(rewriter, loc, matmulResult, outType, act, out);

    rewriter.replaceOp(batchMatmul, expandedResult);
    return success();
  }
};

struct ConvertBroadcastBatchMatmulToMatmulPass
    : public impl::ConvertBroadcastBatchMatmulToMatmulPassBase<
          ConvertBroadcastBatchMatmulToMatmulPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertBroadcastBatchMatmulToMatmul>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
