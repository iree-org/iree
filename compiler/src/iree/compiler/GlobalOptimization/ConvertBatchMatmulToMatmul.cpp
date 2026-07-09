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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTBATCHMATMULTOMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

/// If `op` broadcasts a weight along the batch dimension (dim 0) only, returns
/// the original pre-broadcast weight; otherwise returns failure. Handles both
/// linalg.broadcast and its generic form (after generalization).
static FailureOr<Value> getOriginalBroadcastWeight(Operation *op) {
  if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
    ArrayRef<int64_t> dims = broadcastOp.getDimensions();
    if (dims.size() != 1 || dims[0] != 0) {
      return failure();
    }
    return broadcastOp.getInput();
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) {
    return failure();
  }

  // Check generic is a broadcast: 1 input, 1 output, all parallel iterators
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return failure();
  }

  auto iterTypes = genericOp.getIteratorTypesArray();
  if (!llvm::all_of(iterTypes, [](utils::IteratorType t) {
        return t == utils::IteratorType::parallel;
      })) {
    return failure();
  }

  SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
  AffineMap inputMap = maps[0];
  AffineMap outputMap = maps[1];

  unsigned numDims = outputMap.getNumDims();
  if (!outputMap.isIdentity()) {
    return failure();
  }

  if (!inputMap.isProjectedPermutation()) {
    return failure();
  }

  // Check that the input map is exactly (d0, d1, ..., dN) -> (d1, d2, ..., dN)
  // i.e., only dimension 0 is broadcast (missing in the result affine expr) and
  // the remaining dimensions are in order (not permuted). A permuted map like
  // (d0,d1,d2) -> (d2,d1) represents a fused broadcast+transpose, which this
  // pattern cannot handle.
  if (inputMap.getNumResults() != numDims - 1) {
    return failure();
  }
  for (unsigned i = 0; i < inputMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inputMap.getResult(i));
    if (!dimExpr || dimExpr.getPosition() != i + 1) {
      return failure();
    }
  }

  // Check body simply yields the input argument
  Block *body = genericOp.getBody();
  if (body->getNumArguments() != 2) {
    return failure();
  }
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1) {
    return failure();
  }
  if (yieldOp.getOperand(0) != body->getArgument(0)) {
    return failure();
  }

  return genericOp.getDpsInputs()[0];
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
    // The RHS must be a batch-dim broadcast of a 2-D weight.
    Value rhs = batchMatmul.getDpsInputOperand(1)->get();
    Operation *rhsDefOp = rhs.getDefiningOp();
    if (!rhsDefOp) {
      return failure();
    }
    FailureOr<Value> weight = getOriginalBroadcastWeight(rhsDefOp);
    if (failed(weight)) {
      return failure();
    }

    // Get LHS activation
    Value act = batchMatmul.getDpsInputOperand(0)->get();
    auto actType = cast<RankedTensorType>(act.getType());

    Value out = batchMatmul.getDpsInitOperand(0)->get();
    auto outType = cast<RankedTensorType>(out.getType());

    // Classify the contraction layout by comparing the indexing maps directly.
    // batch_matmul has four loop dimensions (batch, m, n, k); the canonical
    // maps are LHS (batch, m, k), RHS (batch, k, n) and OUT (batch, m, n).
    MLIRContext *ctx = batchMatmul.getContext();
    AffineExpr bDim, mDim, nDim, kDim;
    bindDims(ctx, bDim, mDim, nDim, kDim);
    auto mapOf = [&](ArrayRef<AffineExpr> results) {
      return AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, results, ctx);
    };
    SmallVector<AffineMap> maps = batchMatmul.getIndexingMapsArray();
    if (maps.size() != 3) {
      return failure();
    }
    // Only fold when the LHS is [batch, M, K] and the output is [batch, M, N].
    // A transposed LHS [batch, K, M] cannot be folded: collapsing it merges the
    // batch and leading dims into [batch*K, M], which no longer matches the
    // output's [batch*M, N] collapse.
    if (maps[0] != mapOf({bDim, mDim, kDim}) ||
        maps[2] != mapOf({bDim, mDim, nDim})) {
      return failure();
    }
    // The RHS may be plain (batch, k, n) or transpose_b (batch, n, k).
    bool transposeB;
    if (maps[1] == mapOf({bDim, kDim, nDim})) {
      transposeB = false;
    } else if (maps[1] == mapOf({bDim, nDim, kDim})) {
      transposeB = true;
    } else {
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

    if (transposeB) {
      matmulResult = linalg::MatmulTransposeBOp::create(
                         rewriter, loc, collapsedOutType,
                         ValueRange{collapsedAct, *weight}, collapsedOut)
                         .getResult(0);
    } else {
      matmulResult = linalg::MatmulOp::create(rewriter, loc, collapsedOutType,
                                              ValueRange{collapsedAct, *weight},
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

struct ConvertBatchMatmulToMatmulPass
    : public impl::ConvertBatchMatmulToMatmulPassBase<
          ConvertBatchMatmulToMatmulPass> {
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
