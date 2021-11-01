// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Expands a 2d tensor operand to 4d given its target shape.
// Does not transpose.
// Example: (M, N) --> (M1, M0, N1, N0)
static Value expandTo4D(mlir::Location loc, PatternRewriter &rewriter,
                        Value operand, ArrayRef<int64_t> targetShape) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  auto targetType =
      RankedTensorType::get(targetShape, operandType.getElementType());
  SmallVector<ReassociationIndices> expandIndices = {{0, 1}, {2, 3}};
  Value reshapedOperand = rewriter.create<linalg::TensorExpandShapeOp>(
      loc, targetType, operand, expandIndices);
  return reshapedOperand;
}

// Creates a linalg.generic that transposes operand using permutation indices.
// Example: (M1, M0, N1, N0) -> (M1, N1, M0, N0) if indices = {0, 2, 1, 3}.
static Value transpose(mlir::Location loc, PatternRewriter &rewriter,
                       Value operand, ArrayRef<int64_t> indices) {
  RankedTensorType operandTensorType =
      operand.getType().cast<RankedTensorType>();
  auto nloops = indices.size();
  auto inputShape = operandTensorType.getShape();

  SmallVector<AffineExpr, 4> exprs = llvm::to_vector<4>(
      llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
        return rewriter.getAffineDimExpr(index);
      }));

  SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
      indices, [&](int64_t index) -> int64_t { return inputShape[index]; }));

  Value outputTensor = rewriter.create<linalg::InitTensorOp>(
      loc, targetShape, operandTensorType.getElementType());

  SmallVector<StringRef> loopAttributeTypes(nloops, "parallel");

  SmallVector<AffineMap> indexingMaps = {
      inversePermutation(
          AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
      AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

  auto transposedOp = rewriter.create<linalg::GenericOp>(
      loc, outputTensor.getType(),
      /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  return transposedOp.getResult(0);
};

// Collapses a 4d tensor operand to 2d given its target shape.
// Example: (M1, M0, N1, N0) -> (M, N)
static Value collapseTo2D(mlir::Location loc, PatternRewriter &rewriter,
                          Value operand, ArrayRef<int64_t> targetShape) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  auto targetType =
      RankedTensorType::get(targetShape, operandType.getElementType());
  SmallVector<ReassociationIndices> collapseIndices = {{0, 1}, {2, 3}};
  Value reshapedOperand = rewriter.create<linalg::TensorCollapseShapeOp>(
      loc, targetType, operand, collapseIndices);
  return reshapedOperand;
}

// Converts linalg.matmul to an equivalent subgraph using linalg.mmt4d.
// Currently, M0, N0, K0 are compile time constants.
// TODO(ataei): Move this pattern to linalg transforms upstream.
class LinalgMatmulOpToLinalgMmt4DOpPattern
    : public OpRewritePattern<linalg::MatmulOp> {
 public:
  LinalgMatmulOpToLinalgMmt4DOpPattern(MLIRContext *context, int M0, int K0,
                                       int N0, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        M0(M0),
        K0(K0),
        N0(N0) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();

    Value lhs = matmulOp.getInputOperand(0)->get();
    Value rhs = matmulOp.getInputOperand(1)->get();
    Value dst = matmulOp.getOutputOperand(0)->get();

    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape()) {
      return failure();
    }

    int m = lhsType.getShape()[0];
    int k = rhsType.getShape()[0];
    int n = rhsType.getShape()[1];

    if (m % M0 != 0 || n % N0 != 0 || k % K0 != 0) return failure();

    int m1 = m / M0;
    int k1 = k / K0;
    int n1 = n / N0;

    auto lhs4D = expandTo4D(loc, rewriter, lhs, {m1, M0, k1, K0});
    auto rhs4D = expandTo4D(loc, rewriter, rhs, {k1, K0, n1, N0});
    auto dst4D = expandTo4D(loc, rewriter, dst, {m1, M0, n1, N0});

    auto lhs4DT = transpose(loc, rewriter, lhs4D, {0, 2, 1, 3});
    auto rhs4DT = transpose(loc, rewriter, rhs4D, {2, 0, 3, 1});
    auto dst4DT = transpose(loc, rewriter, dst4D, {0, 2, 1, 3});

    auto mmt4dResult = rewriter.create<linalg::Mmt4DOp>(
        loc, dst4DT.getType(), ValueRange{lhs4DT, rhs4DT}, ValueRange{dst4DT});

    auto mmt4dResultTransposed =
        transpose(loc, rewriter, mmt4dResult.getResult(0), {0, 2, 1, 3});

    Value result = collapseTo2D(loc, rewriter, mmt4dResultTransposed, {m, n});

    rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});

    return success();
  }

 private:
  const int M0;
  const int K0;
  const int N0;
};

/// Canonicalizes [linalg.init_tensor -> linalg.fill -> linalg.generic] ->
/// [linalg.init_tensor -> linalg.fill] where linalg.generic does only copy e.g
/// a transpose.
struct FoldFillGenericOpPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.getNumInputs() != 1) return failure();
    if (genericOp.getNumOutputs() != 1) return failure();

    // Check linalg.generic does have copy only semantics.
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
      return failure();
    auto results =
        llvm::to_vector<4>(genericOp.getBody()->getOps<linalg::YieldOp>());
    if (results.size() != 1) return failure();
    if (results[0].values().size() != 1) return failure();
    auto blockArgument = results[0].values()[0].dyn_cast<BlockArgument>();
    if (!blockArgument || blockArgument.getArgNumber() != 0) return failure();

    auto input = genericOp.inputs()[0];

    auto outputType =
        genericOp.outputs()[0].getType().dyn_cast<RankedTensorType>();

    // TODO: To enable dynamic shapes we need to apply the same permutation on
    // init tensor sizes.
    if (!outputType || !outputType.hasStaticShape()) return failure();

    auto fillOp = dyn_cast<linalg::FillOp>(input.getDefiningOp());
    if (!fillOp) return failure();

    auto loc = genericOp.getLoc();
    Value newInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputType.getShape(), outputType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(genericOp, fillOp.value(),
                                                newInitTensor);

    return success();
  }
};

class ConvertLinalgMatmulToMmt4DPass final
    : public ConvertLinalgMatmulToMmt4DBase<ConvertLinalgMatmulToMmt4DPass> {
 public:
  ConvertLinalgMatmulToMmt4DPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) return failure();
    auto failureWithMessage = [=](const char *msg) {
      llvm::errs() << "illegal options `" << options << "` for pass `"
                   << getArgument() << "`: " << msg << "\n";
      return failure();
    };
    if (M0 == mlir::ShapedType::kDynamicSize ||
        N0 == mlir::ShapedType::kDynamicSize ||
        K0 == mlir::ShapedType::kDynamicSize) {
      return failureWithMessage(
          "currently all three values M0,K0,N0 must be "
          "specified as a fixed size value, not 'dynamic', as the heuristic to "
          "choose these values is not yet implemented.");
    }
    if (M0 == 0 || N0 == 0 || K0 == 0) {
      return failureWithMessage("all three values M0,K0,N0 must be nonzero.");
    }
    return success();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      OwningRewritePatternList patterns(&getContext());
      patterns.insert<LinalgMatmulOpToLinalgMmt4DOpPattern>(context, M0, K0,
                                                            N0);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
    // Canonicalization.
    {
      OwningRewritePatternList patterns(&getContext());
      linalg::TensorExpandShapeOp::getCanonicalizationPatterns(patterns,
                                                               context);
      patterns.insert<FoldFillGenericOpPattern>(context);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertLinalgMatmulToMmt4DPass() {
  return std::make_unique<ConvertLinalgMatmulToMmt4DPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
