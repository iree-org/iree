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
static Value transposeOperand(mlir::Location loc, PatternRewriter &rewriter,
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

// Converts linalg.matmul -> linalg.mmt4d where M0, N0, K0 are compile time
// constants.
// TODO(ataei): Move this pattern to linalg transforms upstream.
class LinalgMatmulOpToLinalgMMT4dOpPattern
    : public OpRewritePattern<linalg::MatmulOp> {
 public:
  LinalgMatmulOpToLinalgMMT4dOpPattern(MLIRContext *context, int M0, int N0,
                                       int K0, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        M0Size(M0),
        N0Size(N0),
        K0Size(K0) {}

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

    // This is for float only matmul for now. Integer data type might require
    // r.h.s layout change.
    if (!lhsType.getElementType().isF32() || !rhsType.getElementType().isF32())
      return failure();

    int m = lhsType.getShape()[0];
    int n = rhsType.getShape()[1];
    int k = rhsType.getShape()[0];

    if (m % M0Size != 0 || n % N0Size != 0 || k % K0Size != 0) return failure();

    int m1 = m / M0Size;
    int n1 = n / N0Size;
    int k1 = k / K0Size;

    auto lhs4D = expandTo4D(loc, rewriter, lhs, {m1, M0Size, k1, K0Size});
    auto rhs4D = expandTo4D(loc, rewriter, rhs, {k1, K0Size, n1, N0Size});
    auto dst4D = expandTo4D(loc, rewriter, dst, {m1, M0Size, n1, N0Size});

    auto lhs4DT = transposeOperand(loc, rewriter, lhs4D, {0, 2, 1, 3});
    auto rhs4DT = transposeOperand(loc, rewriter, rhs4D, {2, 0, 1, 3});
    auto dst4DT = transposeOperand(loc, rewriter, dst4D, {0, 2, 1, 3});

    auto mmt4DResult = rewriter.create<linalg::Mmt4DOp>(
        loc, dst4DT.getType(), ValueRange{lhs4DT, rhs4DT}, ValueRange{dst4DT});

    auto mmt4dResultTransposed =
        transposeOperand(loc, rewriter, mmt4DResult.getResult(0), {0, 2, 1, 3});

    Value result = collapseTo2D(loc, rewriter, mmt4dResultTransposed, {m, n});

    rewriter.replaceOp(matmulOp, ArrayRef<Value>{result});

    return success();
  }

 private:
  int M0Size;
  int N0Size;
  int K0Size;
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

class ConvertLinalgMatmulOpToLinalgMMT4dPass
    : public ConvertMatmulToMMT4dBase<ConvertLinalgMatmulOpToLinalgMMT4dPass> {
 public:
  ConvertLinalgMatmulOpToLinalgMMT4dPass(int M0, int N0, int K0)
      : M0Size(M0), N0Size(K0), K0Size(N0) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      OwningRewritePatternList patterns(&getContext());
      patterns.insert<LinalgMatmulOpToLinalgMMT4dOpPattern>(context, M0Size,
                                                            N0Size, K0Size);
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

 private:
  int M0Size;
  int N0Size;
  int K0Size;
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createConvertLinalgMatmulOpToLinalgMMT4dPass(int M0, int N0, int K0) {
  return std::make_unique<ConvertLinalgMatmulOpToLinalgMMT4dPass>(M0, N0, K0);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
