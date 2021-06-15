// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// A pattern to pad staticly shaped matmul operands to the next integer
/// multiple of padSize.
class PadMatmulOp : public OpRewritePattern<linalg::MatmulOp> {
 public:
  PadMatmulOp(MLIRContext *context, int size, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        paddingSize(size) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();
    auto lhs = matmulOp.inputs()[0];
    auto rhs = matmulOp.inputs()[1];
    auto result = matmulOp.outputs()[0];

    if (lhs.getDefiningOp<linalg::PadTensorOp>() ||
        rhs.getDefiningOp<linalg::PadTensorOp>()) {
      return failure();
    }

    auto lhsShapeType = lhs.getType().cast<ShapedType>();
    auto rhsShapeType = rhs.getType().cast<ShapedType>();

    if (!lhsShapeType || !rhsShapeType) return failure();

    if (!lhsShapeType.hasStaticShape() || !rhsShapeType.hasStaticShape()) {
      return failure();
    }

    auto lhsShape = lhsShapeType.getShape();
    auto rhsShape = rhsShapeType.getShape();

    int M = lhsShape[0], K = lhsShape[1], N = rhsShape[1];

    int newMSize = std::ceil(float(M) / paddingSize) * paddingSize;
    int newNSize = std::ceil(float(N) / paddingSize) * paddingSize;
    int newKSize = std::ceil(float(K) / paddingSize) * paddingSize;

    int paddingForM = newMSize - M;
    int paddingForN = newNSize - N;
    int paddingForK = newKSize - K;

    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();

    auto getPaddedOperand = [&](Value operand, ArrayRef<int64_t> shape,
                                ArrayRef<int64_t> highPadding) -> Value {
      if (llvm::all_of(highPadding,
                       [](int64_t val) -> bool { return val == 0; })) {
        return operand;
      }
      auto elementType =
          operand.getType().cast<RankedTensorType>().getElementType();
      auto paddedType = RankedTensorType::get(shape, elementType);
      Value paddingValue =
          rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elementType));

      auto padTensorOp = rewriter.create<linalg::PadTensorOp>(
          loc, paddedType, operand, ArrayRef<Value>{}, ArrayRef<Value>{},
          rewriter.getI64ArrayAttr({0, 0}),
          rewriter.getI64ArrayAttr(highPadding));

      int rank = padTensorOp.getResultType().getRank();
      SmallVector<Type, 4> blockArgTypes;
      blockArgTypes.assign(rank, rewriter.getIndexType());
      auto &region = padTensorOp.region();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&region, region.end(), blockArgTypes);
      rewriter.create<linalg::YieldOp>(loc, paddingValue);
      return padTensorOp;
    };

    auto paddedLhs =
        getPaddedOperand(lhs, {newMSize, newKSize}, {paddingForM, paddingForK});

    auto paddedrhs =
        getPaddedOperand(rhs, {newKSize, newNSize}, {paddingForK, paddingForN});

    auto resultType = RankedTensorType::get(
        {newMSize, newNSize},
        result.getType().cast<RankedTensorType>().getElementType());

    // Padding for K-dim only result doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs, result});
      rewriter.replaceOp(matmulOp, paddedMatmulOp->getResults());
    } else {
      auto paddedResult = getPaddedOperand(result, {newMSize, newNSize},
                                           {paddingForM, paddingForN});
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs, paddedResult});

      SmallVector<OpFoldResult> offsets(2, rewriter.getI64IntegerAttr(0));
      SmallVector<OpFoldResult> strides(2, rewriter.getI64IntegerAttr(1));
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(M),
                                         rewriter.getIndexAttr(N)};
      rewriter.replaceOpWithNewOp<SubTensorOp>(
          matmulOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }

    return success();
  }

 private:
  int paddingSize;
};

class PadLinalgOpsPass : public PassWrapper<PadLinalgOpsPass, FunctionPass> {
 public:
  PadLinalgOpsPass(int size) : paddingSize(size) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnFunction() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(context);
    patterns.insert<PadMatmulOp>(context, paddingSize);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  int paddingSize;
};
}  // namespace

std::unique_ptr<FunctionPass> createPadLinalgOpsToIntegerMultiplePass(
    int paddingSize) {
  return std::make_unique<PadLinalgOpsPass>(paddingSize);
}

}  // namespace iree_compiler
}  // namespace mlir
