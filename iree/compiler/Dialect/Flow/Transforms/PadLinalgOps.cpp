// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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

    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType) return failure();

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
      return failure();
    }

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    int M = lhsShape[0], K = lhsShape[1], N = rhsShape[1];

    int newMSize = std::ceil(float(M) / paddingSize) * paddingSize;
    int newNSize = std::ceil(float(N) / paddingSize) * paddingSize;
    int newKSize = std::ceil(float(K) / paddingSize) * paddingSize;

    int paddingForM = newMSize - M;
    int paddingForN = newNSize - N;
    int paddingForK = newKSize - K;

    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();
    auto elementType = lhsType.getElementType();

    auto lhsPaddedType =
        RankedTensorType::get({newMSize, newKSize}, elementType);

    auto rhsPaddedType =
        RankedTensorType::get({newKSize, newNSize}, elementType);
    Value paddingValue =
        rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elementType));

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    Value paddedLhs =
        (paddingForM > 0 || paddingForK > 0)
            ? linalg::PadTensorOp::createPadScalarOp(
                  lhsPaddedType, lhs, paddingValue, createPadding({0, 0}),
                  createPadding({paddingForM, paddingForK}), loc, rewriter)
            : lhs;

    auto paddedrhs =
        (paddingForK > 0 || paddingForN > 0)
            ? linalg::PadTensorOp::createPadScalarOp(
                  rhsPaddedType, rhs, paddingValue, createPadding({0, 0}),
                  createPadding({paddingForK, paddingForN}), loc, rewriter)
            : rhs;

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
      auto resultPaddedType =
          RankedTensorType::get({newMSize, newNSize}, elementType);
      Value paddedResult = linalg::PadTensorOp::createPadScalarOp(
          resultPaddedType, result, paddingValue, createPadding({0, 0}),
          createPadding({paddingForM, paddingForN}), loc, rewriter);
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs, paddedResult});

      SmallVector<OpFoldResult> offsets(2, rewriter.getI64IntegerAttr(0));
      SmallVector<OpFoldResult> strides(2, rewriter.getI64IntegerAttr(1));
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(M),
                                         rewriter.getIndexAttr(N)};
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          matmulOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }

    return success();
  }

 private:
  int paddingSize;
};

class PadLinalgOpsPass : public PadLinalgOpsBase<PadLinalgOpsPass> {
 public:
  PadLinalgOpsPass(int size) : paddingSize(size) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(context);
    patterns.insert<PadMatmulOp>(context, paddingSize);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

 private:
  int paddingSize;
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createPadLinalgOpsToIntegerMultiplePass(
    int paddingSize) {
  return std::make_unique<PadLinalgOpsPass>(paddingSize);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
