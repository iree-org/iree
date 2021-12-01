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

    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultType = result.getType().dyn_cast<RankedTensorType>();

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

    auto lhsPaddedType =
        RankedTensorType::get({newMSize, newKSize}, lhsType.getElementType());

    auto rhsPaddedType =
        RankedTensorType::get({newKSize, newNSize}, rhsType.getElementType());

    Value lhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(lhsType.getElementType()));

    Value rhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rhsType.getElementType()));

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
                  lhsPaddedType, lhs, lhsPaddingValue, createPadding({0, 0}),
                  createPadding({paddingForM, paddingForK}), /*nofold=*/false,
                  loc, rewriter)
            : lhs;

    auto paddedrhs =
        (paddingForK > 0 || paddingForN > 0)
            ? linalg::PadTensorOp::createPadScalarOp(
                  rhsPaddedType, rhs, rhsPaddingValue, createPadding({0, 0}),
                  createPadding({paddingForK, paddingForN}), /*nofold=*/false,
                  loc, rewriter)
            : rhs;

    // Padding for K-dim only result doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {resultType},
                     ArrayRef<Value>{paddedLhs, paddedrhs, result});
      rewriter.replaceOp(matmulOp, paddedMatmulOp->getResults());
    } else {
      auto newResultType = RankedTensorType::get({newMSize, newNSize},
                                                 resultType.getElementType());
      auto resultPaddingValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(resultType.getElementType()));
      Value paddedResult = linalg::PadTensorOp::createPadScalarOp(
          newResultType, result, resultPaddingValue, createPadding({0, 0}),
          createPadding({paddingForM, paddingForN}), /*nofold=*/false, loc,
          rewriter);
      auto paddedMatmulOp =
          cast<linalg::LinalgOp>(matmulOp.getOperation())
              .clone(rewriter, loc, {newResultType},
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
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  int paddingSize;
};
}  // namespace

std::unique_ptr<OperationPass<mlir::FuncOp>>
createPadLinalgOpsToIntegerMultiplePass(int paddingSize) {
  return std::make_unique<PadLinalgOpsPass>(paddingSize);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
