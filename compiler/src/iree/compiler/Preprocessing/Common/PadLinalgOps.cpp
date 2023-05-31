// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {
/// A pattern to pad statically shaped matmul operands to the next integer
/// multiple of padSize.
class PadMatmulOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  PadMatmulOp(MLIRContext *context, int size, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), paddingSize(size) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = linalgOp.getOperation();
    const bool isBatchMatmul = isa<linalg::BatchMatmulOp>(op);
    const bool isMatmul = isa<linalg::MatmulOp>(op);
    if (!isBatchMatmul && !isMatmul) return failure();

    Location loc = linalgOp.getLoc();
    Value lhs = linalgOp.getDpsInputOperand(0)->get();
    Value rhs = linalgOp.getDpsInputOperand(1)->get();
    Value result = linalgOp.getDpsInitOperand(0)->get();

    auto lhsType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(result.getType());

    if (!lhsType || !rhsType) return failure();

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return failure();

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    const int B = isBatchMatmul ? lhsShape[0] : -1;
    const int M = isBatchMatmul ? lhsShape[1] : lhsShape[0];
    const int K = lhsShape.back(), N = rhsShape.back();

    int newMSize = std::ceil(float(M) / paddingSize) * paddingSize;
    int newNSize = std::ceil(float(N) / paddingSize) * paddingSize;
    int newKSize = std::ceil(float(K) / paddingSize) * paddingSize;

    int paddingForM = newMSize - M;
    int paddingForN = newNSize - N;
    int paddingForK = newKSize - K;

    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();

    auto getFullShape = [&](ArrayRef<int> dims) {
      SmallVector<int64_t, 3> shape;
      if (isBatchMatmul) shape.push_back(B);
      llvm::append_range(shape, dims);
      return shape;
    };

    auto lhsPaddedType = RankedTensorType::get(
        getFullShape({newMSize, newKSize}), lhsType.getElementType());

    auto rhsPaddedType = RankedTensorType::get(
        getFullShape({newKSize, newNSize}), rhsType.getElementType());

    Value lhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(lhsType.getElementType()));

    Value rhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rhsType.getElementType()));

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      if (isBatchMatmul) {
        result.push_back(rewriter.getI64IntegerAttr(0));
      }
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    Value paddedLhs = lhs;
    if (paddingForM > 0 || paddingForK > 0) {
      paddedLhs = rewriter.create<tensor::PadOp>(
          loc, lhsPaddedType, lhs, createPadding({0, 0}),
          createPadding({paddingForM, paddingForK}), lhsPaddingValue);
    }

    Value paddedRhs = rhs;
    if (paddingForK > 0 || paddingForN > 0) {
      paddedRhs = rewriter.create<tensor::PadOp>(
          loc, rhsPaddedType, rhs, createPadding({0, 0}),
          createPadding({paddingForK, paddingForN}), rhsPaddingValue);
    }

    // Padding for K-dim doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          mlir::clone(rewriter, linalgOp, {resultType},
                      ArrayRef<Value>{paddedLhs, paddedRhs, result});
      rewriter.replaceOp(linalgOp, paddedMatmulOp->getResults());
    } else {
      auto newResultType = RankedTensorType::get(
          getFullShape({newMSize, newNSize}), resultType.getElementType());
      Value resultPaddingValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(resultType.getElementType()));
      Value paddedResult = rewriter.create<tensor::PadOp>(
          loc, newResultType, result, createPadding({0, 0}),
          createPadding({paddingForM, paddingForN}), resultPaddingValue);
      auto paddedMatmulOp =
          mlir::clone(rewriter, linalgOp, {newResultType},
                      ArrayRef<Value>{paddedLhs, paddedRhs, paddedResult});

      auto zero = rewriter.getI64IntegerAttr(0);
      auto one = rewriter.getI64IntegerAttr(1);
      auto mAttr = rewriter.getIndexAttr(M);
      auto nAttr = rewriter.getIndexAttr(N);
      SmallVector<OpFoldResult> offsets, strides, sizes;
      if (isBatchMatmul) {
        offsets.assign(3, zero);
        strides.assign(3, one);
        sizes = {rewriter.getIndexAttr(B), mAttr, nAttr};
      } else {
        offsets.assign(2, zero);
        strides.assign(2, one);
        sizes = {mAttr, nAttr};
      }
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          linalgOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }

    return success();
  }

 private:
  int paddingSize;
};

class PadLinalgOpsPass : public PadLinalgOpsBase<PadLinalgOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PadMatmulOp>(context, paddingSize);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass() {
  return std::make_unique<PadLinalgOpsPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
