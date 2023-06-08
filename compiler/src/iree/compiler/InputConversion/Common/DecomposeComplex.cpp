// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LinalgFill : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.value();
    auto valueTy = value.getType();
    auto complexTy = dyn_cast<ComplexType>(valueTy);
    auto resultTy = op.getType(0).cast<ShapedType>();
    if (!complexTy) return failure();

    auto real = rewriter.create<complex::ReOp>(op.getLoc(), value);
    auto imag = rewriter.create<complex::ImOp>(op.getLoc(), value);

    SmallVector<utils::IteratorType> loops(resultTy.getRank(),
                                           utils::IteratorType::parallel);
    SmallVector<AffineMap> maps{
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};
    auto generic = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultTy, ValueRange{}, ValueRange{op.getOutputs()[0]},
        maps, loops, [&](OpBuilder &b, Location loc, ValueRange args) {
          auto cmplx =
              b.create<complex::CreateOp>(loc, valueTy, real, imag).getResult();
          b.create<linalg::YieldOp>(loc, cmplx);
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct ComplexConstant : public OpRewritePattern<complex::ConstantOp> {
  using OpRewritePattern<complex::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    ArrayAttr attr = op.getValue().dyn_cast<ArrayAttr>();
    auto realAttr = attr[0].cast<TypedAttr>();
    auto imagAttr = attr[1].cast<TypedAttr>();

    Value real = rewriter.create<arith::ConstantOp>(
        op.getLoc(), realAttr.getType(), realAttr);
    Value imag = rewriter.create<arith::ConstantOp>(
        op.getLoc(), imagAttr.getType(), imagAttr);

    auto result = op.getResult();
    for (OpOperand &operand : llvm::make_early_inc_range(result.getUses())) {
      Operation *targetOp = operand.getOwner();
      rewriter.setInsertionPoint(targetOp);
      auto newOp = rewriter.create<complex::CreateOp>(op.getLoc(), op.getType(),
                                                      real, imag);
      rewriter.updateRootInPlace(targetOp, [&]() { operand.set(newOp); });
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Pass that lowers quantized_matmul to matmul.
struct DecomposeComplexPass
    : public DecomposeComplexPassBase<DecomposeComplexPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<ComplexConstant, LinalgFill>(context);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComplexPass() {
  return std::make_unique<DecomposeComplexPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
