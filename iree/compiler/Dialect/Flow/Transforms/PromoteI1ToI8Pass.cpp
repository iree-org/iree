// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Legalizes boolean (i1) constants to i8 with a linalg.generic operation
// downcasting to i1. This occurs as IREE does not currently support tightly
// packing and unpacking i1 buffers.
class ConvertBoolConstantPattern
    : public OpRewritePattern<mlir::arith::ConstantOp> {
 public:
  using OpRewritePattern<mlir::arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = op.getType().dyn_cast<ShapedType>();

    if (!resultTy) return failure();

    auto eTy = resultTy.getElementType();
    if (!eTy.isInteger(1)) return failure();

    // Constant is never used, ignore.
    if (op.getResult().use_empty()) return failure();

    DenseIntElementsAttr attr = op.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!attr) return failure();

    // Create a new ConstantOp that contains the same values as an int8.
    auto newConst = rewriter.createOrFold<arith::ConstantOp>(
        loc, attr.mapValues(rewriter.getIntegerType(8),
                            [&](APInt src) { return src.zext(8); }));

    // We need to move the insertion to just before its first use case. This is
    // needed as it is possible we are reusing an existing ConstantOp
    // containing the same values that occurs in a future line. Moving to the
    // first use case avoids declaring out of order operations.
    Operation *firstUser = *op.getResult().getUsers().begin();
    for (auto checkOp : op.getResult().getUsers()) {
      if (checkOp->isBeforeInBlock(firstUser)) {
        firstUser = checkOp;
      }
    }
    rewriter.setInsertionPoint(firstUser);

    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ArrayRef<Value>({}), resultTy.getShape(),
        resultTy.getElementType());

    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(resultTy.getRank()),
        rewriter.getMultiDimIdentityMap(resultTy.getRank())};

    // Insert a generic op that Truncates the new i8 values to i1 for use as
    // the original value.
    Value genericOp =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange({resultTy}), ValueRange({newConst}),
                ValueRange({initTensor}), indexingMaps,
                SmallVector<StringRef>(resultTy.getRank(),
                                       getParallelIteratorTypeName()),
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange blockArgs) {
                  auto cast = rewriter.create<arith::TruncIOp>(
                      nestedLoc, rewriter.getIntegerType(1), blockArgs[0]);
                  rewriter.create<linalg::YieldOp>(nestedLoc,
                                                   cast->getResult(0));
                })
            ->getResult(0);

    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

}  // namespace

class PromoteI1ToI8Pass : public PromoteI1ToI8Base<PromoteI1ToI8Pass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mlir::StandardOpsDialect,
                    mlir::math::MathDialect, mlir::arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ConvertBoolConstantPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::FuncOp>> createPromoteI1ToI8Pass() {
  return std::make_unique<PromoteI1ToI8Pass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
