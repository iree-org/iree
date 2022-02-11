// Copyright 2021 The IREE Authors
 //
 // Licensed under the Apache License v2.0 with LLVM Exceptions.
 // See https://llvm.org/LICENSE.txt for license information.
 // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "iree-linalg-ext-scan-vec"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

namespace {

static llvm::Optional<vector::CombiningKind>
getKindForOp(Operation *scanOp) {
  return llvm::TypeSwitch<Operation *, llvm::Optional<vector::CombiningKind>>(
             scanOp)
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return vector::CombiningKind::ADD; })
      .Case<arith::AndIOp>([&](auto op) { return vector::CombiningKind::AND; })
      .Case<arith::MaxSIOp>(
          [&](auto op) { return vector::CombiningKind::MAXSI; })
      .Case<arith::MaxFOp>([&](auto op) { return vector::CombiningKind::MAXF; })
      .Case<arith::MinSIOp>(
          [&](auto op) { return vector::CombiningKind::MINSI; })
      .Case<arith::MinFOp>([&](auto op) { return vector::CombiningKind::MINF; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return vector::CombiningKind::MUL; })
      .Case<arith::OrIOp>([&](auto op) { return vector::CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return vector::CombiningKind::XOR; })
      .Default([&](auto op) { return llvm::None; });
}

struct VectorizeScanPattern : public OpRewritePattern<ScanOp> {
  using OpRewritePattern<ScanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScanOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = op.input().getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();
    auto accumulatorType = op.accumulator().getType().cast<ShapedType>();
    auto accumulatorShape = accumulatorType.getShape();
    VectorType ivType = VectorType::get(accumulatorShape, accumulatorType.getElementType());
    VectorType readType = VectorType::get(inputShape, inputType.getElementType());
    BlockAndValueMapping bvm;
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t rank = inputType.getRank();
    SmallVector<Value> indices(rank, zero);
    Value readValue = rewriter.create<vector::TransferReadOp>(loc, readType, op.input(), indices);
    bvm.map(op.region().getArgument(0), readValue);
    SmallVector<Value> accIndices = indices;
    accIndices.pop_back();
    Value accValue = rewriter.create<vector::TransferReadOp>(loc, ivType, op.accumulator(), accIndices);

    llvm::Optional<vector::CombiningKind> maybeKind;
    maybeKind = getKindForOp(&(op.region().front().front()));
    if (!maybeKind)
      return failure();
    int64_t reductionDim = op.dimension();
    auto scanOp = rewriter.create<vector::ScanOp>(loc, readType, ivType,
        *maybeKind, readValue, accValue,
        reductionDim, op.inclusive());
    auto writeScanResultOp = rewriter.create<vector::TransferWriteOp>(loc, 
        scanOp.dest(), op.output(), indices);
    auto writeReductionResultOp = rewriter.create<vector::TransferWriteOp>(loc, 
        scanOp.accumulated_value(), op.accumulator(), accIndices);
    rewriter.replaceOp(op, writeScanResultOp.getResults());
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct ScanVectorizationPass
    : public ScanVectorizationBase<ScanVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<VectorizeScanPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
IREE::LinalgExt::createScanVectorizationPass() {
  return std::make_unique<ScanVectorizationPass>();
}
