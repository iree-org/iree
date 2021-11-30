// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PadTensorToSubTensorInsert.cpp - Pass to legalize linalg.pad_tensor-===//
//
// Pass to convert linalg.pad_tensor to linalg.fill + subtensor_insert
// operations which is the only way Vulkan backend can lower it to a single
// kernel.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Pattern to convert a linalg.pad_tensor operation into a fill + subtensor
/// insert. This is needed till pad_tensor op can be fused with its consumers.
struct PadTensorOpConversion : public OpRewritePattern<linalg::PadTensorOp> {
  using OpRewritePattern<linalg::PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PadTensorOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.region();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<linalg::YieldOp>(block.getTerminator());
    if (!llvm::hasSingleElement(yieldOp.values())) return failure();
    Value yieldVal = yieldOp.values().front();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padTensorOp.source();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<tensor::DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      sourceShape.push_back(sourceDim);
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + attr.cast<IntegerAttr>().getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      Value v = linalg::applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0];
      if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
        outputShape.push_back(cst.value());
      } else {
        outputShape.push_back(v);
      }
    }
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill =
        rewriter.create<linalg::FillOp>(loc, yieldVal, initTensor).getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padTensorOp, source, fill, lowPad, sourceShape, strides);
    return success();
  }
};

struct PadTensorToSubTensorInsertPass
    : public PadTensorToSubTensorInsertBase<PadTensorToSubTensorInsertPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    StandardOpsDialect, mlir::math::MathDialect,
                    mlir::arith::ArithmeticDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PadTensorOpConversion>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createPadTensorToSubTensorInsertPass() {
  return std::make_unique<PadTensorToSubTensorInsertPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
