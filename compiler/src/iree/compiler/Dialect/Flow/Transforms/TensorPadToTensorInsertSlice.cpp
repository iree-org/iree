// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- TensorPadToInsertSlice.cpp ----- Pass to legalize tensor.pad -------===//
//
// Pass to convert tensor.pad to linalg.fill + tensor.insert_slice.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// Pattern to convert a tensor.tensor operation into a fill +
/// tensor.insert_slice. This is needed till tensor.pad op can be fused with its
/// consumers.
struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  TensorPadOpConversion(MLIRContext *context, bool skipSingleLinalgOpUses)
      : OpRewritePattern<tensor::PadOp>(context, skipSingleLinalgOpUses),
        skipSingleLinalgOpUses(skipSingleLinalgOpUses) {}

  LogicalResult matchAndRewrite(tensor::PadOp padTensorOp,
                                PatternRewriter &rewriter) const override {
    // Check that the region is just a yield operation which is returning a
    // scalar that is not one of the arguments of the linalg operation.
    Region &region = padTensorOp.getRegion();
    Block &block = region.front();
    if (!llvm::hasSingleElement(block)) return failure();
    auto yieldOp = cast<tensor::YieldOp>(block.getTerminator());
    Value yieldVal = yieldOp.getValue();
    if (llvm::any_of(block.getArguments(),
                     [&](Value v) { return v == yieldVal; })) {
      return failure();
    }

    if (skipSingleLinalgOpUses && padTensorOp->hasOneUse()) {
      Operation *use = padTensorOp->use_begin()->getOwner();
      // TODO(#10312): Relax the condition to not check quantized ops. They
      // are going to be deprecated. We don't expect them being IREE's input.
      if (isa<linalg::LinalgOp>(use) &&
          !isa<linalg::Conv2DNhwcHwcfQOp, linalg::DepthwiseConv2DNhwcHwcQOp,
               linalg::DepthwiseConv2DNhwcHwcmQOp>(use)) {
        return failure();
      }
    }

    OpBuilder::InsertionGuard g(rewriter);
    Location loc = padTensorOp.getLoc();
    auto lowPad = padTensorOp.getMixedLowPad();
    auto highPad = padTensorOp.getMixedHighPad();
    Value source = padTensorOp.getSource();
    RankedTensorType sourceType = padTensorOp.getSourceType();
    int64_t rank = sourceType.getRank();

    // TODO(ravishankarm): Use shape inference interface to get this.
    SmallVector<OpFoldResult> sourceShape;
    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : llvm::seq<int64_t>(0, rank)) {
      SmallVector<Value> mapValues;
      Value sourceDim = rewriter.createOrFold<tensor::DimOp>(loc, source, dim);
      mapValues.push_back(sourceDim);
      if (auto cstDim = sourceDim.getDefiningOp<arith::ConstantIndexOp>()) {
        sourceShape.push_back(cstDim.getValue());
      } else {
        sourceShape.push_back(sourceDim);
      }
      AffineExpr expr = rewriter.getAffineDimExpr(0);
      unsigned numSymbols = 0;
      auto addValueOrAttr = [&](AffineExpr e, OpFoldResult valueOrAttr) {
        if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
          e = e + llvm::cast<IntegerAttr>(attr).getInt();
          return e;
        }
        e = e + rewriter.getAffineSymbolExpr(numSymbols++);
        mapValues.push_back(valueOrAttr.get<Value>());
        return e;
      };
      expr = addValueOrAttr(expr, lowPad[dim]);
      expr = addValueOrAttr(expr, highPad[dim]);
      Value v = affine::applyMapToValues(
          rewriter, loc, AffineMap::get(1, numSymbols, expr), mapValues)[0];
      if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
        outputShape.push_back(cst.getValue());
      } else {
        outputShape.push_back(v);
      }
    }
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputShape, sourceType.getElementType());
    Value fill = rewriter.create<linalg::FillOp>(loc, yieldVal, emptyTensor)
                     .getResult(0);
    SmallVector<OpFoldResult> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        padTensorOp, source, fill, lowPad, sourceShape, strides);
    return success();
  }

 private:
  // Option to skip the pattern when tensor.pad op has one use and is used by
  // a Linalg op.
  bool skipSingleLinalgOpUses = false;
};

struct TensorPadToTensorInsertSlicePass
    : public TensorPadToTensorInsertSliceBase<
          TensorPadToTensorInsertSlicePass> {
  TensorPadToTensorInsertSlicePass(bool skipSingleLinalgOpUses)
      : skipSingleLinalgOpUses(skipSingleLinalgOpUses) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, memref::MemRefDialect, func::FuncDialect,
                mlir::math::MathDialect, mlir::arith::ArithDialect>();
  }

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    // `skipSingleLinalgOpUses` may have been set to `true` in the constructor
    // already. The |= is so we preserve that rather than overwrite it with the
    // default value `false` of `optionSkipSingleLinalgOpUses`.
    skipSingleLinalgOpUses |= optionSkipSingleLinalgOpUses;
    return success();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<TensorPadOpConversion>(context, skipSingleLinalgOpUses);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  bool skipSingleLinalgOpUses;
};

}  // namespace

std::unique_ptr<Pass> createTensorPadToTensorInsertSlicePass(
    bool skipSingleLinalgOpUses) {
  return std::make_unique<TensorPadToTensorInsertSlicePass>(
      skipSingleLinalgOpUses);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
