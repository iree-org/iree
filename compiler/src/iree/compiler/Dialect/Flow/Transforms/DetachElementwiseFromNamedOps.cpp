// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- DetachElementwiseFromNamedOps.cpp ----------------------------------===//
//
// Detaches elementwise ops from Linalg named ops in preparation for following
// fusion and bufferization.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct DetachElementwisePattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
      return failure();
    }
    if (!linalgOp.hasTensorSemantics()) return failure();

    // Nothing to do if the output tensor operand is already a fill op.
    linalg::OpOperandVector outputOperands = linalgOp.getOutputTensorOperands();
    // Right now all the cases we see have one output. This can be relaxed once
    // we see multiple output ops.
    if (outputOperands.size() != 1) return failure();
    Value outputOperand = outputOperands.front()->get();
    if (outputOperand.getDefiningOp<linalg::FillOp>()) return failure();

    auto outputType = outputOperand.getType().cast<RankedTensorType>();
    if (!outputType.getElementType().isIntOrFloat()) return failure();
    auto elementType = outputType.getElementType();

    Location loc = linalgOp.getLoc();

    // Create a zero tensor as the new output tensor operand to the Linalg
    // contraction op.
    SmallVector<Value> dynamicDims;
    for (unsigned i = 0; i < outputType.getRank(); i++) {
      if (outputType.isDynamicDim(i))
        dynamicDims.push_back(
            rewriter.create<tensor::DimOp>(loc, outputOperand, i));
    }
    auto initOp = rewriter.create<linalg::InitTensorOp>(
        loc, dynamicDims, outputType.getShape(), elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value fill =
        rewriter.create<linalg::FillOp>(loc, zero, initOp.result()).result();

    // Update the contraction op to use the new zero tensor as output operand.
    rewriter.updateRootInPlace(linalgOp,
                               [&]() { linalgOp.setOutputOperand(0, fill); });

    auto outputMap = mlir::compressUnusedDims(
        linalgOp.getTiedIndexingMap(outputOperands.front()));
    // Only support identity map for output access for now; this is the case for
    // all existing contraction/convolution ops.
    if (!outputMap.isIdentity()) return failure();
    SmallVector<AffineMap> maps(3, outputMap);

    SmallVector<StringRef> iterators;
    iterators.reserve(outputMap.getNumResults());
    for (int i = 0, e = outputMap.getNumResults(); i < e; ++i) {
      int pos = outputMap.getResult(i).cast<AffineDimExpr>().getPosition();
      auto attr = linalgOp.getIteratorTypes()[pos].cast<StringAttr>();
      if (!isParallelIterator(attr)) return failure();
      iterators.push_back(attr.getValue());
    }

    // Create a generic op to add back the original output tensor operand.
    rewriter.setInsertionPointAfter(linalgOp);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{linalgOp->getResult(0), outputOperand},
        fill, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result;
          if (elementType.isa<FloatType>()) {
            result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          } else {
            result = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          }
          b.create<linalg::YieldOp>(nestedLoc, result);
        });
    linalgOp->getResult(0).replaceAllUsesExcept(genericOp->getResult(0),
                                                genericOp);
    return success();
  }
};

struct DetachElementwiseFromNamedOpsPass
    : public DetachElementwiseFromNamedOpsBase<
          DetachElementwiseFromNamedOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DetachElementwisePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass() {
  return std::make_unique<DetachElementwiseFromNamedOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
