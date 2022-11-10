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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    OpOperandVector outputOperands;
    if (!linalgOp.hasBufferSemantics()) {
      outputOperands = linalgOp.getDpsInitOperands();
    }
    // Right now all the cases we see have one output. This can be relaxed once
    // we see multiple output ops.
    if (outputOperands.size() != 1) return failure();
    Value outputOperand = outputOperands.front()->get();

    auto outsDefiningOp = outputOperand.getDefiningOp<linalg::LinalgOp>();
    if (!outsDefiningOp || isa<linalg::FillOp>(outsDefiningOp.getOperation())) {
      // If not linalg op, or is a fill op, do nothing.
      return failure();
    }
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
    auto initOp = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(),
                                                   elementType, dynamicDims);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value fill =
        rewriter.create<linalg::FillOp>(loc, zero, initOp.getResult()).result();

    // Update the contraction op to use the new zero tensor as output operand.
    rewriter.updateRootInPlace(linalgOp,
                               [&]() { linalgOp.setDpsInitOperand(0, fill); });

    auto outputMap = mlir::compressUnusedDims(
        linalgOp.getMatchingIndexingMap(outputOperands.front()));
    // Only support identity map for output access for now; this is the case for
    // all existing contraction/convolution ops.
    if (!outputMap.isIdentity()) return failure();
    SmallVector<AffineMap> maps(3, outputMap);

    SmallVector<StringRef> iterators;
    iterators.reserve(outputMap.getNumResults());
    for (int i = 0, e = outputMap.getNumResults(); i < e; ++i) {
      int pos = outputMap.getResult(i).cast<AffineDimExpr>().getPosition();
      StringRef attr = linalgOp.getIteratorTypesArray()[pos];
      if (!linalg::isParallelIterator(attr)) return failure();
      iterators.push_back(attr);
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

/// Replace uses of splat constants as `outs` operands of `LinalgExt`
/// operations. More canonical representation is to use a `init_tensor -> fill
/// -> outs` operand sequence. Splat constants pulled in this way causes issues
/// with allocations. Using `fill` will allow for fusing with the op just like
/// fill -> linalg ops are fused. If not as a fallback they would be converted
/// to a splat, but both without stack allocations.
struct DetachSplatConstantOutsOperands
    : public OpInterfaceRewritePattern<IREE::LinalgExt::LinalgExtOp> {
  using OpInterfaceRewritePattern<
      IREE::LinalgExt::LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::LinalgExtOp linalgExtOp,
                                PatternRewriter &rewriter) const {
    SmallVector<Value> newOutsOperands;
    bool madeChanges = false;
    for (auto outsOperandNum :
         llvm::seq<unsigned>(0, linalgExtOp.getNumOutputs())) {
      OpOperand *outOperand = linalgExtOp.getOutputOperand(outsOperandNum);
      auto constOp = outOperand->get().getDefiningOp<arith::ConstantOp>();
      if (!constOp) continue;

      auto resultType =
          constOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (!resultType || !resultType.getElementType().isIntOrFloat()) continue;

      auto attr = constOp.getValue().dyn_cast<DenseElementsAttr>();
      if (!attr || !attr.isSplat()) continue;

      Location loc = constOp.getLoc();
      Type elementType = resultType.getElementType();
      Value emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), elementType);
      Attribute constValue;
      if (elementType.isa<IntegerType>()) {
        constValue =
            rewriter.getIntegerAttr(elementType, attr.getSplatValue<APInt>());
      } else {
        constValue =
            rewriter.getFloatAttr(elementType, attr.getSplatValue<APFloat>());
      }
      Value scalarConstantOp =
          rewriter.create<arith::ConstantOp>(loc, elementType, constValue);

      Value fillOp = rewriter
                         .create<linalg::FillOp>(
                             loc, resultType, scalarConstantOp, emptyTensorOp)
                         .getResult(0);
      rewriter.updateRootInPlace(linalgExtOp, [&]() {
        linalgExtOp->setOperand(outOperand->getOperandNumber(), fillOp);
      });
      madeChanges = true;
    }
    return success(madeChanges);
  };
};

struct DetachElementwiseFromNamedOpsPass
    : public DetachElementwiseFromNamedOpsBase<
          DetachElementwiseFromNamedOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DetachElementwisePattern, DetachSplatConstantOutsOperands>(
        &getContext());
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
