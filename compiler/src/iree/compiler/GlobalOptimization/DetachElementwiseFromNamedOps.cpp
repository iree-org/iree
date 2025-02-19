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

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_DETACHELEMENTWISEFROMNAMEDOPSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

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
    if (!linalgOp.hasPureTensorSemantics())
      return failure();

    // Nothing to do if the output tensor operand is already a fill op.
    SmallVector<OpOperand *> outputOperands;
    if (!linalgOp.hasPureBufferSemantics()) {
      outputOperands = llvm::to_vector(
          llvm::map_range(linalgOp.getDpsInitsMutable(),
                          [](OpOperand &opOperand) { return &opOperand; }));
    }
    // Right now all the cases we see have one output. This can be relaxed once
    // we see multiple output ops.
    if (outputOperands.size() != 1)
      return failure();
    Value outputOperand = outputOperands.front()->get();

    auto outsDefiningOp = outputOperand.getDefiningOp<linalg::LinalgOp>();
    if (!outsDefiningOp || isa<linalg::FillOp>(outsDefiningOp.getOperation())) {
      // If not linalg op, or is a fill op, do nothing.
      return failure();
    }
    auto outputType = llvm::cast<RankedTensorType>(outputOperand.getType());
    if (!outputType.getElementType().isIntOrFloat())
      return failure();
    auto elementType = outputType.getElementType();

    Location loc = linalgOp.getLoc();

    // Check if the output tensor access is a projected permutation
    if (!linalgOp.getMatchingIndexingMap(outputOperands.front())
             .isProjectedPermutation()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Output indexing map must be a projected permutation.");
    }

    int64_t outputRank = outputType.getRank();
    SmallVector<utils::IteratorType> iterators(outputRank,
                                               utils::IteratorType::parallel);
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(outputRank));

    // Create a zero tensor as the new output tensor operand to the Linalg
    // contraction op.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, outputOperand);
    auto initOp =
        rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value fill =
        rewriter.create<linalg::FillOp>(loc, zero, initOp.getResult()).result();

    // Update the contraction op to use the new zero tensor as output operand.
    rewriter.modifyOpInPlace(linalgOp,
                             [&]() { linalgOp.setDpsInitOperand(0, fill); });

    // Create a generic op to add back the original output tensor operand.
    rewriter.setInsertionPointAfter(linalgOp);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{linalgOp->getResult(0), outputOperand},
        fill, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result;
          if (llvm::isa<FloatType>(elementType)) {
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
/// operations. More canonical representation is to use a `empty -> fill
/// -> outs` operand sequence. Splat constants pulled in this way causes issues
/// with allocations. Using `fill` will allow for fusing with the op just like
/// fill -> linalg ops are fused. If not as a fallback they would be converted
/// to a splat, but both without stack allocations.
template <typename InterfaceOp>
struct DetachSplatConstantOutsOperands
    : public OpInterfaceRewritePattern<InterfaceOp> {
  using OpInterfaceRewritePattern<InterfaceOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(InterfaceOp interfaceOp,
                                PatternRewriter &rewriter) const {
    SmallVector<Value> newOutsOperands;
    auto dpsInterfaceOp =
        dyn_cast<DestinationStyleOpInterface>(interfaceOp.getOperation());
    if (!dpsInterfaceOp) {
      return rewriter.notifyMatchFailure(
          interfaceOp, "expected op to implement DPS interface");
    }
    bool madeChanges = false;
    for (auto outOperand : llvm::enumerate(dpsInterfaceOp.getDpsInits())) {
      auto constOp =
          outOperand.value().template getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        continue;

      auto resultType =
          llvm::dyn_cast<RankedTensorType>(constOp.getResult().getType());
      if (!resultType || !resultType.getElementType().isIntOrFloat())
        continue;

      auto attr = llvm::dyn_cast<ElementsAttr>(constOp.getValue());
      if (!attr || !attr.isSplat())
        continue;

      Location loc = constOp.getLoc();
      Type elementType = resultType.getElementType();
      Value emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), elementType);
      TypedAttr constValue;
      if (llvm::isa<IntegerType>(elementType)) {
        constValue = rewriter.getIntegerAttr(
            elementType, attr.template getSplatValue<APInt>());
      } else {
        constValue = rewriter.getFloatAttr(
            elementType, attr.template getSplatValue<APFloat>());
      }
      Value scalarConstantOp =
          rewriter.create<arith::ConstantOp>(loc, elementType, constValue);

      Value fillOp = rewriter
                         .create<linalg::FillOp>(
                             loc, resultType, scalarConstantOp, emptyTensorOp)
                         .getResult(0);
      rewriter.modifyOpInPlace(dpsInterfaceOp, [&]() {
        dpsInterfaceOp.setDpsInitOperand(outOperand.index(), fillOp);
      });
      madeChanges = true;
    }
    return success(madeChanges);
  };
};

struct DetachElementwiseFromNamedOpsPass
    : public impl::DetachElementwiseFromNamedOpsPassBase<
          DetachElementwiseFromNamedOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DetachElementwisePattern,
                 DetachSplatConstantOutsOperands<IREE::LinalgExt::LinalgExtOp>,
                 DetachSplatConstantOutsOperands<linalg::LinalgOp>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
