// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-promote-convolution-accumulator"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler::GlobalOptimization {

template <typename MatmulTy>
static void promoteMatmul(RewriterBase &rewriter, MatmulTy matmul) {

  Location loc = matmul->getLoc();

  auto outType = cast<RankedTensorType>(matmul.getResult(0).getType());

  if (!outType.getElementType().isF16())
    return;

  RankedTensorType f32OutType =
      RankedTensorType::get(outType.getShape(), rewriter.getF32Type());

  Value baseOut = matmul.getDpsInitOperand(0)->get();
  auto origFill = baseOut.getDefiningOp<linalg::FillOp>();
  if (!origFill)
    return;
  auto origEmpty =
      origFill.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
  if (!origEmpty)
    return;

  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32Type(), rewriter.getZeroAttr(rewriter.getF32Type()));
  auto newEmpty = rewriter.create<tensor::EmptyOp>(loc, f32OutType,
                                                   origEmpty.getDynamicSizes());
  auto newFill = rewriter.create<linalg::FillOp>(loc, f32OutType, zeroF32,
                                                 newEmpty.getResult());

  MatmulTy newMatmul = rewriter.create<MatmulTy>(
      loc, f32OutType, matmul.getDpsInputs(), newFill.getResult(0));

  SmallVector<utils::IteratorType> iteratorTypes(outType.getRank(),
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps(
      2, rewriter.getMultiDimIdentityMap(outType.getRank()));

  linalg::GenericOp newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, outType, newMatmul.getResult(0), origEmpty.getResult(), indexingMaps,
      iteratorTypes, [](OpBuilder &builder, Location loc, ValueRange args) {
        Value trunc =
            builder.create<arith::TruncFOp>(loc, builder.getF16Type(), args[0]);
        builder.create<linalg::YieldOp>(loc, trunc);
      });
  rewriter.replaceOp(matmul, newGenericOp);
}

static void promoteConv(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp conv) {

  Location loc = conv->getLoc();

  auto outType = cast<RankedTensorType>(conv.getResult(0).getType());

  if (!outType.getElementType().isF16())
    return;

  RankedTensorType f32OutType =
      RankedTensorType::get(outType.getShape(), rewriter.getF32Type());

  Value baseOut = conv.getDpsInitOperand(0)->get();
  auto origFill = baseOut.getDefiningOp<linalg::FillOp>();
  if (!origFill)
    return;
  auto origEmpty =
      origFill.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
  if (!origEmpty)
    return;

  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32Type(), rewriter.getZeroAttr(rewriter.getF32Type()));
  auto newEmpty = rewriter.create<tensor::EmptyOp>(loc, f32OutType,
                                                   origEmpty.getDynamicSizes());
  auto newFill = rewriter.create<linalg::FillOp>(loc, f32OutType, zeroF32,
                                                 newEmpty.getResult());

  linalg::Conv2DNhwcHwcfOp newConv = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
      loc, f32OutType, conv.getDpsInputs(), newFill.getResult(0),
      conv.getStrides(), conv.getDilations());

  SmallVector<utils::IteratorType> iteratorTypes(outType.getRank(),
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps(
      2, rewriter.getMultiDimIdentityMap(outType.getRank()));

  linalg::GenericOp newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, outType, newConv.getResult(0), origEmpty.getResult(), indexingMaps,
      iteratorTypes, [](OpBuilder &builder, Location loc, ValueRange args) {
        Value trunc =
            builder.create<arith::TruncFOp>(loc, builder.getF16Type(), args[0]);
        builder.create<linalg::YieldOp>(loc, trunc);
      });
  rewriter.replaceOp(conv, newGenericOp);
}

static void promoteContraction(RewriterBase &rewriter,
                               linalg::GenericOp contract) {

  Location loc = contract->getLoc();

  auto outType = cast<RankedTensorType>(contract.getResult(0).getType());

  if (!outType.getElementType().isF16())
    return;

  RankedTensorType f32OutType =
      RankedTensorType::get(outType.getShape(), rewriter.getF32Type());

  Value baseOut = contract.getDpsInitOperand(0)->get();
  auto origFill = baseOut.getDefiningOp<linalg::FillOp>();
  if (!origFill)
    return;
  auto origEmpty =
      origFill.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
  if (!origEmpty)
    return;

  Value zeroF32 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32Type(), rewriter.getZeroAttr(rewriter.getF32Type()));
  auto newEmpty = rewriter.create<tensor::EmptyOp>(loc, f32OutType,
                                                   origEmpty.getDynamicSizes());
  auto newFill = rewriter.create<linalg::FillOp>(loc, f32OutType, zeroF32,
                                                 newEmpty.getResult());

  linalg::GenericOp newContractionOp = rewriter.create<linalg::GenericOp>(
      loc, f32OutType, contract.getDpsInputs(), newFill.getResult(0),
      contract.getIndexingMapsArray(), contract.getIteratorTypesArray(),
      [](OpBuilder &builder, Location loc, ValueRange args) {
        Value extLhs =
            builder.create<arith::ExtFOp>(loc, builder.getF32Type(), args[0]);
        Value extRhs =
            builder.create<arith::ExtFOp>(loc, builder.getF32Type(), args[1]);
        Value mul = builder.create<arith::MulFOp>(loc, extLhs, extRhs);
        Value add = builder.create<arith::MulFOp>(loc, mul, args[2]);
        builder.create<linalg::YieldOp>(loc, add);
      });

  SmallVector<utils::IteratorType> iteratorTypes(outType.getRank(),
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps(
      2, rewriter.getMultiDimIdentityMap(outType.getRank()));

  linalg::GenericOp newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, outType, newContractionOp.getResult(0), origEmpty.getResult(),
      indexingMaps, iteratorTypes,
      [](OpBuilder &builder, Location loc, ValueRange args) {
        Value trunc =
            builder.create<arith::TruncFOp>(loc, builder.getF16Type(), args[0]);
        builder.create<linalg::YieldOp>(loc, trunc);
      });
  rewriter.replaceOp(contract, newGenericOp);
}

namespace {

struct PromoteConvolutionAccumulatorPass
    : public PromoteConvolutionAccumulatorBase<
          PromoteConvolutionAccumulatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  PromoteConvolutionAccumulatorPass() {}
  PromoteConvolutionAccumulatorPass(
      const PromoteConvolutionAccumulatorPass &pass)
      : PromoteConvolutionAccumulatorPass() {}

  void runOnOperation() override;
};

} // namespace

void PromoteConvolutionAccumulatorPass::runOnOperation() {
  MLIRContext *context = &getContext();

  SmallVector<linalg::Conv2DNhwcHwcfOp> convs;

  getOperation()->walk(
      [&](linalg::Conv2DNhwcHwcfOp conv) { convs.push_back(conv); });

  IRRewriter rewriter(context);
  for (auto conv : convs) {
    rewriter.setInsertionPointAfter(conv);
    promoteConv(rewriter, conv);
  }

  SmallVector<linalg::GenericOp> contracts;

  getOperation()->walk([&](linalg::GenericOp contract) {
    if (linalg::isaContractionOpInterface(contract))
      contracts.push_back(contract);
  });

  for (auto contract : contracts) {
    rewriter.setInsertionPointAfter(contract);
    promoteContraction(rewriter, contract);
  }

  SmallVector<linalg::MatmulTransposeBOp> transposedMatmuls;

  getOperation()->walk([&](linalg::MatmulTransposeBOp matmul) {
    transposedMatmuls.push_back(matmul);
  });

  for (auto matmul : transposedMatmuls) {
    rewriter.setInsertionPointAfter(matmul);
    promoteMatmul(rewriter, matmul);
  }

  SmallVector<linalg::MatmulOp> matmuls;

  getOperation()->walk(
      [&](linalg::MatmulOp matmul) { matmuls.push_back(matmul); });

  for (auto matmul : matmuls) {
    rewriter.setInsertionPointAfter(matmul);
    promoteMatmul(rewriter, matmul);
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPromoteConvolutionAccumulatorPass() {
  return std::make_unique<PromoteConvolutionAccumulatorPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
