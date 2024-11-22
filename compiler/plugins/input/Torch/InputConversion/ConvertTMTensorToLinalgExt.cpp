// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTMTENSORTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

template <typename SrcOpTy, typename TargetOpTy>
struct TMTensorOpConversion : public OpRewritePattern<SrcOpTy> {
  using OpRewritePattern<SrcOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpTy srcOp,
                                PatternRewriter &rewriter) const override {
    OperationState state(srcOp->getLoc(), TargetOpTy::getOperationName(),
                         srcOp->getOperands(), srcOp->getResultTypes(),
                         srcOp->getAttrs(), srcOp->getSuccessors());
    for (Region &srcRegion : srcOp->getRegions()) {
      Region *targetRegion = state.addRegion();
      rewriter.inlineRegionBefore(srcRegion, *targetRegion,
                                  targetRegion->begin());
    }
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }
};

struct ScatterOpConversion
    : public OpRewritePattern<mlir::torch::TMTensor::ScatterOp> {
  using OpRewritePattern<mlir::torch::TMTensor::ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indicesTy = op.getIndicesType();
    if (!indicesTy.hasRank())
      return failure();

    if (indicesTy.isDynamicDim(indicesTy.getRank() - 1)) {
      return rewriter.notifyMatchFailure(op, "number of indices is unknown");
    }

    auto numIndices = indicesTy.getShape().back();
    llvm::SmallVector<int64_t> dimMap(numIndices);
    for (int i = 0; i < numIndices; i++)
      dimMap[i] = i;

    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), op->getResultTypes(), op.getInputs(), op.getOutputs(),
        dimMap, op.getUniqueIndices());

    rewriter.inlineRegionBefore(op.getRegion(), scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};
} // namespace

static SmallVector<AffineMap> getStandardAttentionIndexingMaps(MLIRContext *ctx,
                                                               bool hasMask) {
  AffineExpr m, n, k1, k2;
  bindDims(ctx, m, n, k1, k2);

  auto qMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k1}, ctx);
  auto kMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, k1}, ctx);
  auto vMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, n}, ctx);
  auto sMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, ctx);
  auto rMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, n}, ctx);
  if (hasMask) {
    // Add mask map only if it exists
    auto mMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k2}, ctx);
    return {qMap, kMap, vMap, sMap, mMap, rMap};
  }
  return {qMap, kMap, vMap, sMap, rMap};
}

struct AttentionOpConversion
    : public OpRewritePattern<mlir::torch::TMTensor::AttentionOp> {
  using OpRewritePattern<mlir::torch::TMTensor::AttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    Location loc = op->getLoc();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    std::optional<Value> optionalMask = op.getAttnMask();

    ShapedType outputType = op.getOutputType();

    SmallVector<Value> dynSizes;
    for (int i = 0, s = outputType.getRank() - 1; i < s; ++i) {
      if (outputType.isDynamicDim(i)) {
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, query, i));
      }
    }

    if (outputType.getShape().back() == ShapedType::kDynamic) {
      dynSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, value, outputType.getRank() - 1));
    }

    Value result = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType(), dynSizes);

    // TODO: This is a hack. This should be replaced with a simple getScale()
    // when support for scaling is plumbed to TMTensor on the torch-mlir side.
    // Until then, we are using the default value used in scaled dot product
    // attention by PyTorch (most models use the default value because it makes
    // the variance of the result of softmax 1 when the mean of Q, K is 0).
    // We use scale = 1 / sqrt(d), where d is the head dimension.
    // See https://paperswithcode.com/method/scaled for more details.
    //
    // TODO: We are currently assuming that head dimension is dim = -1. Once we
    // have support for batch dims using more general indexing maps, we should
    // change this and rely on more general mechanisms.
    // TODO: We are currently not handling dynamic shape of head dimensions at
    // all. This is because it messes with dispatch formation. This should be
    // fixed.
    ArrayRef<int64_t> queryShape = op.getQueryType().getShape();
    int64_t headDim = queryShape.back();
    if (headDim == ShapedType::kDynamic) {
      return op->emitOpError("NYI: Dynamic head dimension");
    }

    // Attention only works for FloatType.
    FloatType targetType = cast<FloatType>(op.getQueryType().getElementType());

    double dk = static_cast<double>(headDim);
    dk = 1.0 / std::sqrt(dk);
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, targetType, rewriter.getFloatAttr(targetType, dk));

    // Add batches to standard attention indexing maps.
    SmallVector<AffineMap> indexingMaps =
        getStandardAttentionIndexingMaps(ctx, optionalMask.has_value());

    int64_t numBatches = op.getQueryType().getRank() - 2;
    for (AffineMap &map : indexingMaps) {
      map = map.shiftDims(numBatches);
      if (map.getNumResults() == 0)
        continue;
      for (int batch : llvm::seq<int>(numBatches)) {
        map = map.insertResult(rewriter.getAffineDimExpr(batch), batch);
      }
    }

    auto attention = rewriter.create<IREE::LinalgExt::AttentionOp>(
        loc, result.getType(), query, key, value, scale, result,
        rewriter.getAffineMapArrayAttr(indexingMaps), optionalMask);

    {
      auto *block = rewriter.createBlock(&attention.getRegion());
      OpBuilder::InsertionGuard g(rewriter);
      block->addArgument(rewriter.getF32Type(), loc);
      rewriter.setInsertionPoint(block, block->begin());

      rewriter.create<IREE::LinalgExt::YieldOp>(loc, block->getArgument(0));
    }

    rewriter.replaceOp(op, attention.getResult(0));
    return success();
  }
};

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class ConvertTMTensorToLinalgExtPass final
    : public impl::ConvertTMTensorToLinalgExtPassBase<
          ConvertTMTensorToLinalgExtPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

#define INSERT_TMTENSOR_CONVERSION_PATTERN(Op)                                 \
  patterns.add<                                                                \
      TMTensorOpConversion<mlir::torch::TMTensor::Op, IREE::LinalgExt::Op>>(   \
      context);

    INSERT_TMTENSOR_CONVERSION_PATTERN(YieldOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScanOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(SortOp);

#undef INSERT_TMTENSOR_CONVERSION_PATTERN

    patterns.add<ScatterOpConversion>(context);
    patterns.add<AttentionOpConversion>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
