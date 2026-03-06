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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
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
  using Base::Base;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indicesTy = op.getIndicesType();
    if (!indicesTy.hasRank()) {
      return failure();
    }

    if (indicesTy.isDynamicDim(indicesTy.getRank() - 1)) {
      return rewriter.notifyMatchFailure(op, "number of indices is unknown");
    }

    auto numIndices = indicesTy.getShape().back();
    llvm::SmallVector<int64_t> dimMap(numIndices);
    for (int i = 0; i < numIndices; i++) {
      dimMap[i] = i;
    }

    auto updatesTy = op.getUpdateType();

    // Create a reassociation that drops all unit dims from the indexed portion
    // slice.
    Value updateVal = op.updates();
    SmallVector<int64_t> collapsedShape;
    collapsedShape.push_back(updatesTy.getShape().front());
    if (op.getUpdateSliceRank() > 0) {
      llvm::append_range(collapsedShape,
                         updatesTy.getShape().take_back(
                             op.getUpdateSliceRank() - op.getIndexDepth()));
    }
    if (collapsedShape != updatesTy.getShape()) {
      auto reassocIndices = getReassociationIndicesForCollapse(
          updatesTy.getShape(), collapsedShape);
      if (!reassocIndices.has_value()) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute reassociation indices");
      }
      updateVal = tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(), updateVal, reassocIndices.value());
    }

    Value indicesVal = op.indices();
    auto scatterOp = IREE::LinalgExt::ScatterOp::create(
        rewriter, op.getLoc(), op->getResultTypes(),
        /*updates=*/updateVal, /*indices=*/indicesVal,
        /*original=*/op.getOutputs()[0], dimMap, op.getUniqueIndices());
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
  using Base::Base;
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
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, query, i));
      }
    }

    if (outputType.getShape().back() == ShapedType::kDynamic) {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, value,
                                               outputType.getRank() - 1));
    }

    Value result =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), dynSizes);

    // Compute scale = 1 / sqrt(headDim), where headDim is the last dimension
    // of the query tensor. When headDim is static, fold to a constant.
    FloatType targetType = cast<FloatType>(op.getQueryType().getElementType());
    int64_t headDim = op.getQueryType().getShape().back();
    Value scale;
    if (headDim != ShapedType::kDynamic) {
      double dk = 1.0 / std::sqrt(static_cast<double>(headDim));
      scale = arith::ConstantOp::create(rewriter, loc, targetType,
                                        rewriter.getFloatAttr(targetType, dk));
    } else {
      int64_t queryRank = op.getQueryType().getRank();
      Value headDimIndex =
          tensor::DimOp::create(rewriter, loc, query, queryRank - 1);
      Value headDimInt = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), headDimIndex);
      Value headDimFloat =
          arith::SIToFPOp::create(rewriter, loc, targetType, headDimInt);
      scale = math::RsqrtOp::create(rewriter, loc, headDimFloat);
    }

    // Add batches to standard attention indexing maps.
    SmallVector<AffineMap> indexingMaps =
        getStandardAttentionIndexingMaps(ctx, optionalMask.has_value());

    int64_t numBatches = op.getQueryType().getRank() - 2;
    for (AffineMap &map : indexingMaps) {
      map = map.shiftDims(numBatches);
      if (map.getNumResults() == 0) {
        continue;
      }
      for (int batch : llvm::seq<int>(numBatches)) {
        map = map.insertResult(rewriter.getAffineDimExpr(batch), batch);
      }
    }

    auto attention = IREE::LinalgExt::AttentionOp::create(
        rewriter, loc, result.getType(), query, key, value, scale, result,
        rewriter.getAffineMapArrayAttr(indexingMaps), optionalMask);

    {
      auto *block = rewriter.createBlock(&attention.getRegion());
      OpBuilder::InsertionGuard g(rewriter);
      block->addArgument(rewriter.getF32Type(), loc);
      rewriter.setInsertionPoint(block, block->begin());

      IREE::LinalgExt::YieldOp::create(rewriter, loc, block->getArgument(0));
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

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
