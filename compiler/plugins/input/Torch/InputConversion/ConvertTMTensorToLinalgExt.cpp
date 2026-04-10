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
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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
struct TMTensorOpConversion : OpRewritePattern<SrcOpTy> {
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
    : OpRewritePattern<mlir::torch::TMTensor::ScatterOp> {
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

// Returns indexing maps for OnlineAttentionOp. Order follows ODS:
// Q, K, V, scale, [mask], output/acc, max, sum.
static SmallVector<AffineMap> getOnlineAttentionIndexingMaps(MLIRContext *ctx,
                                                             bool hasMask) {
  AffineExpr m, n, k1, k2;
  bindDims(ctx, m, n, k1, k2);

  auto qMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k1}, ctx);
  auto kMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, k1}, ctx);
  auto vMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, n}, ctx);
  auto sMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, ctx);
  auto accMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, n}, ctx);
  auto maxMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m}, ctx);
  auto sumMap = maxMap;
  if (hasMask) {
    auto mskMap =
        AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k2}, ctx);
    return {qMap, kMap, vMap, sMap, mskMap, accMap, maxMap, sumMap};
  }
  return {qMap, kMap, vMap, sMap, accMap, maxMap, sumMap};
}

struct AttentionOpConversion
    : OpRewritePattern<mlir::torch::TMTensor::AttentionOp> {
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

    SmallVector<Value> outputDynSizes;
    for (int i = 0, s = outputType.getRank() - 1; i < s; ++i) {
      if (outputType.isDynamicDim(i)) {
        outputDynSizes.push_back(
            tensor::DimOp::create(rewriter, loc, query, i));
      }
    }

    if (outputType.getShape().back() == ShapedType::kDynamic) {
      outputDynSizes.push_back(tensor::DimOp::create(rewriter, loc, value,
                                                     outputType.getRank() - 1));
    }

    Value output =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), outputDynSizes);

    // Compute scale = rsqrt(head_dim) in f32.
    int64_t queryRank = op.getQueryType().getRank();
    Value dimIdx =
        rewriter.createOrFold<tensor::DimOp>(loc, query, queryRank - 1);
    Value dimInt = rewriter.createOrFold<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), dimIdx);
    Value dimFloat = rewriter.createOrFold<arith::SIToFPOp>(
        loc, rewriter.getF32Type(), dimInt);
    Value scale = rewriter.createOrFold<math::RsqrtOp>(loc, dimFloat);

    // Build indexing maps for OnlineAttentionOp and add batch dimensions.
    SmallVector<AffineMap> indexingMaps =
        getOnlineAttentionIndexingMaps(ctx, optionalMask.has_value());

    int64_t numBatches = queryRank - 2;
    for (AffineMap &map : indexingMaps) {
      map = map.shiftDims(numBatches);
      if (map.getNumResults() == 0) {
        continue;
      }
      for (int batch : llvm::seq<int>(numBatches)) {
        map = map.insertResult(rewriter.getAffineDimExpr(batch), batch);
      }
    }

    // Identify the acc and sum maps (last 3 in the list: acc, max, sum).
    int64_t numMaps = indexingMaps.size();
    AffineMap accMap = indexingMaps[numMaps - 3];
    AffineMap sumMap = indexingMaps[numMaps - 1];
    auto queryType = cast<ShapedType>(query.getType());
    SmallVector<OpFoldResult> accSize;
    for (int i = 0; i < outputType.getRank(); ++i) {
      if (outputType.isDynamicDim(i)) {
        accSize.push_back(
            Value(tensor::DimOp::create(rewriter, loc, output, i)));
      } else {
        accSize.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      }
    }

    SmallVector<OpFoldResult> rowRedSize;
    for (int i = 0; i < queryRank - 1; ++i) {
      if (queryType.isDynamicDim(i)) {
        rowRedSize.push_back(
            Value(tensor::DimOp::create(rewriter, loc, query, i)));
      } else {
        rowRedSize.push_back(rewriter.getIndexAttr(queryType.getDimSize(i)));
      }
    }

    // Create fills for acc, max, and sum in f32.
    Type f32Type = rewriter.getF32Type();
    Value accEmpty = tensor::EmptyOp::create(rewriter, loc, accSize, f32Type);
    Value rowRedEmpty =
        tensor::EmptyOp::create(rewriter, loc, rowRedSize, f32Type);

    Value accInit =
        arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, rewriter,
                                loc, /*useOnlyFiniteValue=*/true);
    Value maxInit =
        arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type,
                                rewriter, loc, /*useOnlyFiniteValue=*/true);
    Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                            rewriter, loc);

    Value accFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{accInit}, accEmpty)
            .getResult(0);
    Value maxFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{maxInit}, rowRedEmpty)
            .getResult(0);
    Value sumFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{sumInit}, rowRedEmpty)
            .getResult(0);

    // Create OnlineAttentionOp directly.
    auto onlineAttn = IREE::LinalgExt::OnlineAttentionOp::create(
        rewriter, loc,
        TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
        query, key, value, scale, accFill, maxFill, sumFill,
        rewriter.getAffineMapArrayAttr(indexingMaps), optionalMask);

    {
      OpBuilder::InsertionGuard g(rewriter);
      auto *block = rewriter.createBlock(&onlineAttn.getRegion());
      block->addArgument(rewriter.getF32Type(), loc);
      rewriter.setInsertionPoint(block, block->begin());
      IREE::LinalgExt::YieldOp::create(rewriter, loc, block->getArgument(0));
    }

    Value x = onlineAttn.getResult(0);
    Value sum = onlineAttn.getResult(2);

    // Normalize: result = (1 / sum) * acc.
    SmallVector<AffineMap> compressedMaps =
        compressUnusedDims(SmallVector<AffineMap>{sumMap, accMap, accMap});
    SmallVector<utils::IteratorType> iteratorTypes(
        compressedMaps[0].getNumDims(), utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, outputType, ValueRange{sum, x}, output, compressedMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          // Both sum and acc are f32. Compute (1/sum)*acc, then cast to
          // the output element type.
          Value one = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(args[0].getType(), 1.0));
          Value reciprocal = arith::DivFOp::create(b, loc, one, args[0]);
          Value result = arith::MulFOp::create(b, loc, reciprocal, args[1]);
          result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                        /*isUnsignedCast=*/false);
          linalg::YieldOp::create(b, loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResults());
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
