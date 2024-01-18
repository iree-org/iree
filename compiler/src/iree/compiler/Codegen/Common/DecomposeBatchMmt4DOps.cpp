// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

using IREE::Codegen::LoweringConfigAttr;

/// Returns true if:
///    1. `genericOp` is element-wise with all identity indexing maps
///    2. `genericOp` has only one input and one output with the same shape
static bool isElementWiseIdentity(linalg::GenericOp genericOp) {
  return genericOp.getNumDpsInputs() == 1 && genericOp.getNumDpsInits() == 1 &&
         linalg::isElementwise(genericOp) &&
         llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); });
}

/// Drops the outermost unit dimension of the defining op of `input`, as
/// long as it is a linalg::GenericOp that passes `isElementWiseIdentity`.
/// unit dims are dropped using tensor::InsertSliceOp/tensor::ExtractSliceOp
/// in order to fold with other ops introduced by
/// ConvertBatchMmt4DtoMmt4DPattern
static LogicalResult reduceDefiningOp(PatternRewriter &rewriter, Value input) {
  auto producer = input.getDefiningOp<linalg::GenericOp>();
  if (!producer || !isElementWiseIdentity(producer)) {
    return success();
  }
  linalg::ControlDropUnitDims options;
  options.rankReductionStrategy =
      linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
  options.controlFn = [](Operation *op) -> SmallVector<unsigned> {
    return {0};
  };
  return linalg::dropUnitDims(rewriter, producer, options);
}

/// Drops the first element from all the tile sizes list. The first element is
/// for the batch dimension.
static LoweringConfigAttr
dropBatchTileSize(IREE::Codegen::LoweringConfigAttr config) {
  TileSizesListType tileSizesList = config.getTileSizeVals();
  ScalableTileFlagsListType scalableTileFlagsList =
      config.getScalableTileFlagVals();
  for (auto &tileSizes : tileSizesList) {
    tileSizes.erase(tileSizes.begin());
  }
  for (auto &scalableTileFlags : scalableTileFlagsList) {
    if (!scalableTileFlags.empty()) {
      scalableTileFlags.erase(scalableTileFlags.begin());
    }
  }
  return IREE::Codegen::LoweringConfigAttr::get(
      config.getContext(), tileSizesList, scalableTileFlagsList);
}

/// Pattern to convert linalg.batch_mmt4d with batch dim = 1 into mmt4d.
struct ConvertBatchMmt4DtoMmt4DPattern
    : public OpRewritePattern<linalg::BatchMmt4DOp> {
  using OpRewritePattern<linalg::BatchMmt4DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMmt4DOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getDpsInputOperand(0)->get();
    auto rhs = op.getDpsInputOperand(1)->get();
    auto out = op.getDpsInitOperand(0)->get();

    auto outType = out.getType().cast<RankedTensorType>();
    // Batch dim needs to be tiled to 1 first.
    if (outType.getShape()[0] != 1) {
      return rewriter.notifyMatchFailure(op, "batch dim needs to be 1");
    }
    RankedTensorType reducedOutType =
        RankedTensorType::Builder(outType).dropDim(0);
    Value reducedOut;
    Value initTensor;
    // If the init operand is a linalg.fill op, create a new linalg.fill op with
    // the batch dim dropped, so it is easier to identify fill + mmt4d cases.
    if (auto oldFillOp = out.getDefiningOp<linalg::FillOp>()) {
      initTensor = oldFillOp.output();
      auto newInit = tensor::createCanonicalRankReducingExtractSliceOp(
          rewriter, loc, initTensor, reducedOutType);
      reducedOut =
          rewriter
              .create<linalg::FillOp>(loc, ValueRange{oldFillOp.value()},
                                      ValueRange{newInit})
              .result();

      IREE::Codegen::LoweringConfigAttr loweringConfig =
          getLoweringConfig(oldFillOp);
      if (loweringConfig) {
        auto config = dropBatchTileSize(loweringConfig);
        setLoweringConfig(reducedOut.getDefiningOp(), config);
      }
    } else {
      reducedOut = tensor::createCanonicalRankReducingExtractSliceOp(
          rewriter, loc, out, reducedOutType);
      initTensor = out;
    }

    auto lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType reducedLhsType =
        RankedTensorType::Builder(lhsType).dropDim(0);
    auto reducedLhs = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, lhs, reducedLhsType);
    if (failed(reduceDefiningOp(rewriter, lhs))) {
      return rewriter.notifyMatchFailure(
          lhs.getLoc(), "lhs producer should be reduced, but reduction failed");
    }

    auto rhsType = rhs.getType().cast<RankedTensorType>();
    RankedTensorType reducedRhsType =
        RankedTensorType::Builder(rhsType).dropDim(0);
    auto reducedRhs = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, rhs, reducedRhsType);
    if (failed(reduceDefiningOp(rewriter, rhs))) {
      return rewriter.notifyMatchFailure(
          rhs.getLoc(), "rhs producer should be reduced, but reduction failed");
    }

    auto mmt4DOp = rewriter.create<linalg::Mmt4DOp>(
        loc, reducedOut.getType(), ValueRange{reducedLhs, reducedRhs},
        ValueRange{reducedOut});

    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (loweringConfig) {
      auto config = dropBatchTileSize(loweringConfig);
      setLoweringConfig(mmt4DOp, config);
    }

    auto insertSliceOp = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, mmt4DOp.getResult(0), initTensor);
    rewriter.replaceOp(op, insertSliceOp);
    return success();
  }
};

struct DecomposeBatchMmt4DOpsPass
    : public DecomposeBatchMmt4DOpsBase<DecomposeBatchMmt4DOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void DecomposeBatchMmt4DOpsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();
  Operation *errorOp = nullptr;
  IRRewriter rewriter(ctx);

  WalkResult result =
      funcOp.walk([&](linalg::BatchMmt4DOp batchMmt4DOp) -> WalkResult {
        auto out = batchMmt4DOp.getDpsInitOperand(0)->get();
        auto outType = out.getType().cast<RankedTensorType>();
        // Tile only non unit batch dimensions with tile size equals to 1.
        if (outType.getShape()[0] <= 1) {
          return WalkResult::advance();
        }
        SmallVector<int64_t> tileSizes = {1};
        auto tilingInterfaceOp =
            cast<TilingInterface>(batchMmt4DOp.getOperation());
        auto options = scf::SCFTileAndFuseOptions().setTilingOptions(
            scf::SCFTilingOptions().setTileSizes(
                getAsIndexOpFoldResult(ctx, tileSizes)));
        FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
            scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
                rewriter, tilingInterfaceOp, options);
        if (failed(tileAndFuseResult)) {
          errorOp = batchMmt4DOp;
          return WalkResult::interrupt();
        }
        rewriter.replaceOp(
            batchMmt4DOp,
            tileAndFuseResult->replacements[batchMmt4DOp.getResult(0)]);
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    errorOp->emitOpError("failed to tile the batch dimension");
    return signalPassFailure();
  }

  // Convert linalg.batch_mmt4d with batch dim = 1 into linalg.mmt4d.
  RewritePatternSet patterns(ctx);
  patterns.add<ConvertBatchMmt4DtoMmt4DPattern>(ctx);
  // Canonicalize extract and insert slice ops created during the conversion.
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposeBatchMmt4DOpsPass() {
  return std::make_unique<DecomposeBatchMmt4DOpsPass>();
}

} // namespace mlir::iree_compiler
