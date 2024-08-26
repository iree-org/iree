// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_CPUPREPAREUKERNELSPASS
#include "iree/compiler/Codegen/Common/CPU/Passes.h.inc"

namespace {

using IREE::Codegen::LoweringConfigAttr;

static void tileBatchDimsForBatchMmt4dOp(RewriterBase &rewriter,
                                         FunctionOpInterface funcOp) {
  funcOp.walk([&](linalg::BatchMmt4DOp batchMmt4DOp) {
    auto out = batchMmt4DOp.getDpsInitOperand(0)->get();
    auto outType = cast<RankedTensorType>(out.getType());
    // Tile only non unit batch dimensions with tile size equals to 1.
    if (outType.getShape()[0] <= 1) {
      return;
    }
    SmallVector<OpFoldResult> tileSizes = {rewriter.getIndexAttr(1)};
    auto tilingInterfaceOp = cast<TilingInterface>(batchMmt4DOp.getOperation());
    auto options = scf::SCFTileAndFuseOptions().setTilingOptions(
        scf::SCFTilingOptions().setTileSizes(tileSizes));
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                  options);
    assert(succeeded(tileAndFuseResult));
    rewriter.replaceOp(
        batchMmt4DOp,
        tileAndFuseResult->replacements[batchMmt4DOp.getResult(0)]);
  });
}

static void tileNonPackedDimsFor3DPackOps(RewriterBase &rewriter,
                                          FunctionOpInterface funcOp) {
  funcOp.walk([&](tensor::PackOp packOp) {
    if (packOp.getSourceRank() != 3 || packOp.getDestRank() != 5) {
      return;
    }

    OpFoldResult zero = rewriter.getIndexAttr(0);
    OpFoldResult one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> tileSizes(packOp.getSourceRank(), one);
    for (auto dim : packOp.getInnerDimsPos()) {
      tileSizes[dim] = zero;
    }

    // Skip the tiling if the size is already 1.
    RankedTensorType srcType = packOp.getSourceType();
    for (auto [idx, val] : llvm::enumerate(tileSizes)) {
      if (val && srcType.getDimSize(idx) == 1)
        return;
    }

    auto outerDimsPerm = packOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector(tileSizes, outerDimsPerm);
    }

    auto tilingInterfaceOp = cast<TilingInterface>(packOp.getOperation());
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
    assert(succeeded(tilingResult));
    rewriter.replaceOp(packOp, tilingResult->replacements);
  });
}

static void tileNonPackedDimsFor5DPUnpackOps(RewriterBase &rewriter,
                                             FunctionOpInterface funcOp) {
  funcOp.walk([&](tensor::UnPackOp unpackOp) {
    if (unpackOp.getSourceRank() != 5 || unpackOp.getDestRank() != 3) {
      return;
    }

    OpFoldResult zero = rewriter.getIndexAttr(0);
    OpFoldResult one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> tileSizes(unpackOp.getDestRank(), one);

    for (auto dim : unpackOp.getInnerDimsPos()) {
      tileSizes[dim] = zero;
    }

    // Skip the tiling if the size is already 1.
    RankedTensorType destType = unpackOp.getDestType();
    for (auto [idx, val] : llvm::enumerate(tileSizes)) {
      if (val && destType.getDimSize(idx) == 1)
        return;
    }

    auto tilingInterfaceOp = cast<TilingInterface>(unpackOp.getOperation());
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
    auto outerDimsPerm = unpackOp.getOuterDimsPerm();
    if (!outerDimsPerm.empty()) {
      options.setInterchange(outerDimsPerm);
    }
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);
    assert(succeeded(tilingResult));
    rewriter.replaceOp(unpackOp, tilingResult->replacements);
  });
}

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
  FailureOr<linalg::DropUnitDimsResult> result =
      linalg::dropUnitDims(rewriter, producer, options);
  if (failed(result)) {
    return failure();
  }
  rewriter.replaceOp(producer, result->replacements);
  return success();
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

    auto outType = cast<RankedTensorType>(out.getType());
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

      auto loweringConfig =
          getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(oldFillOp);
      if (loweringConfig) {
        auto config = dropBatchTileSize(loweringConfig);
        setLoweringConfig(reducedOut.getDefiningOp(), config);
      }
    } else {
      reducedOut = tensor::createCanonicalRankReducingExtractSliceOp(
          rewriter, loc, out, reducedOutType);
      initTensor = out;
    }

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    RankedTensorType reducedLhsType =
        RankedTensorType::Builder(lhsType).dropDim(0);
    auto reducedLhs = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, lhs, reducedLhsType);
    if (failed(reduceDefiningOp(rewriter, lhs))) {
      return rewriter.notifyMatchFailure(
          lhs.getLoc(), "lhs producer should be reduced, but reduction failed");
    }

    auto rhsType = cast<RankedTensorType>(rhs.getType());
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

    auto loweringConfig =
        getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op);
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

struct Convert3DPackto2DPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getSourceRank() != 3 || packOp.getDestRank() != 5) {
      return failure();
    }

    int64_t srcPos = 0;
    llvm::SmallDenseSet<int64_t> s;
    s.insert(packOp.getInnerDimsPos().begin(), packOp.getInnerDimsPos().end());
    for (auto dim : llvm::seq<int64_t>(0, packOp.getSourceRank())) {
      if (s.contains(dim))
        continue;
      srcPos = dim;
      break;
    }

    int destPos = srcPos;
    for (auto [idx, val] : llvm::enumerate(packOp.getOuterDimsPerm())) {
      if (val == srcPos)
        destPos = idx;
    }

    if (packOp.getSourceType().getDimSize(srcPos) != 1) {
      return rewriter.notifyMatchFailure(packOp, "srcPos != 1");
    }
    if (packOp.getDestType().getDimSize(destPos) != 1) {
      return rewriter.notifyMatchFailure(packOp, "destPos != 1");
    }

    SmallVector<int64_t> newInnerDimsPos(packOp.getInnerDimsPos());
    for (auto &val : newInnerDimsPos) {
      assert(val != srcPos);
      if (val > srcPos)
        val--;
    }
    SmallVector<int64_t> newOuterDimsPerm(packOp.getOuterDimsPerm());
    if (!newOuterDimsPerm.empty()) {
      newOuterDimsPerm.erase(newOuterDimsPerm.begin() + destPos);
      for (auto &val : newOuterDimsPerm) {
        if (val > srcPos)
          val--;
      }
    }

    Location loc = packOp.getLoc();
    auto reducedSrcType =
        RankedTensorType::Builder(packOp.getSourceType()).dropDim(srcPos);
    auto reducedSrc = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, packOp.getSource(), reducedSrcType);

    auto reducedDestType =
        RankedTensorType::Builder(packOp.getDestType()).dropDim(destPos);
    auto reducedDest = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, packOp.getDest(), reducedDestType);

    auto newPackOp = rewriter.create<tensor::PackOp>(
        loc, reducedSrc, reducedDest, newInnerDimsPos, packOp.getMixedTiles(),
        packOp.getPaddingValue(), newOuterDimsPerm);

    auto insertSliceOp = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, newPackOp.getResult(), packOp.getDest());
    rewriter.replaceOp(packOp, insertSliceOp);
    return success();
  }
};

struct Convert5DUnPackto4DUnPackPattern
    : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp unpackOp,
                                PatternRewriter &rewriter) const override {
    if (unpackOp.getSourceRank() != 5 || unpackOp.getDestRank() != 3) {
      return failure();
    }

    llvm::SmallDenseSet<int64_t> s(unpackOp.getInnerDimsPos().begin(),
                                   unpackOp.getInnerDimsPos().end());

    SmallVector<int64_t> seqOrOuterDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, unpackOp.getDestRank()));
    if (!unpackOp.getOuterDimsPerm().empty()) {
      applyPermutationToVector(seqOrOuterDimsPerm, unpackOp.getOuterDimsPerm());
    }

    int64_t srcPos = 0;
    int64_t destPos = 0;

    for (auto [idx, val] : llvm::enumerate(seqOrOuterDimsPerm)) {
      if (s.contains(val))
        continue;
      srcPos = idx;
      destPos = val;
      break;
    }

    if (unpackOp.getSourceType().getDimSize(srcPos) != 1) {
      return rewriter.notifyMatchFailure(unpackOp, "srcPos != 1");
    }

    if (unpackOp.getDestType().getDimSize(destPos) != 1) {
      return rewriter.notifyMatchFailure(unpackOp, "destPos != 1");
    }

    // Calculate the new innerDimsPos and outerDimsPerm after removal of the
    // unit non packed/tiled dimension.
    SmallVector<int64_t> newInnerDimsPos(unpackOp.getInnerDimsPos());
    for (auto &val : newInnerDimsPos) {
      assert(val != destPos);
      if (val > destPos)
        val--;
    }

    SmallVector<int64_t> newOuterDimsPerm(unpackOp.getOuterDimsPerm());
    if (!newOuterDimsPerm.empty()) {
      newOuterDimsPerm.erase(newOuterDimsPerm.begin() + srcPos);
      for (auto &val : newOuterDimsPerm) {
        if (val > destPos)
          val--;
      }
    }

    Location loc = unpackOp.getLoc();
    auto reducedSrcType =
        RankedTensorType::Builder(unpackOp.getSourceType()).dropDim(srcPos);
    auto reducedSrc = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, unpackOp.getSource(), reducedSrcType);

    auto reducedDestType =
        RankedTensorType::Builder(unpackOp.getDestType()).dropDim(destPos);
    auto reducedDest = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, unpackOp.getDest(), reducedDestType);

    auto newUnpackOp = rewriter.create<tensor::UnPackOp>(
        loc, reducedSrc, reducedDest, newInnerDimsPos, unpackOp.getMixedTiles(),
        newOuterDimsPerm);

    auto insertSliceOp = tensor::createCanonicalRankReducingInsertSliceOp(
        rewriter, loc, newUnpackOp.getResult(), unpackOp.getDest());
    rewriter.replaceOp(unpackOp, insertSliceOp);
    return success();
  }
};

struct CPUPrepareUkernelsPass
    : public impl::CPUPrepareUkernelsPassBase<CPUPrepareUkernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

} // namespace

void CPUPrepareUkernelsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  auto funcOp = getOperation();
  IRRewriter rewriter(ctx);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);

  if (hasUkernel(targetAttr, "mmt4d")) {
    tileBatchDimsForBatchMmt4dOp(rewriter, funcOp);
    patterns.add<ConvertBatchMmt4DtoMmt4DPattern>(ctx);
  }
  if (hasUkernel(targetAttr, "pack")) {
    tileNonPackedDimsFor3DPackOps(rewriter, funcOp);
    patterns.add<Convert3DPackto2DPackPattern>(ctx);
  }
  if (hasUkernel(targetAttr, "unpack")) {
    tileNonPackedDimsFor5DPUnpackOps(rewriter, funcOp);
    patterns.add<Convert5DUnPackto4DUnPackPattern>(ctx);
  }

  // Canonicalize extract and insert slice ops created during the conversion.
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::populateFoldTensorSubsetOpPatterns(patterns);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
