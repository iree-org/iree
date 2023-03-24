// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <optional>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/CodegenStrategy.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Sandbox/PassDetail.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
// using namespace mlir::linalg;

using mlir::iree_compiler::IREE::LinalgExt::CodegenStrategy;
using mlir::iree_compiler::IREE::LinalgExt::LinalgTransformationFilter;
using mlir::iree_compiler::IREE::LinalgExt::LinalgTransforms;
using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorizationOptions;
using mlir::iree_compiler::IREE::LinalgExt::LinalgVectorLoweringOptions;

#define DEBUG_TYPE "iree-linalg-tensor-codegen-driver"

//===----------------------------------------------------------------------===//
// IREE specific functions
//===----------------------------------------------------------------------===//

/// Returns the op that contains lowering config. Checks whether the provided op
/// contains the lowering config and returns it. Otherwise, tries to find the
/// lowering config across the function. If there are multiple ops with the same
/// lowering configs, returns the first one found. Returns failure if there are
/// multiple op with different lowering config.
static FailureOr<Operation *> getRootOp(Operation *op) {
  // Check for self first.
  if (iree_compiler::getLoweringConfig(op)) {
    return op;
  }

  // Get the function op.
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    funcOp = op->getParentOfType<func::FuncOp>();
  }

  assert(funcOp && "Missing funcOp");

  Operation *rootOp = nullptr;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr rootLoweringConfig;
  auto result = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = iree_compiler::getLoweringConfig(op);
    if (!loweringConfig) {
      return WalkResult::advance();
    }
    if (rootLoweringConfig) {
      if (rootLoweringConfig != loweringConfig) {
        return WalkResult::interrupt();
      }
    } else {
      rootOp = op;
      rootLoweringConfig = loweringConfig;
    }
    return WalkResult::advance();
  });

  if (!rootOp || result.wasInterrupted()) {
    return failure();
  }
  return rootOp;
}

/// Builds a proper tile sizes vector for the op.
/// scf::tileUsingSCFForOp expects the num of tile sizes = num of loops. This
/// method returns a proper tile sizes vector for each op during tiling.
static SmallVector<Value> buildTileSizesForOp(OpBuilder &b, Operation *op,
                                              ArrayRef<int64_t> tileSizes) {
  auto tilingOp = cast<TilingInterface>(op);

  SmallVector<int64_t> newTileSizes(tileSizes);
  newTileSizes.resize(tilingOp.getLoopIteratorTypes().size(), /*default=*/0);

  OpBuilder::InsertionGuard guard(b);
  return llvm::to_vector(map_range(newTileSizes, [&](int64_t size) {
    Value v = b.create<arith::ConstantIndexOp>(tilingOp->getLoc(), size);
    return v;
  }));
}

/// Default method to initialize the tiling options in IREE. These could be
/// overriden by the command line options if specified. For now the sentinel
/// -1 is used for avoiding querying the lowering config.
static bool getTilingOptionsFromConfig(func::FuncOp funcOp, int64_t tilingLevel,
                                       scf::SCFTilingOptions &tilingOptions) {
  if (tilingLevel != -1) {
    FailureOr<Operation *> rootOp = getRootOp(funcOp);
    if (failed(rootOp)) {
      return false;
    }
    SmallVector<int64_t> tileSizes =
        mlir::iree_compiler::getTileSizes(rootOp.value(), tilingLevel);
    if (llvm::all_of(tileSizes, [](int v) { return v == 0; })) return false;
    tilingOptions.setTileSizeComputationFunction(
        [tileSizes](OpBuilder &b, Operation *op) {
          return buildTileSizesForOp(b, op, tileSizes);
        });
    return true;
  }
  return false;
}

/// Computes the canonical shape used to vectorize this dispatch. Retrieves
/// the vectorization tile sizes (parallel and reduction levels) out of the
/// lowering config and adjusts them to the format expected by the Linalg
/// vectorizer.
static SmallVector<int64_t> getCanonicalVectorShape(func::FuncOp funcOp) {
  FailureOr<Operation *> rootOp = getRootOp(funcOp);
  if (failed(rootOp)) {
    return {};
  }

  unsigned numTileLevels =
      mlir::iree_compiler::getNumTileLevels(rootOp.value());
  if (numTileLevels < 3) {
    return {};
  }

  // Retrieve the tile sizes from the last two tiling levels (parallel and
  // reduction) used for vectorization.
  SmallVector<int64_t> canonicalVectorShape =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 2);
  SmallVector<int64_t> reductionTileSizes =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 1);

  if (!reductionTileSizes.empty()) {
    assert(canonicalVectorShape.size() == reductionTileSizes.size() &&
           "Unexpected tile sizes");

    // Combine the reduction tile sizes with the parallel tile sizes already in
    // the canonical vector shape.
    for (int i = 0, end = canonicalVectorShape.size(); i < end; ++i) {
      if (reductionTileSizes[i] > 0)
        canonicalVectorShape[i] = reductionTileSizes[i];
    }
  }

  // Replace zeros in canonical vector shape to turn it into a valid shape.
  std::replace(canonicalVectorShape.begin(), canonicalVectorShape.end(), 0, 1);
  return canonicalVectorShape;
}

// Give the canonical vector shape of a dispatch, returns the vector sizes for a
// particular linalg op within that dispatch.
static SmallVector<int64_t> getVectorSizes(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> canonicalVectorShape) {
  FailureOr<Operation *> rootOp = getRootOp(linalgOp);
  if (failed(rootOp)) {
    return {};
  }

  // TODO: Infer the tiles sizes for an op that is not the root op.
  if (*rootOp != linalgOp.getOperation()) {
    return {};
  }

  if (canonicalVectorShape.empty()) {
    return {};
  }

  assert(canonicalVectorShape.size() >= linalgOp.getNumLoops() &&
         "Unexpected canonical vector shape or number of loops");

  // Return the valid canonical vector shape subset based on the number of loops
  // of the linalg op.
  SmallVector<int64_t> vecSize(
      canonicalVectorShape.take_front(linalgOp.getNumLoops()));
  for (auto [idx, val] : llvm::enumerate(linalgOp.getStaticLoopRanges())) {
    if (ShapedType::isDynamic(val)) continue;
    vecSize[idx] = std::max(vecSize[idx], val);
  }

  return vecSize;
}

/// Default method to get tile sizes for tile-and-fuse in IREE. These could be
/// ovveridden by the command line options if specified.
static LogicalResult getTileAndFuseOptionsFromConfig(
    func::FuncOp funcOp, int64_t tilingLevel,
    SmallVector<int64_t> &tileAndFuseSizes, SmallVector<int64_t> &tileOnlySizes,
    SmallVector<int64_t> &tileInterchange) {
  if (tilingLevel == -1) {
    return success();
  }

  FailureOr<Operation *> rootOp = getRootOp(funcOp);
  if (failed(rootOp) || !rootOp.value()) return failure();
  auto tilingOp = cast<TilingInterface>(rootOp.value());

  iree_compiler::IREE::Codegen::LoweringConfigAttr loweringConfig =
      iree_compiler::getLoweringConfig(tilingOp);
  SmallVector<int64_t> tileSizes = loweringConfig.getTileSizeVals(tilingLevel);

  auto iteratorTypes = tilingOp.getLoopIteratorTypes();
  tileAndFuseSizes = SmallVector<int64_t>(iteratorTypes.size(), /*default=*/0);
  tileOnlySizes = SmallVector<int64_t>(iteratorTypes.size(), /*default=*/0);
  // Splits the tileSizes into two sizes vectors. We want to tile-and-fuse on
  // parallel dims and tile-only on reduction dims.
  for (size_t i = 0; i < tileSizes.size() && i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::parallel) {
      tileAndFuseSizes[i] = tileSizes[i];
    } else {
      tileOnlySizes[i] = tileSizes[i];
    }
  }

  tileInterchange = loweringConfig.getTileInterchangeVals(tilingLevel);

  return success();
}

/// Default method to initialize the split reduction size in IREE. These could
/// be overriden by the command line options if specified.
static FailureOr<int64_t> getSplitReductionSizeFromConfig(func::FuncOp funcOp) {
  FailureOr<Operation *> rootOp = getRootOp(funcOp);
  if (failed(rootOp)) {
    return failure();
  }
  unsigned numTileLevels =
      mlir::iree_compiler::getNumTileLevels(rootOp.value());
  assert(numTileLevels >= 1 && "at least 1 tiling level must be present");

  // The last one is the reduction dimension.
  auto reductionSizes =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 1);
  if (reductionSizes.size() == 0) {
    return failure();
  }
  return reductionSizes[reductionSizes.size() - 1];
}

//===----------------------------------------------------------------------===//
// From Sandbox
//===----------------------------------------------------------------------===//

namespace {

struct LinalgSplitReductionPass
    : public LinalgSplitReductionBase<LinalgSplitReductionPass> {
  LinalgSplitReductionPass(bool enableFpReductionReordering, int64_t size = 0) {
    this->size.setValue(size);
    fpReductionReordering = enableFpReductionReordering;
  }
  void runOnOperation() override;

 private:
  bool fpReductionReordering = false;
};

struct LinalgSingleTilingExpertPass
    : public LinalgSingleTilingExpertBase<LinalgSingleTilingExpertPass> {
  LinalgSingleTilingExpertPass() = default;
  LinalgSingleTilingExpertPass(
      const LinalgSingleTilingExpertPassOptions &options) {
    this->anchorFuncOpName = options.anchorFuncOpName;
    this->anchorOpName = options.anchorOpName;
    this->tileSizes = options.tileSizes;
    this->tileInterchange = options.tileInterchange;
    this->generalize = options.generalize;
    this->iteratorInterchange = options.iteratorInterchange;
    this->decomposeToLowerDimOp = options.decomposeToLowerDimOp;
    this->vectorize = options.vectorize;
    this->enableVectorMasking = options.enableVectorMasking;
    this->vectorizePadding = options.vectorizePadding;
    this->tilingLevel = options.tilingLevel;
  }
  LinalgSingleTilingExpertPass(const LinalgSingleTilingExpertPass &pass) {}

  /// Function pass entry point.
  void runOnOperation() override;
};

struct LinalgVectorLoweringPass
    : public LinalgVectorLoweringBase<LinalgVectorLoweringPass> {
  LinalgVectorLoweringPass(int64_t vectorLoweringStage = 0) {
    this->vectorLoweringStage.setValue(vectorLoweringStage);
  }
  LinalgVectorLoweringPass(const LinalgVectorLoweringPass &pass) {}
  LinalgVectorLoweringPass(const LinalgVectorLoweringPassOptions &options) {
    this->vectorLoweringStage = options.vectorLoweringStage;
    this->splitVectorTransfersTo = options.splitVectorTransfersTo;
    this->lowerVectorTransposeTo = options.lowerVectorTransposeTo;
    this->lowerVectorTransposeToAVX2 = options.lowerVectorTransposeToAVX2;
    this->lowerVectorMultiReductionTo = options.lowerVectorMultiReductionTo;
    this->lowerVectorContractionTo = options.lowerVectorContractionTo;
    this->unrollVectorTransfers = options.unrollVectorTransfers;
    this->maxTransferRank = options.maxTransferRank;
  }

  void runOnOperation() override;
};
}  // namespace

/// Pattern to wrap splitReduction transformation.
struct CodegenSplitReduction
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  CodegenSplitReduction(MLIRContext *context, bool fpReductionReordering,
                        int64_t size, LinalgTransformationFilter filter,
                        PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        fpReductionReordering(fpReductionReordering),
        size(size),
        filter(std::move(filter)) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // Make sure that
    // - the pass has not been applied before
    // - has tensor semantics
    // - number of reduction loops == 1
    // - has exactly 1 output
    // - index map has only projected permutations
    // - is a linalg generic op
    // - has exactly 1 input
    // - if enableReductionReordering is not set, then operand is an int
    // - innermost dimension of the input operand is reduction
    // TODO: support named ops, numInputs > 1, and modify lastDim check below
    // accordingly. If fpReductionReordering is not enabled by default, it must
    // be an integer or index type to proceed to allow associative reordering.
    if (failed(filter.checkAndNotify(rewriter, op))) {
      return rewriter.notifyMatchFailure(op, "pass has been applied before");
    }
    if (!op.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(op, "doesn't have tensor semantics");
    }
    if (op.getNumReductionLoops() != 1) {
      return rewriter.notifyMatchFailure(op, "number of reduction loops != 1");
    }
    if (op.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(op, "doesn't have exactly 1 output");
    }
    if (!op.hasOnlyProjectedPermutations()) {
      return rewriter.notifyMatchFailure(
          op, "index map doesn't have only projected permutations");
    }
    if (!isa<linalg::GenericOp>(op)) {
      return rewriter.notifyMatchFailure(op, "is not a generic op");
    }
    if (op.getNumDpsInputs() != 1) {
      return rewriter.notifyMatchFailure(op, "doesn't have exactly 1 input");
    }
    auto elementType = op.getDpsInputOperand(0)
                           ->get()
                           .getType()
                           .dyn_cast<ShapedType>()
                           .getElementType();
    if (!(fpReductionReordering || elementType.isIntOrIndex())) {
      return rewriter.notifyMatchFailure(
          op, "enable reordering is not set and operand is not an int");
    }

    SmallVector<unsigned> dims;
    op.getReductionDims(dims);
    AffineMap map = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
    unsigned lastIdx = map.getNumResults() - 1;
    unsigned lastDim = map.getDimPosition(lastIdx);
    if (lastDim != dims[0]) {
      return rewriter.notifyMatchFailure(
          op, "innermost dimension of the input operand is not reduction");
    }

    linalg::ControlSplitReductionFn fn = [this, lastIdx](linalg::LinalgOp op) {
      return linalg::SplitReductionOptions{size, lastIdx,
                                           /*innerParallel=*/true};
    };

    auto numLoops = op.getNumLoops();

    // 1) Tile to extract a single vector-length array.
    SmallVector<int64_t> tileSizesSVFirst(numLoops, 1);
    tileSizesSVFirst[numLoops - 1] = 0;
    auto optionsFirst = scf::SCFTilingOptions().setTileSizes(tileSizesSVFirst);
    FailureOr<scf::SCFTilingResult> tileResFirst = scf::tileUsingSCFForOp(
        rewriter, cast<TilingInterface>(op.getOperation()), optionsFirst);
    if (failed(tileResFirst)) return failure();
    rewriter.replaceOp(op, tileResFirst->replacements);
    for (auto tiledOp : tileResFirst->tiledOps) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    }

    // 2) Apply splitReduction on the single vector-length array. splitReduction
    // already replaces the op.
    FailureOr<linalg::SplitReductionResult> splitRes =
        splitReduction(rewriter, tileResFirst->tiledOps.back(), fn);
    if (failed(splitRes)) return failure();
    filter.replaceLinalgTransformationFilter(rewriter, splitRes->splitLinalgOp);
    filter.replaceLinalgTransformationFilter(rewriter,
                                             splitRes->resultCombiningLinalgOp);

    // 3) Tile the first op generated by splitReduction with tile size of 1, to
    // essentially create a reduction loop.
    // Note that splitRes->splitLinalgOp.getNumLoops() = numLoops + 1.
    SmallVector<int64_t> tileSizesSV(splitRes->splitLinalgOp.getNumLoops(), 0);
    // The reduction happens only in the penultimate dimension, which we now
    // tile.
    tileSizesSV[numLoops - 1] = 1;
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizesSV);
    FailureOr<scf::SCFTilingResult> tileRes = scf::tileUsingSCFForOp(
        rewriter, cast<TilingInterface>(splitRes->splitLinalgOp.getOperation()),
        options);
    if (failed(tileRes)) return failure();
    rewriter.replaceOp(splitRes->splitLinalgOp, tileRes->replacements);

    return success();
  }

 private:
  bool fpReductionReordering;
  int64_t size;
  LinalgTransformationFilter filter;
};

void LinalgSplitReductionPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  int64_t useSize = size.getValue();
  if (useSize == 0) {
    auto splitReductionOptions = getSplitReductionSizeFromConfig(funcOp);
    if (failed(splitReductionOptions)) {
      return;
    }
    useSize = *splitReductionOptions;
  }
  RewritePatternSet patterns(&getContext());
  patterns.add<CodegenSplitReduction>(
      &getContext(), fpReductionReordering, useSize,
      LinalgTransformationFilter(
          ArrayRef<StringAttr>{},
          StringAttr::get(&getContext(), "CODEGEN_SPLIT")));

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
  // Remove all the markers at the end.
  funcOp->walk([&](linalg::LinalgOp op) {
    op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
  });
}

void LinalgSingleTilingExpertPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  scf::SCFTilingOptions tilingOptions;
  bool doTiling =
      getTilingOptionsFromConfig(funcOp, tilingLevel, tilingOptions);
  if (!tileSizes.empty()) {
    doTiling = true;
    SmallVector<int64_t> clonedTileSizes = llvm::to_vector(tileSizes);
    tilingOptions.setTileSizeComputationFunction(
        [clonedTileSizes](OpBuilder &b, Operation *op) {
          return buildTileSizesForOp(b, op, clonedTileSizes);
        });
  }
  if (!tileInterchange.empty()) {
    tilingOptions = tilingOptions.setInterchange(tileInterchange);
  }

  LinalgVectorizationOptions vectorizationOptions;
  vectorizationOptions.setVectorizePadding(vectorizePadding);
  vectorizationOptions.setEnableVectorMasking(enableVectorMasking);
  if (enableVectorMasking) {
    vectorizationOptions.setCanonicalVectorSizes(
        getCanonicalVectorShape(funcOp));
    vectorizationOptions.setVectorSizeComputationFunction(getVectorSizes);
  }

  CodegenStrategy strategy;
  StringRef genericOpName = linalg::GenericOp::getOperationName();
  strategy.tileIf(doTiling, anchorOpName, tilingOptions)
      .decomposeIf(decomposeToLowerDimOp)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName,
                   vectorizationOptions);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(func::FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  dynamicPM.addPass(
      iree_compiler::IREE::LinalgExt::createLinalgStrategyEnablePass());
  if (failed(runPipeline(dynamicPM, funcOp))) {
    return signalPassFailure();
  }
}

void LinalgVectorLoweringPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\n ---- Stage : " << vectorLoweringStage;);
  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          lowerVectorTransposeTo.getValue())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
          .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
          .Default(vector::VectorTransposeLowering::EltWise);
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      llvm::StringSwitch<vector::VectorMultiReductionLowering>(
          lowerVectorMultiReductionTo.getValue())
          .Case("innerreduction",
                vector::VectorMultiReductionLowering::InnerReduction)
          .Default(vector::VectorMultiReductionLowering::InnerParallel);
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          lowerVectorContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Per-function lowering pipeline.
  vector::VectorTransformsOptions vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);
  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll(unrollVectorTransfers);

  LinalgVectorLoweringOptions vectorLoweringOptions =
      LinalgVectorLoweringOptions()
          // Lowering of vector contractions.
          .enableContractionLowering(vectorLoweringStage >= 0)
          // Lowering of vector multi_reduction.
          .enableMultiReductionLowering(vectorLoweringStage >= 1)
          // Whether to split full/partial vector.transfer ops.
          .enableTransferPartialRewrite(vectorLoweringStage >= 2 &&
                                        vectorTransferSplit !=
                                            vector::VectorTransferSplit::None)

          // Set the maximum vector load / store rank.
          .setMaxTransferRank(maxTransferRank)
          // Lower vector.transfer to vector.transfer of max rank.
          .enableTransferLowering(vectorLoweringStage >= 3)
          // Conversion to scf.
          .enableTransferToSCFConversion(vectorLoweringStage >= 4)
          .setVectorTransferToSCFOptions(vectorTransferToSCFOptions)
          // Lowering of vector.transpose.
          .enableVectorTransposeLowering(vectorLoweringStage >= 5)
          .setVectorTransformsOptions(vectorTransformOptions)
          .enableAVX2Lowering(lowerVectorTransposeToAVX2)
          .setAVX2LoweringOptions(
              x86vector::avx2::LoweringOptions().setTransposeOptions(
                  x86vector::avx2::TransposeLoweringOptions()
                      .lower4x8xf32(lowerVectorTransposeToAVX2)
                      .lower8x8xf32(lowerVectorTransposeToAVX2)))
          // Lowering of vector.shape_cast.
          .enableShapeCastLowering(vectorLoweringStage >= 6);

  CodegenStrategy strategy;
  strategy.vectorLowering(vectorLoweringOptions);
  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(func::FuncOp::getOperationName());
  func::FuncOp funcOp = getOperation();
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  dynamicPM.addPass(
      iree_compiler::IREE::LinalgExt::createLinalgStrategyEnablePass());
  if (failed(runPipeline(dynamicPM, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgSplitReductionPass(const bool enableFpReductionReordering,
                                     const int64_t size) {
  return std::make_unique<LinalgSplitReductionPass>(enableFpReductionReordering,
                                                    size);
}
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgSingleTilingExpertPass() {
  return std::make_unique<LinalgSingleTilingExpertPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgSingleTilingExpertPass(
    const LinalgSingleTilingExpertPassOptions &passOptions) {
  return std::make_unique<LinalgSingleTilingExpertPass>(passOptions);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgVectorLoweringPass(int64_t vectorLoweringStage) {
  return std::make_unique<LinalgVectorLoweringPass>(vectorLoweringStage);
}
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgVectorLoweringPass(
    const LinalgVectorLoweringPassOptions &options) {
  return std::make_unique<LinalgVectorLoweringPass>(options);
}

//===----------------------------------------------------------------------===//
// Transforms
//===----------------------------------------------------------------------===//

void mlir::addLowerToVectorTransforms(OpPassManager &passManager,
                                      LinalgVectorLoweringPassOptions options) {
  for (int i = 0; i < 7; ++i) {
    options.vectorLoweringStage = i;
    passManager.addPass(createLinalgVectorLoweringPass(options));
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
  }
}

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Sandbox/Passes.h.inc"
}  // namespace mlir

void mlir::iree_compiler::registerSandboxPasses() { registerPasses(); }
