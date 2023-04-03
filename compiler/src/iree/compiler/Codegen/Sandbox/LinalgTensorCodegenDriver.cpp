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

//===----------------------------------------------------------------------===//
// From Sandbox
//===----------------------------------------------------------------------===//

namespace {

struct LinalgSingleTilingExpertPass
    : public LinalgSingleTilingExpertBase<LinalgSingleTilingExpertPass> {
  LinalgSingleTilingExpertPass() = default;
  LinalgSingleTilingExpertPass(
      const LinalgSingleTilingExpertPassOptions &options) {
    this->anchorFuncOpName = options.anchorFuncOpName;
    this->anchorOpName = options.anchorOpName;
    this->generalize = options.generalize;
    this->iteratorInterchange = options.iteratorInterchange;
    this->vectorize = options.vectorize;
    this->enableVectorMasking = options.enableVectorMasking;
    this->vectorizePadding = options.vectorizePadding;
    this->vectorizeGatherAccesses = options.vectorizeGatherAccesses;
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

void LinalgSingleTilingExpertPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  LinalgVectorizationOptions vectorizationOptions;
  vectorizationOptions.setVectorizePadding(vectorizePadding);
  vectorizationOptions.setVectorizeGatherAccesses(vectorizeGatherAccesses);
  vectorizationOptions.setEnableVectorMasking(enableVectorMasking);
  if (enableVectorMasking) {
    vectorizationOptions.setCanonicalVectorSizes(
        getCanonicalVectorShape(funcOp));
    vectorizationOptions.setVectorSizeComputationFunction(getVectorSizes);
  }

  CodegenStrategy strategy;
  StringRef genericOpName = linalg::GenericOp::getOperationName();
  strategy.vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName,
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
