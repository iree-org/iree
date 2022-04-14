// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Sandbox/PassDetail.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "iree-linalg-tensor-codegen-driver"

//===----------------------------------------------------------------------===//
// IREE specific functions
//===----------------------------------------------------------------------===//

/// Default method to initialize the tiling options in IREE. These could be
/// overriden by the command line options if specified. For now the sentinel
/// -1 is used for avoiding querying the lowering config.
static bool getTilingOptionsFromConfig(int64_t tilingLevel,
                                       LinalgTilingOptions &tilingOptions) {
  if (tilingLevel != -1) {
    tilingOptions.setTileSizeComputationFunction(
        [tilingLevel](OpBuilder &builder,
                      Operation *operation) -> SmallVector<Value, 4> {
          return ::mlir::iree_compiler::getTileSizes(builder, operation,
                                                     tilingLevel);
        });
    return true;
  }
  return false;
}

/// Default method to initialize the tiling options for fusion in IREE. These
/// could be ovveridden by the command line options if specified.
static FailureOr<LinalgTilingAndFusionOptions> getTileAndFuseOptionsFromConfig(
    func::FuncOp funcOp, int64_t tilingLevel) {
  SmallVector<Operation *> computeOps;
  SmallVector<mlir::iree_compiler::LoopTilingAndDistributionInfo> tiledLoops;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr loweringConfig;
  if (tilingLevel != -1 &&
      succeeded(getComputeOps(funcOp, computeOps, tiledLoops))) {
    for (auto op : computeOps) {
      if (auto currLoweringConfig = iree_compiler::getLoweringConfig(op)) {
        if (loweringConfig) {
          return LogicalResult(funcOp.emitOpError(
              "unhandled multiple lowering configurations in compute ops"));
        }
        loweringConfig = currLoweringConfig;
      }
    }
  }
  if (!loweringConfig) {
    return LinalgTilingAndFusionOptions();
  }
  LinalgTilingAndFusionOptions options;
  options.tileSizes.assign(loweringConfig.getTileSizeVals(tilingLevel));
  options.tileInterchange.assign(
      loweringConfig.getTileInterchangeVals(tilingLevel));
  return options;
}

//===----------------------------------------------------------------------===//
// From Sandbox
//===----------------------------------------------------------------------===//

namespace {

static void getAtMostNEnclosingLoops(
    Operation *op, int64_t nLoops,
    SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = op->getParentOp();
  while (nLoops-- > 0 &&
         (outermostEnclosingForOp = dyn_cast<scf::ForOp>(nextEnclosingOp))) {
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingOp = outermostEnclosingForOp->getParentOp();
  }
}

struct LinalgFusePass : public LinalgFuseBase<LinalgFusePass> {
  LinalgFusePass(int64_t tilingLevel = -1, bool vectorize = false) {
    this->tilingLevel.setValue(tilingLevel);
    this->vectorize.setValue(vectorize);
  }
  LinalgFusePass(const LinalgFusePass &pass) {}
  LinalgFusePass(const LinalgFusePassOptions &options) {
    this->anchorFuncOpName = options.anchorFuncOpName;
    this->anchorOpName = options.anchorOpName;
    this->tileSizes = options.tileSizes;
    this->tileInterchange = options.tileInterchange;
    this->pad = options.pad;
    this->paddingValues = options.paddingValues;
    this->paddingDimensions = options.paddingDimensions;
    this->packPaddings = options.packPaddings;
    this->hoistPaddings = options.hoistPaddings;
    this->transposePaddings = options.transposePaddings;
    this->vectorize = options.vectorize;
    this->vectorizePadding = options.vectorizePadding;
    this->tilingLevel = options.tilingLevel;
    this->doIREEDistribution = options.doIREEDistribution;
  }
  void runOnOperation() override;
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
    this->peeledLoops = options.peeledLoops;
    this->pad = options.pad;
    this->paddingValues = options.paddingValues;
    this->packPaddings = options.packPaddings;
    this->hoistPaddings = options.hoistPaddings;
    this->transposePaddings = options.transposePaddings;
    this->packPaddings = options.packPaddings;
    this->scalarizeDynamicDims = options.scalarizeDynamicDims;
    this->generalize = options.generalize;
    this->iteratorInterchange = options.iteratorInterchange;
    this->decomposeToLowerDimOp = options.decomposeToLowerDimOp;
    this->vectorize = options.vectorize;
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

struct UnrollOneVectorOpPass
    : public UnrollOneVectorOpBase<UnrollOneVectorOpPass> {
  UnrollOneVectorOpPass() = default;
  UnrollOneVectorOpPass(const UnrollOneVectorOpPass &pass) {}
  void runOnOperation() override;
};

struct UnrollOneParentLoopPass
    : public UnrollOneParentLoopBase<UnrollOneParentLoopPass> {
  UnrollOneParentLoopPass() = default;
  UnrollOneParentLoopPass(const UnrollOneParentLoopPass &pass) {}
  void runOnOperation() override;
};

struct OutlineOneParentLoopPass
    : public OutlineOneParentLoopBase<OutlineOneParentLoopPass> {
  OutlineOneParentLoopPass() = default;
  OutlineOneParentLoopPass(const OutlineOneParentLoopPass &pass) {}
  void runOnOperation() override;
};
}  // namespace

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgFusePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  FailureOr<LinalgTilingAndFusionOptions> defaultTilingOptions =
      getTileAndFuseOptionsFromConfig(funcOp, tilingLevel);
  if (failed(defaultTilingOptions)) {
    return signalPassFailure();
  }
  LinalgTilingAndFusionOptions tilingOptions = defaultTilingOptions.getValue();
  bool doTiling = !tilingOptions.tileSizes.empty();
  if (!tileSizes.empty()) {
    doTiling = true;
    tilingOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
  }
  if (!tileInterchange.empty()) {
    tilingOptions.tileInterchange = {tileInterchange.begin(),
                                     tileInterchange.end()};
  }
  if (doIREEDistribution) {
    tilingOptions.setDistributionOptions(
        ::mlir::iree_compiler::getIREELinalgLoopDistributionOptions());
  }

  // Parse the padding values.
  SmallVector<Attribute> paddingValueAttributes;
  for (const std::string &paddingValue : paddingValues) {
    paddingValueAttributes.push_back(
        parseAttribute(paddingValue, &getContext()));
  }

  // Set up padding options.
  SmallVector<SmallVector<int64_t>> transposePaddingVectors;
  for (const std::string &transposePadding : transposePaddings) {
    SmallVector<int64_t> transposeVector = {};
    SmallVector<StringRef> tokens;
    StringRef(transposePadding).split(tokens, ':');
    for (StringRef token : tokens)
      transposeVector.push_back(std::stoi(token.str()));
    transposePaddingVectors.push_back(transposeVector);
  }

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValueAttributes);
  paddingOptions.setPaddingDimensions(
      SmallVector<int64_t>{paddingDimensions.begin(), paddingDimensions.end()});
  paddingOptions.setPackPaddings(
      SmallVector<bool>{packPaddings.begin(), packPaddings.end()});
  paddingOptions.setHoistPaddings(
      SmallVector<int64_t>{hoistPaddings.begin(), hoistPaddings.end()});
  paddingOptions.setTransposePaddings(transposePaddingVectors);

  CodegenStrategy strategy;
  strategy.tileAndFuseIf(doTiling, anchorOpName, tilingOptions)
      .padIf(pad, "", paddingOptions)
      .vectorizeIf(vectorize, "", nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(func::FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());

  if (failed(runPipeline(dynamicPM, funcOp))) {
    return signalPassFailure();
  }
}

void LinalgSingleTilingExpertPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  LinalgTilingOptions tilingOptions;
  bool doTiling = getTilingOptionsFromConfig(tilingLevel, tilingOptions);
  if (!tileSizes.empty()) {
    doTiling = true;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  }
  if (!tileInterchange.empty()) {
    tilingOptions = tilingOptions.setInterchange(
        SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));
  }
  if (scalarizeDynamicDims) {
    doTiling = true;
    tilingOptions = tilingOptions.scalarizeDynamicDims();
  }
  tilingOptions = tilingOptions.setPeeledLoops(peeledLoops);

  // Parse the padding values.
  SmallVector<Attribute> paddingValueAttributes;
  for (const std::string &paddingValue : paddingValues) {
    paddingValueAttributes.push_back(
        parseAttribute(paddingValue, &getContext()));
  }

  // Set up padding options.
  SmallVector<SmallVector<int64_t>> transposePaddingVectors;
  for (const std::string &transposePadding : transposePaddings) {
    SmallVector<int64_t> transposeVector = {};
    SmallVector<StringRef> tokens;
    StringRef(transposePadding).split(tokens, ':');
    for (StringRef token : tokens)
      transposeVector.push_back(std::stoi(token.str()));
    transposePaddingVectors.push_back(transposeVector);
  }

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValueAttributes);
  paddingOptions.setPackPaddings(
      SmallVector<bool>{packPaddings.begin(), packPaddings.end()});
  paddingOptions.setHoistPaddings(
      SmallVector<int64_t>{hoistPaddings.begin(), hoistPaddings.end()});
  paddingOptions.setTransposePaddings(transposePaddingVectors);

  CodegenStrategy strategy;
  StringRef genericOpName = GenericOp::getOperationName();
  strategy.tileIf(doTiling, anchorOpName, tilingOptions)
      .padIf(pad, anchorOpName, paddingOptions)
      .decomposeIf(decomposeToLowerDimOp)
      .generalizeIf(generalize, anchorOpName)
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName,
                   nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(func::FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
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
      VectorTransferToSCFOptions()
          .enableFullUnroll(unrollVectorTransfers)
          .enableLowerPermutationMaps();

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
  if (failed(runPipeline(dynamicPM, funcOp))) {
    return signalPassFailure();
  }
}

void UnrollOneVectorOpPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName) return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShape(targetShape)
                    .setFilterConstraint([&](Operation *op) {
                      auto unrollInterface =
                          dyn_cast<VectorUnrollOpInterface>(op);
                      if (!unrollInterface ||
                          op->getName().getStringRef() != anchorOpName ||
                          !sourceShape.hasValue() ||
                          !unrollInterface.getShapeForUnroll().hasValue())
                        return failure();

                      ArrayRef<int64_t> sourceShapeToMatch{sourceShape};
                      auto shapeForUnroll =
                          unrollInterface.getShapeForUnroll().getValue();
                      ArrayRef<int64_t> actualSourceShape{
                          shapeForUnroll.begin(), shapeForUnroll.end()};
                      return success(sourceShapeToMatch == actualSourceShape);
                    }));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void UnrollOneParentLoopPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName) return;

  // Poor man's op targeting.
  getOperation().walk([&](Operation *op) {
    if (op->getName().getStringRef() != anchorOpName)
      return WalkResult::advance();
    SmallVector<scf::ForOp> reverseEnclosingLoops;
    getAtMostNEnclosingLoops(op, parentLoopNum, reverseEnclosingLoops);
    if (failed(loopUnrollByFactor(reverseEnclosingLoops.back(), unrollFactor)))
      signalPassFailure();
    return WalkResult::interrupt();
  });
}

scf::ExecuteRegionOp outlineInExecuteRegion(RewriterBase &b, Operation *op) {
  if (op->getNumRegions() != 1) return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

void OutlineOneParentLoopPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName) return;

  // Poor man's op targeting.
  getOperation().walk([&](Operation *op) {
    if (op->getName().getStringRef() != anchorOpName)
      return WalkResult::advance();
    SmallVector<scf::ForOp> reverseEnclosingLoops;
    getAtMostNEnclosingLoops(op, parentLoopNum, reverseEnclosingLoops);
    IRRewriter b(op->getContext());
    scf::ExecuteRegionOp exec =
        outlineInExecuteRegion(b, reverseEnclosingLoops.back());
    if (failed(outlineSingleBlockRegion(b, op->getLoc(), exec.getRegion(),
                                        resultFuncName)))
      signalPassFailure();
    return WalkResult::interrupt();
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgFusePass() {
  return std::make_unique<LinalgFusePass>();
}
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgFusePass(
    const mlir::LinalgFusePassOptions &options) {
  return std::make_unique<LinalgFusePass>(options);
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

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createUnrollOneVectorOpPass() {
  return std::make_unique<UnrollOneVectorOpPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createUnrollOneParentLoopPass() {
  return std::make_unique<UnrollOneParentLoopPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createOutlineOneParentLoopPass() {
  return std::make_unique<OutlineOneParentLoopPass>();
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

//===----------------------------------------------------------------------===//
// IREE specific pass creation methods to allow invocation from within IREEs
// backend pipelines
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::iree_compiler::createLinalgFusePass(int64_t tilingLevel, bool vectorize) {
  return std::make_unique<LinalgFusePass>(tilingLevel, vectorize);
}

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Sandbox/Passes.h.inc"
}  // namespace mlir

void mlir::iree_compiler::registerSandboxPasses() { registerPasses(); }
