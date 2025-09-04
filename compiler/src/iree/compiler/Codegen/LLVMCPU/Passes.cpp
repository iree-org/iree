// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmcpu-pass-pipelines"

namespace mlir::iree_compiler {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<bool> clFailOnOutOfBoundsStackAllocation(
    "iree-llvmcpu-fail-on-out-of-bounds-stack-allocation",
    llvm::cl::desc("fail if the upper bound of dynamic stack allocation cannot "
                   "be solved"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clFailOnLargeVector(
    "iree-llvmcpu-fail-on-large-vector",
    llvm::cl::desc("fail if there are operations with large vectors"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clCheckLinalgVectorization(
    "iree-llvmcpu-check-linalg-vectorization",
    llvm::cl::desc(
        "Runs the pass to check if all the Linalg ops are vectorized"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clUseFastMinMaxOps(
    "iree-llvmcpu-use-fast-min-max-ops",
    llvm::cl::desc(
        "Use `arith.minf/maxf` instead of `arith.minimumf/maximumf` ops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableReassociateFpReductions(
    "iree-llvmcpu-reassociate-fp-reductions",
    llvm::cl::desc("Enables reassociation for FP reductions"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clSkipIntermediateRoundings(
    "iree-llvmcpu-skip-intermediate-roundings",
    llvm::cl::desc(
        "Allow skipping intermediate roundings. For example, in f16 matmul "
        "kernels on targets with only f32 arithmetic, we have to perform each "
        "multiply-accumulate in f32, and if this flag is false, then we have "
        "to round those f32 accumulators to the nearest f16 every time, which "
        "is slow."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clInstrumentMemoryAccesses{
    "iree-llvmcpu-instrument-memory-accesses",
    llvm::cl::desc("Instruments memory accesses in dispatches when dispatch "
                   "instrumentation is enabled."),
    llvm::cl::init(false)};

static llvm::cl::opt<bool> clUseSoftmaxInterFusion(
    "iree-llvmcpu-use-decompose-softmax-fuse",
    llvm::cl::desc("Enables inter-pass fusion for the DecomposeSoftmax pass."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableVectorContractCustomKernels(
    "iree-llvmcpu-enable-vector-contract-custom-kernels",
    llvm::cl::desc("Enables vector contract custom kernels for "
                   "LLVMCPUMmt4dVectorLowering pass."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clTileDispatchUsingForall(
    "iree-llvmcpu-tile-dispatch-using-forall",
    llvm::cl::desc("Enable tile and distribute to workgroups using scf.forall"),
    llvm::cl::init(true));

// By default, IREE does not enable the Armv9-A streaming SVE mode in the
// presence of scalable vectors (even when using `+sme`), as currently there's
// no cost model of when it could be beneficial. This flag will effectively make
// IREE/LLVM switch from SVE to SSVE in dispatch regions with supported
// scalable vector operations.
static llvm::cl::opt<bool> clForceArmStreaming(
    "iree-llvmcpu-force-arm-streaming",
    llvm::cl::desc(
        "Enables Armv9-A streaming SVE mode for any dispatch region that "
        "contains supported scalable vector operations (i.e., use SSVE rather "
        "than SVE). Requires the +sme feature flag."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clPatchFuncOps(
    "iree-llvmcpu-debug-patch-func-ops",
    llvm::cl::desc(
        "Perform the patches on func ops for debugging purpose. It should be "
        "used with `--iree-codegen-debug-patched-func-ops-file-name`."),
    llvm::cl::init(false), llvm::cl::Hidden);

// TODO: Enable `TileDispatchUsingForall` for every pipeline.
static void
addTileAndDistributePasses(OpPassManager &funcPassManager,
                           const LLVMCPUPipelineOptions &pipelineOpt) {
  if (pipelineOpt.disableDistribution) {
    return;
  }
  if (clTileDispatchUsingForall) {
    funcPassManager.addPass(
        createTileAndDistributeToWorkgroupsUsingForallOpPass());
    funcPassManager.addPass(createBufferizeDispatchTensorLoadStorePass());
    funcPassManager.addPass(createCombineLayoutTransformationPass());
  } else {
    funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createConvertToDestinationPassingStylePass());
    funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  }
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
}

//===---------------------------------------------------------------------===//
// Codegen configuration verifications.
//===---------------------------------------------------------------------===//

static bool isValidInterchange(ArrayRef<int64_t> interchange, int numLoops) {
  if (interchange.empty()) {
    return true;
  }
  return isPermutationVector(interchange) && interchange.size() == numLoops;
}

LogicalResult verifyMultiTilingExpertPassPipelineConfig(
    Operation *op, IREE::CPU::LoweringConfigAttr loweringConfig) {

  auto interfaceOp = dyn_cast_or_null<TilingInterface>(op);
  if (!interfaceOp) {
    return success();
  }

  // Collects parallel loops.
  llvm::SmallDenseSet<unsigned> pLoopsSet;
  for (auto [index, iteratorType] :
       llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
    if (iteratorType == utils::IteratorType::parallel) {
      pLoopsSet.insert(index);
    }
  }

  for (int i = 0, e = IREE::CPU::TilingLevel::MaxNumTileLevels; i < e; ++i) {
    if (!loweringConfig.hasTilingLevel(i)) {
      continue;
    }

    auto level = static_cast<IREE::CPU::TilingLevel>(i);
    auto tilingLevelAttr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    switch (level) {
    case IREE::CPU::TilingLevel::DistributionTiles:
    case IREE::CPU::TilingLevel::CacheParallelTiles:
    case IREE::CPU::TilingLevel::VectorCommonParallelTiles:
    case IREE::CPU::TilingLevel::VectorInnerParallelTiles: {
      for (auto [index, tileSize] :
           llvm::enumerate(tilingLevelAttr.getSizes())) {
        if (tileSize != 0 && !pLoopsSet.contains(index)) {
          return op->emitOpError(
                     "expected only parallel dims to be set in the ")
                 << IREE::CPU::getTilingLevelName(level)
                 << " tiling level, but tile size at index (" << index
                 << ") was also set";
        }
      }
      break;
    }
    case IREE::CPU::TilingLevel::CacheReductionTiles:
    case IREE::CPU::TilingLevel::VectorReductionTiles: {
      for (auto [index, tileSize] :
           llvm::enumerate(tilingLevelAttr.getSizes())) {
        if (tileSize != 0 && pLoopsSet.contains(index)) {
          return op->emitOpError(
                     "expected only reduction dims to be set in the ")
                 << IREE::CPU::getTilingLevelName(level)
                 << " tiling level, but tile size at index (" << index
                 << ") was also set";
        }
      }
      break;
    }
    case IREE::CPU::TilingLevel::MaxNumTileLevels:
    case IREE::CPU::TilingLevel::InvalidLevel:
      break;
    };

    ArrayRef<int64_t> interchange = tilingLevelAttr.getInterchange();
    size_t expectedSize = tilingLevelAttr.getSizes().size();
    if (!isValidInterchange(interchange, expectedSize)) {
      return op->emitOpError("expected [0, ")
             << expectedSize << ") to be set exactly once in interchange for "
             << IREE::CPU::getTilingLevelName(level) << " tiling level";
    }
  }

  return success();
}

LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, IREE::CPU::LoweringConfigAttr loweringConfig) {
  if (!isa<linalg::ConvolutionOpInterface>(op)) {
    return success();
  }

  auto getTileSizeAtIndex = [](ArrayRef<int64_t> sizes,
                               ArrayRef<bool> scalableFlags,
                               unsigned index) -> std::pair<int64_t, bool> {
    return std::make_pair(sizes[index],
                          index < scalableFlags.size() && scalableFlags[index]);
  };

  SmallVector<IREE::CPU::TilingLevel> requiredLevels = {
      IREE::CPU::DistributionTiles, IREE::CPU::VectorCommonParallelTiles,
      IREE::CPU::VectorReductionTiles};
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  SmallVector<int64_t> shapeAfterTiling = linalgOp.getStaticLoopRanges();
  for (auto level : requiredLevels) {
    if (!loweringConfig.hasTilingLevel(level)) {
      return op->emitOpError("expected ")
             << IREE::CPU::getTilingLevelName(level) << " is set";
    }
    auto tilingLevelAttr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        loweringConfig.getTilingLevelAttr(level));
    for (size_t i = 0, e = tilingLevelAttr.getSizes().size(); i < e; ++i) {
      auto [size, scalableFlag] = getTileSizeAtIndex(
          tilingLevelAttr.getSizes(), tilingLevelAttr.getScalableFlags(), i);
      if (scalableFlag) {
        shapeAfterTiling[i] = ShapedType::kDynamic;
        continue;
      }
      if (size == 1) {
        shapeAfterTiling[i] = 1;
        continue;
      }
      if (ShapedType::isDynamicShape(shapeAfterTiling[i]) ||
          ShapedType::isDynamic(size) || size == 0) {
        continue;
      }
      if (shapeAfterTiling[i] % size != 0) {
        shapeAfterTiling[i] = ShapedType::kDynamic;
      } else {
        shapeAfterTiling[i] = size;
      }
    }
  }

  int64_t khSize, kwSize, ohSize, owSize;
  auto isSizeExtracted =
      TypeSwitch<Operation *, LogicalResult>(op)
          .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
                linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
                linalg::PoolingNhwcMinUnsignedOp>([&](auto) {
            // shape: N, OH, OW, OC, KH, KW, (IC)
            khSize = shapeAfterTiling[4];
            kwSize = shapeAfterTiling[5];
            ohSize = shapeAfterTiling[1];
            owSize = shapeAfterTiling[2];
            return success();
          })
          .Case<linalg::Conv2DNchwFchwOp>([&](auto) {
            // shape: N, OC, OH, OW, (IC), KH, KW
            khSize = shapeAfterTiling[5];
            kwSize = shapeAfterTiling[6];
            ohSize = shapeAfterTiling[2];
            owSize = shapeAfterTiling[3];
            return success();
          })
          .Case<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>([&](auto) {
            // shape: N, OC, OH, OW, KH, KW
            khSize = shapeAfterTiling[4];
            kwSize = shapeAfterTiling[5];
            ohSize = shapeAfterTiling[2];
            owSize = shapeAfterTiling[3];
            return success();
          })
          .Default([&](auto) { return failure(); });
  if (failed(isSizeExtracted)) {
    return op->emitOpError("unsupported conv types");
  }

  bool removeH = (khSize == 1 && ohSize == 1);
  bool removeW = (kwSize == 1 && owSize == 1);
  if (!removeH && !removeW) {
    return op->emitOpError("can't decompose the conv op");
  }

  return success();
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void buildLLVMCPUVectorLoweringPipeline(
    OpPassManager &funcPassManager,
    const LLVMCPUVectorLoweringPassOptions &options) {
  funcPassManager.addPass(createDropVectorUnitDimsPass());
  funcPassManager.addPass(createLLVMCPUVirtualVectorLoweringPass(
      LLVMCPUVirtualVectorLoweringPassOptions{options.splitVectorTransfersTo,
                                              options.enableArmI8mm}));

  // Make sure we remove redundant vector ops (e.g., vector transposes) before
  // we lower them and can't be optimized away anymore.
  funcPassManager.addPass(createCanonicalizerPass());

  VectorTransferLoweringPassOptions transferLoweringOptions{};
  if (!options.enableArmSME) {
    // The ArmSME dialect has its own (more specific) lowerings for scalable
    // vectors that occur later in the pipeline, so only enable the general
    // lowerings if SME is not available.
    transferLoweringOptions.enableScalableLowerings = true;
  }
  funcPassManager.addPass(
      createVectorTransferLoweringPass(transferLoweringOptions));
  funcPassManager.addPass(createLLVMCPUVectorTransposeLoweringPass(
      LLVMCPUVectorTransposeLoweringPassOptions{
          options.lowerVectorTransposeToAVX2}));

  // Potentially removes shape_cast and broadcast on unit dims before shape_cast
  // lowering.
  funcPassManager.addPass(createCanonicalizerPass());

  // 'vector.shape_cast' are very expensive operations that are even generated
  // by some of the lowerings above (e.g., transpose lowering). There are
  // chances to cancel them out if they are not lowered too early so we lower
  // them at the very end of the pass.
  funcPassManager.addPass(createLLVMCPUVectorShapeCastLoweringPass());
}

void addCPUBufferOpsTileAndVectorizePipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);

  // Skip tiling reduction loops because this is expected to apply on copy ops
  // only.
  funcPassManager.addPass(createLLVMCPUTilePass(
      tilingConfig.getVectorCommonParallelLevel(), /*skipRootOp=*/false));
  funcPassManager.addPass(createLLVMCPUPeelPass());
  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    options.vectorizeGatherAccesses = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    options.enableArmSME = pipelineOpt.enableAArch64SME;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addMultiTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);
  for (int i = 0, e = IREE::CPU::TilingLevel::MaxNumTileLevels; i < e; ++i) {
    auto level = static_cast<IREE::CPU::TilingLevel>(i);
    if (!tilingConfig.isValidLevel(level)) {
      continue;
    }

    switch (level) {
    case IREE::CPU::TilingLevel::CacheParallelTiles:
    case IREE::CPU::TilingLevel::VectorCommonParallelTiles:
      funcPassManager.addPass(
          createLLVMCPUTileRootAndFuseProducerConsumerPass(level));
      break;
    case IREE::CPU::TilingLevel::CacheReductionTiles:
      funcPassManager.addPass(
          createLLVMCPUTileRootAndFuseInputOperandsPass(level));
      break;
    case IREE::CPU::TilingLevel::VectorReductionTiles:
      // Run SplitReductionPass before the final reduction Fuse pass, because
      // SplitReductionPass takes care of banked-tiling.
      funcPassManager.addPass(
          createLLVMCPUSplitReductionPass(clEnableReassociateFpReductions));
      funcPassManager.addPass(
          createLLVMCPUTileRootAndFuseInputOperandsPass(level));
      // Tile all the reduction ops for target vector sizes, which ensures
      // that all the dimensions are tiled in all the reduction ops. The root
      // op is already tiled, so it is skipped in the pass.
      funcPassManager.addPass(createLLVMCPUTilePass(
          static_cast<IREE::CPU::TilingLevel>(i), /*skipRootOp=*/true));
      break;
    case IREE::CPU::TilingLevel::VectorInnerParallelTiles:
    case IREE::CPU::TilingLevel::DistributionTiles:
    case IREE::CPU::TilingLevel::MaxNumTileLevels:
    case IREE::CPU::TilingLevel::InvalidLevel:
      continue;
    };
    funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
    funcPassManager.addPass(createConcretizePadResultShapePass());
  }

  // `VectorInnerParallelTiles` level models the tiling and fusion for the
  // dimensions that are not captured in root op. I.e., root op may not have the
  // config for the level. Thus, we run the LLVMCPUTileAndFuse pass for
  // consumers.
  funcPassManager.addPass(createLLVMCPUTileAndFusePass(
      IREE::CPU::TilingLevel::VectorInnerParallelTiles));
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  funcPassManager.addPass(createForallToForPass());
  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  if (pipelineOpt.enableAArch64SME) {
    funcPassManager.addPass(createLLVMCPU2DScalableTo1DScalablePass());
  }

  {
    funcPassManager.addPass(createTensorToVectorVectorizePadPass());
    if (pipelineOpt.decomposePackUnPackOps) {
      funcPassManager.addPass(createDecomposePackUnPackOpsPass());
      funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
      funcPassManager.addPass(createCSEPass());
    }

    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addCPUBufferizePasses(funcPassManager);

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    options.enableArmSME = pipelineOpt.enableAArch64SME;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addConvTileAndDecomposeExpertPassPipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);

  funcPassManager.addPass(createLLVMCPUTileRootAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  // Convert forall to for before vectorization preparation.
  funcPassManager.addPass(iree_compiler::createForallToForPass());

  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  {
    funcPassManager.addPass(createTensorToVectorVectorizePadPass());
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  // Eliminate redundant transfer_read/write to avoid stack allocations.
  funcPassManager.addPass(createOptimizeVectorTransferPass(
      OptimizeVectorTransferPassOptions{/*flatten=*/true}));

  addCPUBufferizePasses(funcPassManager);

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "shuffle";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    options.enableArmSME = pipelineOpt.enableAArch64SME;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addMmt4dTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);

  funcPassManager.addPass(createLLVMCPUTileRootAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  // The below two passes are nop if the "mmt4d" is explicitly excluded in the
  // ukernels attribute.
  funcPassManager.addPass(createCPUPrepareUkernelsPass());
  funcPassManager.addPass(
      createCPULowerToUKernelsPass(clSkipIntermediateRoundings));
  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(iree_compiler::createForallToForPass());

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addCPUBufferizePasses(funcPassManager);

  // Vector lowering of Mmt4d.
  funcPassManager.addPass(createLLVMCPUMmt4dVectorLoweringPass(
      LLVMCPUMmt4dVectorLoweringPassOptions{
          clEnableVectorContractCustomKernels}));

  // Generic vector lowering.
  LLVMCPUVectorLoweringPassOptions options;
  options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
  options.splitVectorTransfersTo = "linalg-copy";
  options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
  options.enableArmSME = pipelineOpt.enableAArch64SME;
  buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
}

void addCPUDataTilingPipeline(OpPassManager &funcPassManager,
                              TilingConfig &tilingConfig,
                              LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);

  // The below two passes are nop if pack/unpack is not specified in ukernels
  // attribute. By default, they are disabled.
  funcPassManager.addPass(createCPUPrepareUkernelsPass());
  funcPassManager.addPass(
      createCPULowerToUKernelsPass(clSkipIntermediateRoundings));

  funcPassManager.addPass(createLLVMCPUTilePass(
      tilingConfig.getVectorCommonParallelLevel(), /*skipRootOp=*/false));
  if (pipelineOpt.decomposePackUnPackOps) {
    funcPassManager.addPass(createDecomposePackUnPackOpsPass());
  }

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.vectorizePadding = true;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addCPUBufferizePasses(funcPassManager);

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    options.enableArmSME = pipelineOpt.enableAArch64SME;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addCPULinalgExtTileAndVectorizePipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager, pipelineOpt);
  funcPassManager.addPass(createLLVMCPUTileRootAndFuseProducerConsumerPass(
      IREE::CPU::TilingLevel::VectorCommonParallelTiles));
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());
  funcPassManager.addPass(createLLVMCPUTileRootAndFuseInputOperandsPass(
      IREE::CPU::TilingLevel::VectorReductionTiles));
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(iree_compiler::createForallToForPass());

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
    options.enableVectorMasking = pipelineOpt.enableVectorMasking;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    if (clFailOnLargeVector) {
      funcPassManager.addPass(createLLVMCPUVerifyVectorSizeLegalityPass());
    }
  }

  addCPUBufferizePasses(funcPassManager);

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    options.enableArmSME = pipelineOpt.enableAArch64SME;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addCPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               std::unique_ptr<TilingConfig> &tilingConfig,
                               LLVMCPUPipelineOptions &pipelineOpt) {
  if (tilingConfig && tilingConfig->getNumTilingLevels() > 1) {
    addTileAndDistributePasses(funcPassManager, pipelineOpt);
    funcPassManager.addPass(createLLVMCPUTileAndFusePass(
        tilingConfig->getVectorCommonParallelLevel()));
  }
  addCPUBufferizePasses(funcPassManager);
}

static void addLowerToLLVMPasses(OpPassManager &modulePassManager,
                                 bool enableAArch64SME) {
  // TODO: Remove the following pass and plumb support for #hal.descriptor_type
  // memory space through the stack.
  FunctionLikeNest(modulePassManager)
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  // Lower `ukernel.*` ops to function calls
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  FunctionLikeNest(modulePassManager)
      // LinalgExt -> SCF
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPredicatedPass(clCheckLinalgVectorization,
                         createLLVMCPUEmitVectorizationRemarksPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor-type constants.
  modulePassManager.addPass(createIREEBufferizeConstantsPass());

  FunctionLikeNest(modulePassManager)
      .addPass(createFoldTensorExtractOpPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Math dialect ops rewrites, approximations, casts.
      .addPass(createMathTransformPass)
      .addPass(createHoistStaticallyBoundAllocationsPass)
      // Use `arith.minf/maxf` instead of `arith.minimumf/maximumf`.
      .addPredicatedPass(clUseFastMinMaxOps, createReplaceSlowMinMaxOpsPass);

  if (enableAArch64SME) {
    modulePassManager.addPass(mlir::arm_sme::createVectorLegalizationPass());
    FunctionLikeNest(modulePassManager)
        .addPredicatedPass(
            clForceArmStreaming,
            [] {
              // 1. Enable Armv9-A streaming mode without ZA (i.e., SSVE) for
              // dispatch regions that contain scalable vectors when forced via
              // the --iree-llvmcpu-force-arm-streaming flag.
              return mlir::arm_sme::createEnableArmStreamingPass(
                  mlir::arm_sme::ArmStreamingMode::StreamingLocally,
                  mlir::arm_sme::ArmZaMode::Disabled,
                  /*ifRequiredByOps=*/false,
                  /*ifContainsScalableVectors=*/true);
            })
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(mlir::createArithToArmSMEConversionPass)
        .addPass(mlir::createConvertVectorToArmSMEPass)
        .addPass([] {
          // 2. Enable ZA for dispatch regions that contain ArmSME ops (which
          // all make use of the ZA state).
          return mlir::arm_sme::createEnableArmStreamingPass(
              mlir::arm_sme::ArmStreamingMode::StreamingLocally,
              mlir::arm_sme::ArmZaMode::NewZA,
              /*ifRequiredByOps=*/true);
        })
        .addPass(mlir::createConvertArmSMEToSCFPass);
  }

  VectorTransferLoweringPassOptions transferLoweringOptions;
  if (!enableAArch64SME) {
    // The ArmSME dialect has its own (more specific) lowerings for scalable
    // vectors that occur later in the pipeline, so only enable the general
    // lowerings if SME is not available.
    transferLoweringOptions.enableScalableLowerings = true;
  }

  FunctionLikeNest(modulePassManager)
      // All structural buffer manipulations must conclude before this point.

      // The subview folding doesn't like potentially-out-of-bounds
      // vector.transfer_read and vector.transfer_write, lower them to loads and
      // stores here.
      .addPass([&]() {
        return createVectorTransferLoweringPass(transferLoweringOptions);
      })
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(createIREEExpandStridedMetadataPass)
      .addPass(createCleanupBufferAllocViewPass)
      // Checking stack allocation before converting to CF dialect is easier.
      .addPass([&]() {
        return createLLVMCPUCheckIRBeforeLLVMConversionPass(
            LLVMCPUCheckIRBeforeLLVMConversionPassOptions{
                clFailOnOutOfBoundsStackAllocation});
      })
      // SCF -> CF
      .addPass(createSCFToControlFlowPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // (HAL, IREE, Linalg, CF) -> LLVM
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass([&]() {
        arith::ArithExpandOpsPassOptions options;
        options.includeBf16 = true;
        options.includeF4E2M1 = true;
        options.includeF8E8M0 = true;
        return arith::createArithExpandOpsPass(options);
      })
      .addPass(createEmulateNarrowTypePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPredicatedPass(clInstrumentMemoryAccesses,
                         createInstrumentMemoryAccessesPass);

  if (enableAArch64SME) {
    FunctionLikeNest(modulePassManager).addPass([&] {
      return createConvertArmSMEToLLVMPass();
    });
  }
  modulePassManager.addPass(
      createConvertToLLVMPass(clEnableReassociateFpReductions));
  modulePassManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());
  modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
      createAddFastMathFlagsPass());
}

void buildLLVMCPUCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager,
                                                 clUseSoftmaxInterFusion);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createRematerializeParallelOpsPass)
      // TODO(#13888): This(createExpandF16OpToF32Pass()) pass is being added
      // way to late and should insted be be done during lowering to LLVM.
      .addPass(createExpandF16OpToF32Pass)
      .addPass(createMaterializeDeviceEncodingPass)
      .addPass(createConvertAccGEMMToGEMMPass)
      // TODO: Remove the following pass the plumb support for
      // #hal.descriptor_type memory space through the stack.
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);

  modulePassManager.addPass(createLLVMCPUSelectLoweringStrategyPass());
  LLVM_DEBUG({
    llvm::dbgs() << "LLVMCPU codegen configuration pass pipeline:\n";
    modulePassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

void buildLLVMCPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildLLVMCPUCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool enableAArch64SME) {

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createLLVMCPULowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
    if (clPatchFuncOps) {
      modulePassManager.addPass(createPatchFuncOpsPass());
    }
  }

  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createLowerAffinePass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass());

  // Run conversion to LLVM at `ModuleOp` granularity.
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    addLowerToLLVMPasses(modulePassManager, enableAArch64SME);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "LLVMCPU codegen pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildLLVMCPULinkingPassPipeline(OpPassManager &modulePassManager,
                                     std::optional<std::string> target) {
  // Link together executables. This may produce some IR duplication.
  LLVMCPULinkExecutablesPassOptions linkOptions;
  linkOptions.target = target.value_or("");
  modulePassManager.addPass(createLLVMCPULinkExecutablesPass(linkOptions));

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant and import ordinals.
  auto &variantPassManager = modulePassManager.nest<IREE::HAL::ExecutableOp>()
                                 .nest<IREE::HAL::ExecutableVariantOp>();
  variantPassManager.addPass(createLLVMCPUAssignConstantOrdinalsPass());
  variantPassManager.addPass(createLLVMCPUAssignImportOrdinalsPass());
}

//===---------------------------------------------------------------------===//
// Register LLVMCPU Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"
} // namespace

void registerCodegenLLVMCPUPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> LLVMCPUConfigPipeline(
      "iree-codegen-llvmcpu-configuration-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for CPU",
      [](OpPassManager &modulePassManager) {
        buildLLVMCPUCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LLVMCPUBufferizationPipeline(
      "iree-codegen-llvmcpu-bufferization-pipeline",
      "Runs the bufferization pipeline for CPU",
      [](OpPassManager &funcPassManager) {
        addCPUBufferizePasses(funcPassManager);
      });

  static PassPipelineRegistration<> LLVMCPUVectorLoweringPipeline(
      "iree-codegen-llvmcpu-vector-lowering-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for CPU",
      [](OpPassManager &funcPassManager) {
        LLVMCPUVectorLoweringPassOptions options;
        options.splitVectorTransfersTo = "linalg-copy";
        buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
      });

  struct LinalgToLLVMPipelineOptions
      : public PassPipelineOptions<LinalgToLLVMPipelineOptions> {
    Option<bool> enableArmSME{
        *this, "enable-arm-sme",
        llvm::cl::desc("Enable the ArmSME lowering pipeline.")};
  };

  static PassPipelineRegistration<LinalgToLLVMPipelineOptions>
      LinalgLLVMPipeline(
          "iree-codegen-linalg-to-llvm-pipeline",
          "Runs the progressive lowering pipeline from Linalg to LLVM",
          [](OpPassManager &variantPassManager,
             LinalgToLLVMPipelineOptions const &options) {
            buildLLVMCPUCodegenPassPipeline(variantPassManager,
                                            options.enableArmSME);
          });

  static PassPipelineRegistration<> LLVMCPULinkingPipeline(
      "iree-codegen-llvmcpu-linking-pipeline",
      "Runs the LLVMCPU HAL executable linking pipeline",
      [](OpPassManager &modulePassManager) {
        buildLLVMCPULinkingPassPipeline(modulePassManager);
      });
}

} // namespace mlir::iree_compiler
