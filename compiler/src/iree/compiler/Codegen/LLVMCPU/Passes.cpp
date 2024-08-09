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
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

static void addTileAndDistributePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createConvertToDestinationPassingStylePass());
  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());
}

//===---------------------------------------------------------------------===//
// Codegen configuration verifications.
//===---------------------------------------------------------------------===//

static bool isValidInterchange(ArrayRef<int64_t> interchange, int numLoops) {
  if (interchange.empty())
    return true;
  llvm::SmallDenseSet<int64_t> s;
  s.insert(interchange.begin(), interchange.end());
  for (int i = 0; i < numLoops; ++i) {
    if (!s.contains(i))
      return false;
  }
  return true;
}

LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, TilingConfig &tilingConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  if (!workgroupSize.empty()) {
    return op->emitOpError(
        "expected workgroup size to be empty for CPU pipelines");
  }

  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                CPUDoubleTilingExpert);
  }

  if (tilingConfig.getNumTilingLevels() == 6) {
    // TODO: update verification.
    return success();
  }

  if (tilingConfig.getNumTilingLevels() != 4) {
    return op->emitOpError("expected four tiling levels, got ")
           << tilingConfig.getNumTilingLevels();
  }

  auto interfaceOp = dyn_cast_or_null<TilingInterface>(op);
  if (interfaceOp) {
    llvm::SmallDenseSet<unsigned> pLoopsSet;
    for (auto [index, iteratorType] :
         llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
      if (iteratorType == utils::IteratorType::parallel) {
        pLoopsSet.insert(index);
      }
    }

    SmallVector<int64_t> secondLevelTileSizes;
    std::tie(secondLevelTileSizes, std::ignore) =
        tilingConfig.getVectorCommonParallelSizes();
    for (auto [index, tileSize] : llvm::enumerate(secondLevelTileSizes)) {
      if (tileSize != 0 && !pLoopsSet.contains(index)) {
        return op->emitOpError(
                   "expected only parallel dims to be set in the second tiling "
                   "level, got ")
               << index << "-th tile size set";
      }
    }

    SmallVector<int64_t> thirdLevelTileSizes;
    std::tie(thirdLevelTileSizes, std::ignore) =
        tilingConfig.getVectorReductionSizes();
    for (auto [index, tileSize] : llvm::enumerate(thirdLevelTileSizes)) {
      if (tileSize != 0 && pLoopsSet.contains(index)) {
        return op->emitOpError(
                   "expected only reduction dims to be set in the third tiling "
                   "level, got ")
               << index << "-th tile size set";
      }
    }
  }

  // Verify interchange
  auto tileSizesForLevel = tilingConfig.getTileSizes();
  for (int level = 0; level < tilingConfig.getNumTilingLevels(); level++) {
    auto interchange = tilingConfig.getTileInterchangeSizes(level);
    auto &tileSizes = tileSizesForLevel[level];
    if (!isValidInterchange(interchange, tileSizes.size())) {
      return op->emitOpError("expected [0, ")
             << tileSizes.size() << ") to be set exactly once in interchange #"
             << level;
    }
  }

  // Verify that native vector size is empty.
  SmallVector<int64_t> nativeVectorSize = tilingConfig.getNativeVectorSizes();
  if (!nativeVectorSize.empty()) {
    return op->emitOpError("native_vector_size must be empty");
  }
  return success();
}

LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, TilingConfig &tilingConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  if (!isa<linalg::ConvolutionOpInterface>(op))
    return success();

  if (tilingConfig.getNumTilingLevels() == 6) {
    // TODO: update verification.
    return success();
  }

  if (tilingConfig.getNumTilingLevels() != 4) {
    return op->emitOpError("expected four tiling levels, got ")
           << tilingConfig.getNumTilingLevels();
  }

  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  SmallVector<int64_t> shape = linalgOp.getStaticLoopRanges();
  for (auto sizes : tilingConfig.getTileSizes()) {
    for (auto [i, size] : llvm::enumerate(sizes)) {
      if (size == 1)
        shape[i] = 1;
      if (shape[i] == -1 || size == 0)
        continue;
      if (shape[i] % size != 0) {
        shape[i] = -1;
      } else {
        shape[i] = size;
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
            // Shape: N, OH, OW, OC, KH, KW, (IC)
            khSize = shape[4];
            kwSize = shape[5];
            ohSize = shape[1];
            owSize = shape[2];
            return success();
          })
          .Case<linalg::Conv2DNchwFchwOp>([&](auto) {
            // Shape: N, OC, OH, OW, (IC), KH, KW
            khSize = shape[5];
            kwSize = shape[6];
            ohSize = shape[2];
            owSize = shape[3];
            return success();
          })
          .Case<linalg::PoolingNchwSumOp, linalg::PoolingNchwMaxOp>([&](auto) {
            // Shape: N, OC, OH, OW, KH, KW
            khSize = shape[4];
            kwSize = shape[5];
            ohSize = shape[2];
            owSize = shape[3];
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
  funcPassManager.addPass(createLLVMCPUDropVectorUnitDimsPass());
  funcPassManager.addPass(createLLVMCPUVirtualVectorLoweringPass(
      LLVMCPUVirtualVectorLoweringPassOptions{options.splitVectorTransfersTo,
                                              options.enableArmI8mm}));

  // Make sure we remove redundant vector ops (e.g., vector tranposes) before we
  // lower them and can't be optimized away anymore.
  funcPassManager.addPass(createCanonicalizerPass());

  funcPassManager.addPass(createLLVMCPUVectorTransferLoweringPass());
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
  addTileAndDistributePasses(funcPassManager);

  // Skip tiling reduction loops because this is expected to apply on copy ops
  // only.
  funcPassManager.addPass(
      createLLVMCPUTilePass(tilingConfig.getVectorCommonParallelLevel()));
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
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addMultiTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager);

  SmallVector<int64_t> allFusableLevels(tilingConfig.getFusableLevels());
  // Apply tile and fuse to all the non-distribution fusable levels. Skip
  // distribution level as that level has been fused already.
  if (allFusableLevels.size() > 1) {
    llvm::SmallSetVector<int64_t, 4> fusableLevels(allFusableLevels.begin(),
                                                   allFusableLevels.end());
    for (int i = 0; i < tilingConfig.getNumTilingLevels(); ++i) {
      if (i == tilingConfig.getDistributionLevel())
        continue;
      if (fusableLevels.contains(i)) {
        funcPassManager.addPass(createLLVMCPUTileAndFusePass(i));
        funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
        funcPassManager.addPass(createConcretizePadResultShapePass());
        continue;
      }

      if (i == tilingConfig.getVectorReductionLevel()) {
        // Run SplitReductionPass before the final reduction Fuse pass, because
        // SplitReductionPass takes care of banked-tiling.
        funcPassManager.addPass(
            createLLVMCPUSplitReductionPass(clEnableReassociateFpReductions));
        funcPassManager.addPass(createLLVMCPUTilePass(i));
        continue;
      }

      funcPassManager.addPass(createLLVMCPUTilePass(i));
    }
  }

  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  if (pipelineOpt.enableAArch64SSVE) {
    funcPassManager.addPass(createLLVMCPU2DScalableTo1DScalablePass());
  }

  {
    funcPassManager.addPass(createVectorizePadPass());
    if (pipelineOpt.decomposePackUnPackOps) {
      funcPassManager.addPass(createDecomposePackUnPackOpsPass());
      funcPassManager.addPass(createCanonicalizerPass());
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
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addConvTileAndDecomposeExpertPassPipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager);

  // Run LLVMTileAndFuse firstly in case that we have fill + conv + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is conv + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.

  funcPassManager.addPass(createLLVMCPUTileAndFusePass(
      tilingConfig.getVectorCommonParallelLevel()));
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  funcPassManager.addPass(
      createLLVMCPUTilePass(tilingConfig.getVectorReductionLevel()));
  funcPassManager.addPass(
      createLLVMCPUTileAndFusePass(tilingConfig.getVectorInnerParallelLevel()));
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());

  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());

  if (pipelineOpt.enablePeeling) {
    funcPassManager.addPass(createLLVMCPUPeelPass());
  }

  {
    funcPassManager.addPass(createVectorizePadPass());
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
  funcPassManager.addPass(createOptimizeVectorTransferPass(/*flatten=*/true));

  addCPUBufferizePasses(funcPassManager);

  // Run IREE specific passes before vector lowering expert.
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  {
    LLVMCPUVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = pipelineOpt.lowerToAVX2;
    options.splitVectorTransfersTo = "shuffle";
    options.enableArmI8mm = pipelineOpt.enableAArch64I8mm;
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addMmt4dTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager);

  funcPassManager.addPass(createLLVMCPUTileAndFusePass(
      static_cast<int64_t>(tilingConfig.getVectorCommonParallelLevel())));
  // The below two passes are nop if the "mmt4d" is explicitly excluded in the
  // ukernels attribute.
  funcPassManager.addPass(createCPUPrepareUkernelsPass());
  funcPassManager.addPass(
      createCPULowerToUKernelsPass(clSkipIntermediateRoundings));
  funcPassManager.addPass(createLLVMCPUTilePass(
      static_cast<int64_t>(tilingConfig.getVectorReductionLevel())));

  {
    GenericVectorizationPassOptions options;
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
  buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
}

void addCPUDataTilingPipeline(OpPassManager &funcPassManager,
                              TilingConfig &tilingConfig,
                              LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager);

  // The below two passes are nop if pack/unpack is not specified in ukernels
  // attribute. By default, they are disabled.
  funcPassManager.addPass(createCPUPrepareUkernelsPass());
  funcPassManager.addPass(
      createCPULowerToUKernelsPass(clSkipIntermediateRoundings));

  funcPassManager.addPass(
      createLLVMCPUTilePass(tilingConfig.getVectorCommonParallelLevel()));
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
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addCPULinalgExtTileAndVectorizePipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt) {
  addTileAndDistributePasses(funcPassManager);
  funcPassManager.addPass(
      createLLVMCPUTilePass(tilingConfig.getVectorCommonParallelLevel()));
  // TODO: Remove the pass once we have PartialReductionOpInterface implemented
  // for AttentionOp.
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());
  funcPassManager.addPass(
      createLLVMCPUTilePass(tilingConfig.getVectorReductionLevel()));
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());

  {
    GenericVectorizationPassOptions options;
    options.useConfiguredVectorSizes = pipelineOpt.useConfiguredVectorSizes;
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
    buildLLVMCPUVectorLoweringPipeline(funcPassManager, options);
  }
}

void addCPUDefaultPassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributePasses(funcPassManager);
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
  addConstantBufferizePasses(modulePassManager);

  FunctionLikeNest(modulePassManager)
      .addPass(createFoldTensorExtractOpPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // math dialect elementry functions -> polynomial form.
      .addPass(createPolynomialApproximationPass)
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

  FunctionLikeNest(modulePassManager)
      // Resolve get_buffer_descriptor ops. All structural buffer manipulations
      // must conclude before this point.
      .addPass(createIREEExpandStridedMetadataPass)
      .addPass(createCleanupBufferAllocViewPass)
      // Checking stack allocation before converting to CF dialect is easier.
      .addPass([&]() {
        return createLLVMCPUCheckIRBeforeLLVMConversionPass(
            LLVMCPUCheckIRBeforeLLVMConversionPassOptions{
                clFailOnOutOfBoundsStackAllocation});
      })
      // SCF -> CF
      .addPass(createConvertSCFToCFPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // (HAL, IREE, Linalg, CF) -> LLVM
      .addPass(arith::createArithExpandOpsPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
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
      .addPass(createCPUMaterializeDeviceEncodingPass)
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
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildLLVMCPUCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool enableAArch64SME) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createLLVMCPULowerExecutableTargetPass);

  // Run conversion to LLVM at `ModuleOp` granularity.
  addLowerToLLVMPasses(modulePassManager, enableAArch64SME);
  LLVM_DEBUG({
    llvm::dbgs() << "LLVMCPU codegen pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildLLVMCPULinkingPassPipeline(OpPassManager &modulePassManager) {
  // Link together executables. This may produce some IR duplication.
  modulePassManager.addPass(createLLVMCPULinkExecutablesPass());

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
        buildLLVMCPUCodegenConfigurationPassPipeline(modulePassManager);
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
