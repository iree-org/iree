// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<bool> clCheckIRBeforeLLVMConversion(
    "iree-codegen-check-ir-before-llvm-conversion",
    llvm::cl::desc("Runs the pass to check the IR generated from LLVMCPU "
                   "before conversion to LLVM IR"),
    llvm::cl::init(true));

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
//===---------------------------------------------------------------------===//

static Value cpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeDeallocationFn(OpBuilder &builder,
                                                             Location loc,
                                                             Value allocation) {
  return success();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static void addCPUIREEComprehensiveBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);
}

//===---------------------------------------------------------------------===//
// Codegen configuration verifications.
//===---------------------------------------------------------------------===//

static bool isValidInterchange(ArrayRef<int64_t> interchange, int numLoops) {
  if (interchange.empty()) return true;
  llvm::SmallDenseSet<int64_t> s;
  s.insert(interchange.begin(), interchange.end());
  for (int i = 0; i < numLoops; ++i) {
    if (!s.contains(i)) return false;
  }
  return true;
}

LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  if (!workgroupSize.empty()) {
    return op->emitOpError(
        "expected workgroup size to be empty for CPU pipelines");
  }

  // Verify that the translation info is using the right pipeline.
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (translationInfo.getDispatchLoweringPassPipeline() != pipeline) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << pipelineName;
  }

  // Verify that the workload per workgroup is not set.
  // TODO(ravishankarm): Remove workload_per_wg eventually.
  SmallVector<int64_t> workloadPerWorkgroup =
      translationInfo.getWorkloadPerWorkgroupVals();
  if (!workloadPerWorkgroup.empty()) {
    return op->emitOpError(
               "workload_per_wg expected to be empty since its internal "
               "compiler implementation detail")
           << kNumMaxParallelDims;
  }

  if (loweringConfig.getTileSizes().size() !=
      static_cast<unsigned>(StrategyTilingLevel::NumStrategyTileLevels)) {
    return op->emitOpError("expected three tiling sizes for ")
           << pipelineName << ", got " << loweringConfig.getTileSizes().size();
  }

  IREE::Flow::PartitionableLoopsInterface interfaceOp =
      dyn_cast_or_null<IREE::Flow::PartitionableLoopsInterface>(op);
  if (interfaceOp) {
    SmallVector<int64_t> firstLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(StrategyTilingLevel::WorkGroupTiles));
    // This is needed to fuse and distribute all ops together.
    if (firstLevelTileSizes.size() != interfaceOp.getNumLoops()) {
      return op->emitOpError(
          "mismatch between number of loops and first level of tiling");
    }

    llvm::SmallDenseSet<unsigned> pLoopsSet;
    for (auto iteratorType : llvm::enumerate(interfaceOp.getIteratorTypes())) {
      if (iteratorType.value() == getParallelIteratorTypeName()) {
        pLoopsSet.insert(iteratorType.index());
      }
    }

    SmallVector<int64_t> secondLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(StrategyTilingLevel::ParallelTiles));
    for (auto en : llvm::enumerate(secondLevelTileSizes)) {
      if (en.value() != 0 && !pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only parallel dims to be set in the "
                   "second tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }

    SmallVector<int64_t> thirdLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(StrategyTilingLevel::ReductionTiles));
    for (auto en : llvm::enumerate(thirdLevelTileSizes)) {
      if (en.value() != 0 && pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only reduction dims to be set in the third "
                   "tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }
  }

  // Verify interchange
  if (!loweringConfig.getTileInterchange().empty()) {
    for (auto level : llvm::seq<unsigned>(
             0, static_cast<unsigned>(
                    loweringConfig.getTileInterchange().size()))) {
      auto tileSizes = loweringConfig.getTileSizeVals(level);
      auto interchange = loweringConfig.getTileInterchangeVals(level);
      if (!isValidInterchange(interchange, tileSizes.size())) {
        return op->emitOpError("expected [0, ")
               << tileSizes.size()
               << ") to be set exactly once in interchange #" << level;
      }
    }
  }

  // Verify that native vector size is empty.
  SmallVector<int64_t> nativeVectorSize =
      loweringConfig.getNativeVectorSizeVals();
  if (!nativeVectorSize.empty()) {
    return op->emitOpError("native_vector_size must be empty");
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addSingleTilingExpertPassPipeline(OpPassManager &passManager) {
  // Do first level of tiling and distribution.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());
  passManager.addPass(createCanonicalizerPass());
  // Add the sandbox single tiling expert to tile and vectorize.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.vectorize = true;
    options.tilingLevel = static_cast<int64_t>(TilingLevel::L1Tiles);
    passManager.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
  }

  // TODO(ravishankarm): This is commented cause this is WIP, to be enabled
  // soon.
  // auto callbacks =
  //     std::make_unique<linalg::comprehensive_bufferize::AllocationCallbacks>(
  //         cpuComprehensiveBufferizeAllocationFn,
  //         cpuComprehensiveBufferizeDeallocationFn,
  //         cpuComprehensiveBufferizeCopyFn);
  // addIREEComprehensiveBufferizePasses(passManager, std::move(callbacks));
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addCPUBufferOpsTileAndVectorizePipeline(OpPassManager &passManager) {
  // Do first level of tiling and distribution.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // Run IREE specific passes before vector lowering expert.
  passManager.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addDoubleTilingExpertPassPipeline(OpPassManager &passManager,
                                       bool lowerToAVX2) {
  passManager.addPass(createVerifyLinalgTransformLegalityPass());

  // Do first level of tiling and distribution.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());

  // Run LinalgFusePass firstly in case that we have fill + matmul + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is matmul + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    passManager.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to tile and vectorize.
  {
    // The options are derived from sandbox codegen driver. hoistPadding options
    // does not work in IREE cases. It's fine to not have it, since it's already
    // generating the IR as same as sandbox.
    LinalgSingleTilingExpertPassOptions options;
    options.vectorize = true;
    options.vectorizePadding = true;
    // TODO(#8228): Enable the padding once we know how to deal with fusion. For
    // now, we don't enable padding because alloca ops will be created in
    // bufferization for some cases.
    // options.pad = true;
    // options.packPaddings = {1, 1, 0};
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ReductionTiles);
    passManager.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
  }

  addCPUIREEComprehensiveBufferizePasses(passManager);

  // Run IREE specific passes before vector lowering expert.
  passManager.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.lowerVectorTransposeToAVX2 = lowerToAVX2;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createVerifyLinalgTransformLegalityPass());

  // Do first level of tiling and distribution.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());

  // Run LinalgFusePass firstly in case that we have fill + conv + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is conv + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ParallelTiles);
    passManager.addNestedPass<func::FuncOp>(createLinalgFusePass(options));
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to tile and vectorize.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.decomposeToLowerDimOp = true;
    options.vectorize = true;
    options.vectorizePadding = true;
    options.tilingLevel =
        static_cast<int64_t>(StrategyTilingLevel::ReductionTiles);
    passManager.addNestedPass<func::FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
  }

  addCPUIREEComprehensiveBufferizePasses(passManager);

  // Run IREE specific passes before vector lowering expert.
  passManager.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<func::FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "shuffle";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addTileFuseAndVectorizePassPipeline(OpPassManager &passManager,
                                         bool lowerToVectors) {
  // Do first level of tile and distribute to workgroups.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());
  passManager.addPass(createCanonicalizerPass());

  // Tile and vectorize linalg ops on tensors.
  passManager.addNestedPass<func::FuncOp>(
      createLLVMCPUTileFuseAndVectorizePass(lowerToVectors));
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  addCPUIREEComprehensiveBufferizePasses(passManager);
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  passManager.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  passManager.addNestedPass<func::FuncOp>(createOptimizeVectorTransferPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager) {
  // Do first level of tile and distribute to workgroups.
  passManager.addNestedPass<func::FuncOp>(createInsertDistributionInfoPass());
  passManager.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  passManager.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // TODO(#9004): Use upstream bufferization once the bufferization of LinalgExt
  // ops are implemented.
  // passManager.addNestedPass<func::FuncOp>(
  // createConvertToDestinationPassingStylePass());
  // passManager.addPass(createCanonicalizerPass());
  // addCPUIREEComprehensiveBufferizePasses(passManager);
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
}

void addLinalgTransformInterpPasses(OpPassManager &passManager) {
  // Give control to the linalg_transform dialect.
  passManager.addPass(createLinalgTransformInterpreterPass());

  // Dropping the schedule is only needed if we want to embed the transform in
  // the module: we should drop the schedule once applied.
  // This pass does nothing in the case where we apply a separate policy
  // through a file.
  passManager.addPass(createDropSchedulePass());
}

static void addLowerToLLVMPasses(OpPassManager &passManager) {
  // LinalgExt -> SCF
  passManager.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  passManager.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  passManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());

  // SCF -> STD
  passManager.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());

  if (clCheckIRBeforeLLVMConversion) {
    passManager.addPass(createLLVMCPUCheckIRBeforeLLVMConversionPass());
  }
  // Handled tensor-type constants.
  passManager.addPass(arith::createConstantBufferizePass());
  passManager.addPass(createFoldTensorExtractOpPass());

  // math dialect elementry functions -> polynomial form.
  passManager.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  passManager.addNestedPass<func::FuncOp>(
      arith::createArithmeticExpandOpsPass());
  passManager.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  passManager.addPass(createConvertToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  passManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager) {
  passManager.nest<ModuleOp>().nest<func::FuncOp>().addPass(
      createTypePropagationPass());
  passManager.nest<ModuleOp>().addPass(createBufferizeCopyOnlyDispatchesPass());
  passManager.addPass(createLLVMCPULowerExecutableTargetPass());
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addLowerToLLVMPasses(nestedModulePM);
}

}  // namespace iree_compiler
}  // namespace mlir
