// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
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
    llvm::cl::init(false));

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
//===---------------------------------------------------------------------===//

// Default allocation function to use with IREEs bufferization.
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

//===---------------------------------------------------------------------===//
// Codegen configuration verifications.
//===---------------------------------------------------------------------===//

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

  if (loweringConfig.getTileSizes().size() != 3) {
    return op->emitOpError("expected three tiling sizes for ")
           << pipelineName << ", got " << loweringConfig.getTileSizes().size();
  }

  IREE::Flow::PartitionableLoopsInterface interfaceOp =
      dyn_cast_or_null<IREE::Flow::PartitionableLoopsInterface>(op);
  if (interfaceOp) {
    llvm::SmallDenseSet<unsigned> pLoopsSet;
    for (auto iteratorType : llvm::enumerate(interfaceOp.getIteratorTypes())) {
      if (iteratorType.value() == getParallelIteratorTypeName()) {
        pLoopsSet.insert(iteratorType.index());
      }
    }

    SmallVector<int64_t> secondLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(TilingLevel::L1Tiles));
    for (auto en : llvm::enumerate(secondLevelTileSizes)) {
      if (en.value() != 0 && !pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only parallel dims to be set in the "
                   "second tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }

    SmallVector<int64_t> thirdLevelTileSizes = loweringConfig.getTileSizeVals(
        static_cast<unsigned>(TilingLevel::VectorTiles));
    for (auto en : llvm::enumerate(thirdLevelTileSizes)) {
      if (en.value() != 0 && pLoopsSet.contains(en.index())) {
        return op->emitOpError(
                   "expected only reduction dims to be set in the third "
                   "tiling sizes, got ")
               << en.index() << "-th tile size set";
      }
    }
  }

  // Verify that native vector size is either empty, or if set is same as the
  // last level of tiling
  SmallVector<int64_t> nativeVectorSize =
      loweringConfig.getNativeVectorSizeVals();
  if (!nativeVectorSize.empty()) {
    if (nativeVectorSize !=
        loweringConfig.getTileSizeVals(
            static_cast<unsigned>(TilingLevel::VectorTiles))) {
      return op->emitOpError(
          "native_vector_size must be same as the last level of tiling");
    }
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addSingleTilingExpertPassPipeline(OpPassManager &passManager) {
  // Do first level of tiling and distribution.
  passManager.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<FuncOp>(
      createConvertToDestinationPassingStylePass());
  passManager.addPass(createCanonicalizerPass());
  // Add the sandbox single tiling expert to tile and vectorize.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.vectorize = true;
    options.tilingLevel = static_cast<int64_t>(TilingLevel::L1Tiles);
    passManager.addNestedPass<FuncOp>(
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
    OpPassManager &nestedFuncPassManager = passManager.nest<FuncOp>();
    LinalgVectorLoweringPassOptions options;
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addDoubleTilingExpertPassPipeline(OpPassManager &passManager) {
  // Do first level of tiling and distribution.
  passManager.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<FuncOp>(
      createConvertToDestinationPassingStylePass());

  passManager.addPass(createCanonicalizerPass());

  // Run LinalgFusePass firstly in case that we have fill + matmul + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is matmul + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel = static_cast<int64_t>(TilingLevel::L1Tiles);
    passManager.addNestedPass<FuncOp>(createLinalgFusePass(options));
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<FuncOp>(createCSEPass());
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
    options.tilingLevel = static_cast<int64_t>(TilingLevel::VectorTiles);
    passManager.addNestedPass<FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<FuncOp>(createCSEPass());
  }

  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);

  // Run IREE specific passes before vector lowering expert.
  passManager.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "linalg-copy";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager) {
  // Do first level of tiling and distribution.
  passManager.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<FuncOp>(
      createConvertToDestinationPassingStylePass());

  passManager.addPass(createCanonicalizerPass());

  // Run LinalgFusePass firstly in case that we have fill + conv + generic
  // ops. At this stage, we do not apply vectorization. The reduction dim won't
  // get tiled if the case is conv + generic op. In this case, we have to tile
  // along reduction dim again, which needs them to be Linalg ops form.
  {
    LinalgFusePassOptions options;
    options.tilingLevel = static_cast<int64_t>(TilingLevel::L1Tiles);
    passManager.addNestedPass<FuncOp>(createLinalgFusePass(options));
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<FuncOp>(createCSEPass());
  }

  // Add the sandbox single tiling expert to tile and vectorize.
  {
    LinalgSingleTilingExpertPassOptions options;
    options.decomposeToLowerDimOp = true;
    options.vectorize = true;
    options.vectorizePadding = true;
    options.tilingLevel = static_cast<int64_t>(TilingLevel::VectorTiles);
    passManager.addNestedPass<FuncOp>(
        createLinalgSingleTilingExpertPass(options));
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<FuncOp>(createCSEPass());
  }

  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);

  // Run IREE specific passes before vector lowering expert.
  passManager.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());

  // Add the vector lowering expert.
  {
    OpPassManager &nestedFuncPassManager = passManager.nest<FuncOp>();
    LinalgVectorLoweringPassOptions options;
    options.splitVectorTransfersTo = "shuffle";
    addLowerToVectorTransforms(nestedFuncPassManager, options);
  }
}

void addTileFuseAndVectorizePassPipeline(OpPassManager &passManager,
                                         bool lowerToVectors) {
  // Do first level of tile and distribute to workgroups.
  passManager.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // Tile and vectorize linalg ops on tensors.
  passManager.addNestedPass<FuncOp>(
      createLLVMCPUTileFuseAndVectorizePass(lowerToVectors));
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Use stack allocation on CPU side.

  // TODO(ravishankarm): This is commented cause this is WIP, to be enabled
  // soon.
  //
  // auto callbacks =
  //    std::make_unique<linalg::comprehensive_bufferize::AllocationCallbacks>(
  //        cpuComprehensiveBufferizeAllocationFn,
  //        cpuComprehensiveBufferizeDeallocationFn,
  //        cpuComprehensiveBufferizeCopyFn);
  // addIREEComprehensiveBufferizePasses(passManager, std::move(callbacks));

  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  passManager.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  passManager.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager) {
  // Do first level of tile and distribute to workgroups.
  passManager.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
}

void addSandboxCodegenPasses(OpPassManager &passManager) {
  // Sets the number of workgroups to be {1, 1, 1} for now.
  passManager.addPass(createSetNumWorkgroupsPass());

  OpPassManager &modulePM = passManager.nest<ModuleOp>();
  // Bufferize the dispatch.
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn =
      cpuComprehensiveBufferizeDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = cpuComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(modulePM, allocationFn, deallocationFn,
                                      memcpyFn);
}

static void addLowerToLLVMPasses(OpPassManager &passManager) {
  // LinalgExt -> SCF
  passManager.addNestedPass<FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  passManager.addNestedPass<FuncOp>(createMemrefCopyToLinalgPass());
  passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  passManager.addNestedPass<FuncOp>(createConvertSCFToCFPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  if (clCheckIRBeforeLLVMConversion) {
    passManager.addPass(createLLVMCPUCheckIRBeforeLLVMConversionPass());
  }
  // Handled tensor-type constants.
  passManager.addPass(arith::createConstantBufferizePass());
  passManager.addPass(createFoldTensorExtractOpPass());

  // math dialect elementry functions -> polynomial form.
  passManager.addNestedPass<FuncOp>(createPolynomialApproximationPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  passManager.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  passManager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  passManager.addPass(createConvertToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  passManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager) {
  passManager.nest<ModuleOp>().nest<FuncOp>().addPass(
      createTypePropagationPass());
  passManager.addPass(createLLVMCPULowerExecutableTargetPass());
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addLowerToLLVMPasses(nestedModulePM);
}

}  // namespace iree_compiler
}  // namespace mlir
