// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static Value cpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

LogicalResult verifyTensorToVectorsPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  if (!workgroupSize.empty()) {
    return op->emitOpError(
        "expected workgroup size to be empty for CPU pipelines");
  }

  // Verify that the translation info is using the right pipeline.
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::CPUTensorToVectors;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (translationInfo.getDispatchLoweringPassPipeline() != pipeline) {
    return op->emitOpError("expected pipeline in translation.info to be ")
           << pipelineName;
  }

  // Verify that the workload per workgroup is set and is non-zero.
  SmallVector<int64_t> workloadPerWorkgroup =
      translationInfo.getWorkloadPerWorkgroupVals();
  if (workloadPerWorkgroup.size() > kNumMaxParallelDims) {
    return op->emitOpError("workload_per_wg size should be less than ")
           << kNumMaxParallelDims;
  }
  if (isa<linalg::LinalgOp, IREE::LinalgExt::TiledOpInterface>(op)) {
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    if (workloadPerWorkgroup.size() != partitionedLoops.size()) {
      return op->emitOpError("expected ")
             << partitionedLoops.size()
             << " entries for workload_per_wg, but got "
             << workloadPerWorkgroup.size();
    }
  }
  if (llvm::any_of(workloadPerWorkgroup,
                   [](int64_t val) { return val == 0; })) {
    return op->emitOpError("invalid to use 0 in workload_per_wg");
  }

  if (loweringConfig.getTileSizes().size() != 3) {
    return op->emitOpError("expected three levels of tile sizes for ")
           << pipelineName << ", got " << loweringConfig.getTileSizes().size();
  }
  SmallVector<int64_t> firstLevelTileSizes = loweringConfig.getTileSizeVals(0);
  if (!firstLevelTileSizes.empty()) {
    // Verify that if the first-level tile sizes are set, they are the same as
    // workload_per_wg for the partitioned loops.
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    size_t minElements =
        (partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1);
    if (firstLevelTileSizes.size() < minElements) {
      return op->emitOpError("expected at least ")
             << minElements
             << " size for first level tiling to get the distribution fully "
                "specified.";
    }
    llvm::SmallDenseSet<unsigned> partitionedLoopsSet;
    partitionedLoopsSet.insert(partitionedLoops.begin(),
                               partitionedLoops.end());
    SmallVector<int64_t> partitionedTileSizes;
    for (auto tileSize : llvm::enumerate(firstLevelTileSizes)) {
      if (!partitionedLoopsSet.count(tileSize.index())) {
        continue;
      }
      partitionedTileSizes.push_back(tileSize.value());
    }
    for (auto val : llvm::enumerate(llvm::reverse(workloadPerWorkgroup))) {
      if (val.value() != partitionedTileSizes[val.index()]) {
        return op->emitOpError("mismatch in distributed tile size value ")
               << partitionedTileSizes[val.index()] << " at position "
               << val.index() << " and workload_per_wg value " << val.value();
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

void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors) {
  passManager.addPass(createCanonicalizerPass());

  // Tile and vectorize linalg ops on tensors.
  passManager.addNestedPass<FuncOp>(
      createLLVMCPUTileAndVectorizePass(lowerToVectors));
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  passManager.addNestedPass<FuncOp>(createForOpCanonicalizationPass());

  passManager.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addTileFuseAndVectorizePassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());

  // Tile and vectorize linalg ops on tensors.
  passManager.addNestedPass<FuncOp>(createLLVMCPUTileFuseAndVectorizePass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  passManager.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  passManager.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());
  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
}

static void addLowerToLLVMPasses(OpPassManager &passManager) {
  // LinalgExt -> SCF
  passManager.addNestedPass<FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addNestedPass<FuncOp>(
      Shape::createFoldDimOverShapeCarryingOpPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  passManager.addNestedPass<FuncOp>(createLowerToCFGPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  passManager.addPass(createTensorConstantBufferizePass());
  passManager.addPass(createFoldTensorExtractOpPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  passManager.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  passManager.addNestedPass<FuncOp>(createStdExpandOpsPass());
  passManager.addPass(createConvertToLLVMPass());

  // We rely on MLIR symbol visibility being correct after this point and need
  // to mirror the LLVM linkage that was assigned during conversion.
  passManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createLLVMCPULowerExecutableTargetPass());
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addLowerToLLVMPasses(nestedModulePM);
}

}  // namespace iree_compiler
}  // namespace mlir
