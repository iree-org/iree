// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
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

void addCPUVectorizationPassPipeline(OpPassManager &passManager,
                                     bool lowerToVectors) {
  passManager.addPass(createCanonicalizerPass());

  // TODO(ataei): This causes segmentation fault on Android. Fix it and
  // re-enable.
  // passManager.addNestedPass<FuncOp>(createPadLinalgWorkgroupTilesPass());

  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Tile and vectorize linalg ops on buffers.
  passManager.addNestedPass<FuncOp>(
      createLLVMCPUVectorizationPass(lowerToVectors));
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  passManager.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
}

void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors,
                                    bool useTileAndVectorizeV2) {
  passManager.addPass(createCanonicalizerPass());

  // Tile and vectorize linalg ops on tensors.
  if (useTileAndVectorizeV2) {
    passManager.addNestedPass<FuncOp>(
        createLLVMCPUTileFuseAndVectorizePass(lowerToVectors));
  } else {
    passManager.addNestedPass<FuncOp>(
        createLLVMCPUTileAndVectorizePass(lowerToVectors));
  }
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
