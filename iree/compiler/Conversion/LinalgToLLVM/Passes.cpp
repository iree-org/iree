// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Common/Passes.h"

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
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
                                     LLVMCodegenOptions options) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addPass(createCanonicalizerPass());

  // TODO(ataei): This causes segmentation fault on Android. Fix it and
  // re-enable.
  // nestedModulePM.addNestedPass<FuncOp>(createPadLinalgWorkgroupTilesPass());

  // TODO(ataei): We want to enable when tensor -> vector pass is fully
  // supported which requires first moving vector-tiling before this step.
  if (options.useLinalgOnTensorsToVectors) {
    nestedModulePM.addNestedPass<FuncOp>(createLinalgVectorizePass());
  }
  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(nestedModulePM, cpuAllocationFunction);

  // Tile and vectorize linalg ops.
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(
      createLinalgTileAndVectorizeWorkgroupsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createForOpCanonicalizationPass());

  nestedModulePM.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager,
                               LLVMCodegenOptions options) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addPass(createCanonicalizerPass());
  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(nestedModulePM, cpuAllocationFunction);
  nestedModulePM.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());
}

void addLowerToLLVMPasses(OpPassManager &passManager,
                          LLVMCodegenOptions options) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  // Linalg -> SCF
  nestedModulePM.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  nestedModulePM.addNestedPass<FuncOp>(createLowerToCFGPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  nestedModulePM.addPass(createTensorConstantBufferizePass());
  nestedModulePM.addPass(createFoldTensorExtractOpPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  nestedModulePM.addPass(createConvertToLLVMPass(options));

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void buildLLVMTransformPassPipeline(OpPassManager &passManager,
                                    LLVMCodegenOptions options) {
  passManager.addPass(createLowerExecutableTargetPass(options));
}

static PassPipelineRegistration<> linalgLLVMVPipeline(
    "iree-codegen-linalg-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager,
                                     getLLVMCodegenOptionsFromClOptions());
    });

}  // namespace iree_compiler
}  // namespace mlir
