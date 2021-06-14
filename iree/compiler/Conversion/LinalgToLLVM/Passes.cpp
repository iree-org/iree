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

static llvm::cl::opt<bool> clUseTensorPadTileAndVectorize(
    "iree-codegen-linalg-to-llvm-use-tensor-to-vectors",
    llvm::cl::desc("If enabled will use tensor -> vector transformation pass"),
    llvm::cl::init(false));

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

  if (clUseTensorPadTileAndVectorize) {
    // Tile and vectorize linalg ops on tensors.
    passManager.addNestedPass<FuncOp>(
        createTilePadAndVectorizeWorkgroupsPass());
    passManager.addNestedPass<FuncOp>(createCSEPass());
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  }

  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  if (!clUseTensorPadTileAndVectorize) {
    // Tile and vectorize linalg ops on buffers.
    passManager.addNestedPass<FuncOp>(
        createLinalgTileAndVectorizeWorkgroupsPass(lowerToVectors));
    passManager.addNestedPass<FuncOp>(createCSEPass());
    passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  }

  passManager.addNestedPass<FuncOp>(createForOpCanonicalizationPass());

  passManager.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());
}

void addCPUDefaultPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());
  // Use stack allocation on CPU side.
  addLinalgBufferizePasses(passManager, cpuAllocationFunction);
  passManager.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());
}

static void addLowerToLLVMPasses(
    OpPassManager &passManager,
    const LLVMTransformPassPipelineOptions &options) {
  // Linalg -> SCF
  passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
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
  passManager.addPass(createConvertToLLVMPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void buildLLVMTransformPassPipeline(
    OpPassManager &passManager,
    const LLVMTransformPassPipelineOptions &options) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  addLowerToLLVMPasses(nestedModulePM, options);
}

static PassPipelineRegistration<LLVMTransformPassPipelineOptions>
    linalgLLVMVPipeline(
        "iree-codegen-linalg-to-llvm-pipeline",
        "Runs the progressive lowering pipeline from Linalg to LLVM",
        [](OpPassManager &passManager,
           const LLVMTransformPassPipelineOptions &options) {
          buildLLVMTransformPassPipeline(passManager, options);
        });

}  // namespace iree_compiler
}  // namespace mlir
