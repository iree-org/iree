// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"

#include <memory>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::VMVX {

// ---------------------------------------------------------------------------
// Variant configuration
// ---------------------------------------------------------------------------

void buildVMVXConfigurationPassPipeline(OpPassManager &passManager) {
  // ---------------------------------------------------------------------------
  // Tensor-level optimization, kernel dispatch and lower to buffers.
  // ---------------------------------------------------------------------------
  addCommonTargetExecutablePreprocessingPasses(passManager);
  passManager.nest<ModuleOp>().addNestedPass<func::FuncOp>(
      createCPUMaterializeEncodingPass());
  // TODO: Remove the following pass the plumb support for #hal.descriptor_type
  // memory space through the stack.
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(createVMVXSelectLoweringStrategyPass());
}

// ---------------------------------------------------------------------------
// Variant Translation
// ---------------------------------------------------------------------------

static void buildVectorVMVXTransformPassPipeline(OpPassManager &passManager) {
  // ---------------------------------------------------------------------------
  // Tensor-level optimization, kernel dispatch and lower to buffers.
  // ---------------------------------------------------------------------------
  passManager.addPass(createVMVXLowerExecutableTargetPass());

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  // ---------------------------------------------------------------------------
  // Linalg -> Vectors
  // ---------------------------------------------------------------------------

  // Tiling and distribution.
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // TODO(#5925): This can also be modified to just use the dynamic pass
  // pipeline like the CPU side.
  // nestedModulePM.addNestedPass<func::FuncOp>(
  //     createLinalgTileAndVectorizeWorkgroupsPass());

  // Linalg -> SCF.
  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  // Handle tensor-type constants.
  nestedModulePM.addPass(arith::createConstantBufferizePass());
  nestedModulePM.addPass(createFoldTensorExtractOpPass());

  // Resolve get_buffer_descriptor ops. All structural buffer manipulations
  // must conclude before this point.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createIREEExpandStridedMetadataPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createResolveBufferDescriptorsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createCleanupBufferAllocViewPass());

  // Flatten and cleanup memrefs.
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addPass(createFlattenMemRefSubspanPass());
  nestedModulePM.addPass(memref::createNormalizeMemRefsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      affine::createAffineScalarReplacementPass());
  nestedModulePM.addPass(createCanonicalizerPass());
}

static void
buildLoopOptimizationVMVXTransformPassPipeline(OpPassManager &passManager) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  nestedModulePM.addNestedPass<func::FuncOp>(createLowerAffinePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
}

void buildVMVXTransformPassPipeline(OpPassManager &passManager) {
  // ---------------------------------------------------------------------------
  // Linalg -> Scalars/Vectors
  // ---------------------------------------------------------------------------

  buildVectorVMVXTransformPassPipeline(passManager);

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // ---------------------------------------------------------------------------
  // Standard/Vector/HAL/etc -> VMVX conversion
  // ---------------------------------------------------------------------------

  passManager.addNestedPass<mlir::ModuleOp>(createMaterializeConstantsPass());
  passManager.addNestedPass<mlir::ModuleOp>(createConversionPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // ---------------------------------------------------------------------------
  // Cleanup and canonicalization
  // ---------------------------------------------------------------------------

  buildLoopOptimizationVMVXTransformPassPipeline(passManager);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h.inc"
} // namespace

void registerVMVXPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> configurationPassPipeline(
      "iree-vmvx-configuration-pipeline",
      "Runs the full IREE VMVX dialect configuration pipeline",
      [](OpPassManager &passManager) {
        buildVMVXConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> transformPassPipeline(
      "iree-vmvx-transformation-pipeline",
      "Runs the full IREE VMVX dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMVXTransformPassPipeline(passManager);
      });
}

} // namespace mlir::iree_compiler::IREE::VMVX
