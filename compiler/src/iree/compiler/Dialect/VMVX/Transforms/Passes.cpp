// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::VMVX {

// ---------------------------------------------------------------------------
// Variant configuration
// ---------------------------------------------------------------------------

static void
buildVMVXConfigurationPassPipelineImpl(OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    // ---------------------------------------------------------------------------
    // Tensor-level optimization, kernel dispatch and lower to buffers.
    // ---------------------------------------------------------------------------
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  FunctionLikeNest(modulePassManager)
      .addPass([&]() { return createCPUMaterializeEncodingPass(); })
      // TODO: Remove the following pass the plumb support for
      // #hal.descriptor_type memory space through the stack.
      .addPass(createEraseHALDescriptorTypeFromMemRefPass);
  modulePassManager.addPass(createVMVXSelectLoweringStrategyPass());
}

void buildVMVXConfigurationPassPipeline(OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildVMVXConfigurationPassPipelineImpl(modulePassManager);
}

// ---------------------------------------------------------------------------
// Variant Translation
// ---------------------------------------------------------------------------

static void
buildVectorVMVXTransformPassPipeline(OpPassManager &variantPassManager) {

  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  // ---------------------------------------------------------------------------
  // Tensor-level optimization, kernel dispatch and lower to buffers.
  // ---------------------------------------------------------------------------
  {
    FunctionLikeNest(modulePassManager)
        .addPass(createVMVXLowerExecutableTargetPass);
  }
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  // ---------------------------------------------------------------------------
  // Linalg -> Vectors
  // ---------------------------------------------------------------------------

  FunctionLikeNest(modulePassManager)
      .addPass(createCanonicalizerPass)

      // Linalg -> SCF.
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
      .addPass(createMemrefCopyToLinalgPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass([]() { return createConvertVectorToSCFPass(); })
      .addPass(createCanonicalizerPass)
      .addPass(memref::createExpandOpsPass);

  // Handle tensor-type constants.
  addConstantBufferizePasses(modulePassManager);
  FunctionLikeNest(modulePassManager)
      .addPass(createFoldTensorExtractOpPass)

      // Resolve get_buffer_descriptor ops. All structural buffer manipulations
      // must conclude before this point.
      .addPass(createIREEExpandStridedMetadataPass)
      .addPass(createResolveBufferDescriptorsPass)
      .addPass(createCleanupBufferAllocViewPass)

      // Flatten and cleanup memrefs.
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  modulePassManager.addPass(createFlattenMemRefSubspanPass());
  modulePassManager.addPass(memref::createNormalizeMemRefsPass());

  FunctionLikeNest(modulePassManager)
      .addPass(affine::createAffineScalarReplacementPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);
}

static void buildLoopOptimizationVMVXTransformPassPipeline(
    FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createLowerAffinePass)
      .addPass(createForOpCanonicalizationPass)
      .addPass(createLoopInvariantCodeMotionPass);
}

void buildVMVXTransformPassPipeline(OpPassManager &variantPassManager) {
  // ---------------------------------------------------------------------------
  // Linalg -> Scalars/Vectors
  // ---------------------------------------------------------------------------

  buildVectorVMVXTransformPassPipeline(variantPassManager);

  // ---------------------------------------------------------------------------
  // Standard/Vector/HAL/etc -> VMVX conversion
  // ---------------------------------------------------------------------------

  OpPassManager &modulePassManager = variantPassManager.nest<mlir::ModuleOp>();
  modulePassManager.addPass(createMaterializeConstantsPass());
  modulePassManager.addPass(createConversionPass());

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createCanonicalizerPass).addPass(createCSEPass);

  // ---------------------------------------------------------------------------
  // Cleanup and canonicalization
  // ---------------------------------------------------------------------------

  buildLoopOptimizationVMVXTransformPassPipeline(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass).addPass(createCSEPass);
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
      [](OpPassManager &modulePassManager) {
        buildVMVXConfigurationPassPipeline(modulePassManager);
      });

  static PassPipelineRegistration<> transformPassPipeline(
      "iree-vmvx-transformation-pipeline",
      "Runs the full IREE VMVX dialect transformation pipeline",
      [](OpPassManager &variantPassManager) {
        buildVMVXTransformPassPipeline(variantPassManager);
      });
}

} // namespace mlir::iree_compiler::IREE::VMVX
