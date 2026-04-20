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
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::VMVX {

// ---------------------------------------------------------------------------
// Module-scope translation
// ---------------------------------------------------------------------------

static void
buildVectorVMVXTransformPassPipeline(OpPassManager &modulePassManager) {
  // ---------------------------------------------------------------------------
  // Tensor-level optimization, kernel dispatch and lower to buffers.
  // ---------------------------------------------------------------------------
  buildVMVXLoweringPassPipeline(modulePassManager);

  // ---------------------------------------------------------------------------
  // Linalg -> Vectors
  // ---------------------------------------------------------------------------
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

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
      .addPass([&]() {
        arith::ArithExpandOpsPassOptions options;
        options.includeBf16 = true;
        options.includeF8E8M0 = true;
        return arith::createArithExpandOpsPass(options);
      })
      .addPass(createConvertUnsupportedFloatArithPass)
      .addPass([]() {
        return createEmulateNarrowTypePass(
            EmulateNarrowTypePassOptions{/*disableAtomicRMW=*/true});
      });

  // Handle tensor-type constants.
  modulePassManager.addPass(createIREEBufferizeConstantsPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createFoldTensorExtractOpPass)

      // Resolve get_buffer_descriptor ops. All structural buffer manipulations
      // must conclude before this point.
      .addPass(memref::createFoldMemRefAliasOpsPass)
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
  funcPassManager.addPass(createIREELowerAffinePass)
      .addPass(createForOpCanonicalizationPass)
      .addPass(createIREELoopInvariantCodeMotionPass);
}

void buildVMVXTransformPassPipeline(OpPassManager &modulePassManager) {
  // ---------------------------------------------------------------------------
  // Linalg -> Scalars/Vectors
  // ---------------------------------------------------------------------------

  buildVectorVMVXTransformPassPipeline(modulePassManager);

  // ---------------------------------------------------------------------------
  // Standard/Vector/HAL/etc -> VMVX conversion
  // ---------------------------------------------------------------------------

  modulePassManager.addPass(createMaterializeConstantsPass());
  modulePassManager.addPass(createConversionPass());

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createCanonicalizerPass).addPass(createCSEPass);

  // ---------------------------------------------------------------------------
  // Cleanup and canonicalization
  // ---------------------------------------------------------------------------

  buildLoopOptimizationVMVXTransformPassPipeline(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(IREE::Util::createDropCompilerHintsPass);
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
      [](OpPassManager &modulePassManager) {
        buildVMVXTransformPassPipeline(modulePassManager);
      });
}

} // namespace mlir::iree_compiler::IREE::VMVX
