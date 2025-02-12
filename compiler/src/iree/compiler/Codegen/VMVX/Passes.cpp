// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Transforms/Passes.h"

#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.

static llvm::cl::opt<bool> clSkipIntermediateRoundings(
    "iree-vmvx-skip-intermediate-roundings",
    llvm::cl::desc(
        "Allow skipping intermediate roundings. For example, in f16 matmul "
        "kernels on targets with only f32 arithmetic, we have to perform each "
        "multiply-accumulate in f32, and if this flag is false, then we have "
        "to round those f32 accumulators to the nearest f16 every time, which "
        "is slow."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableUKernelsDecomposeLinalgGeneric(
    "iree-vmvx-enable-ukernels-decompose-linalg-generic",
    llvm::cl::desc("Enables decomposition of linalg.generic ops when "
                   "ukernels are enabled (experimental)"),
    llvm::cl::init(true));

static void addTileAndDistributePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createConvertToDestinationPassingStylePass());
  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  funcPassManager.addPass(createConcretizePadResultShapePass());
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
}

void addVMVXDefaultPassPipeline(OpPassManager &funcPassManager,
                                bool enableUKernels) {
  addTileAndDistributePasses(funcPassManager);

  if (enableUKernels) {
    funcPassManager.addPass(createCPUPrepareUkernelsPass());
    funcPassManager.addPass(
        createCPULowerToUKernelsPass(clSkipIntermediateRoundings));
  }

  // Tensor-level micro-kernel optimizations.
  // Note that this must be done post-tiling because it changes the structure
  // of the dispatch region such that tiling is not always possible.
  if (enableUKernels && clEnableUKernelsDecomposeLinalgGeneric) {
    funcPassManager.addPass(createDecomposeLinalgGenericPass());
  }

  // Lower to buffers.
  addCPUBufferizePasses(funcPassManager);

  // Cleanup the IR that may now have unused loops.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  // Convert buffer-level microkernels.
  if (enableUKernels) {
    funcPassManager.addPass(createVMVXLowerLinalgMicrokernelsPass());
  }
}

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildVMVXLinkingPassPipeline(OpPassManager &modulePassManager) {
  // Link together executables. This may produce some IR duplication.
  modulePassManager.addPass(createVMVXLinkExecutablesPass());

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant ordinals.
  modulePassManager.nest<IREE::HAL::ExecutableOp>()
      .addNestedPass<IREE::HAL::ExecutableVariantOp>(
          createVMVXAssignConstantOrdinalsPass());
}

//===---------------------------------------------------------------------===//
// Register VMVX Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/VMVX/Passes.h.inc"
} // namespace

void registerCodegenVMVXPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> VMVXLinkingPipeline(
      "iree-codegen-vmvx-linking-pipeline",
      "Runs the VMVX HAL executable linking pipeline",
      [](OpPassManager &modulePassManager) {
        buildVMVXLinkingPassPipeline(modulePassManager);
      });
}

} // namespace mlir::iree_compiler
