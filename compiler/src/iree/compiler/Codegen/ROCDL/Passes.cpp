// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/ROCDL/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-rocdl-lowering-pass-pipeline"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

// Add passes to make the address computation more explicit and optimize them.
//
// The idea here is to be less dependent on what the LLVM backend is able to do,
// by heavy lifting most of the work while we still have the information about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void addLowerAndOptimizeAddressComputationPasses(OpPassManager &pm) {
  pm.addPass(createExtractAddressComputationGPUPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  // Hoist loop invariant variables to give affine decomposition pass the right
  // loop dependencies.
  pm.addPass(createLoopInvariantCodeMotionPass());
  // Decompose affine ops.
  pm.addPass(createDecomposeAffineOpsPass());
  // Get rid of the redundant computations.
  pm.addPass(createCSEPass());
  // Hoist the resulting decompositions.
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createLowerAffinePass());
}

static void addLowerToROCDLPasses(OpPassManager &pm, bool useROCM) {
  pm.addPass(createConvertHALDescriptorTypeToGPUAddressSpacePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerUKernelOpsToCallsPass());

  // LinalgExt -> SCF
  pm.addNestedPass<func::FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  pm.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Pad allocations with dynamic dimension after linalg lowering but before
  // lowering SCF and affine ops.
  pm.addNestedPass<func::FuncOp>(createPadDynamicAlloc());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Handled tensor constants.
  pm.addPass(arith::createConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  pm.addNestedPass<func::FuncOp>(createLLVMGPUVectorLoweringPass());

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(pm);

  // Run checks on shared memory usage.
  // TODO: query this from the target.
  auto getSharedMemoryLimit = [](mlir::FunctionOpInterface) {
    return 163 * 1024;
  };
  auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
  pm.addPass(
      createGPUCheckResourceUsagePass(getSharedMemoryLimit, getIndexBitwidth));

  // SCF -> CF
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Handle complex operation conversion.
  pm.addPass(createConvertComplexToStandardPass());

  // Convert BF16 operations to occur as F32.
  pm.addPass(createConvertBf16ArithToF32Pass());
  pm.addPass(createConvertBf16ToUInt16BuffersPass());

  // Convert math dialect elementry functions to polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createEmulateNarrowTypePass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Strip out the debug info for the kernel.
  pm.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  pm.addPass(createLLVMGPUCastAddressSpaceFunction());

  pm.addPass(createConvertToROCDLPass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void buildROCDLCodegenConfigurationPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUGeneralizeNamedOpsPass());
  pm.addPass(createROCDLSelectLoweringStrategyPass());
}

void buildROCDLCodegenPassPipeline(OpPassManager &pm, bool useROCM) {
  pm.addPass(createROCDLLowerExecutableTargetPass());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  addLowerToROCDLPasses(nestedModulePM, useROCM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using ROCDL pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===---------------------------------------------------------------------===//
// Pass Registration
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/ROCDL/Passes.h.inc"
} // namespace

void registerCodegenROCDLPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> ROCDLConfigPipeline(
      "iree-codegen-rocdl-configuration-pipeline",
      "Runs pass pipeline to select a suitable lowering strategy for ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline2",
      "Runs pass pipeline to progressively lower Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenPassPipeline(passManager, true);
      });
}

} // namespace mlir::iree_compiler
