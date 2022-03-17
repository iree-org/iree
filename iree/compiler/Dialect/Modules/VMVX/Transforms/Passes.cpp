// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"

#include <memory>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

static void buildVectorVMVXTransformPassPipeline(OpPassManager &passManager) {
  passManager.nest<ModuleOp>().nest<FuncOp>().addPass(
      createTypePropagationPass());
  passManager.addPass(createLLVMCPULowerExecutableTargetPass());

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  // ---------------------------------------------------------------------------
  // Linalg -> Vectors
  // ---------------------------------------------------------------------------

  // Tiling and distribution.
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  // TODO(#5925): This can also be modified to just use the dynamic pass
  // pipeline like the CPU side.
  // nestedModulePM.addNestedPass<FuncOp>(
  //     createLinalgTileAndVectorizeWorkgroupsPass());

  // Linalg -> SCF.
  nestedModulePM.addNestedPass<FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());
  nestedModulePM.addNestedPass<FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<FuncOp>(createConvertVectorToSCFPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  nestedModulePM.addNestedPass<FuncOp>(memref::createExpandOpsPass());

  // Handle tensor-type constants.
  nestedModulePM.addPass(arith::createConstantBufferizePass());
  nestedModulePM.addPass(createFoldTensorExtractOpPass());

  // Flatten and cleanup memrefs.
  nestedModulePM.addNestedPass<FuncOp>(memref::createFoldSubViewOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addPass(createFlattenMemRefSubspanPass());
  nestedModulePM.addPass(memref::createNormalizeMemRefsPass());
  nestedModulePM.addNestedPass<FuncOp>(createAffineScalarReplacementPass());
  nestedModulePM.addPass(createCanonicalizerPass());
}

static void buildLoopOptimizationVMVXTransformPassPipeline(
    OpPassManager &passManager) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  nestedModulePM.addNestedPass<FuncOp>(createLowerAffinePass());
  nestedModulePM.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
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

  passManager.addNestedPass<mlir::ModuleOp>(createConversionPass());
  passManager.nest<mlir::ModuleOp>().addNestedPass<FuncOp>(
      memref::createFoldSubViewOpsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // ---------------------------------------------------------------------------
  // Cleanup and canonicalization
  // ---------------------------------------------------------------------------

  buildLoopOptimizationVMVXTransformPassPipeline(passManager);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void createVMVXTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vmvx-transformation-pipeline",
      "Runs the full IREE VMVX dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMVXTransformPassPipeline(passManager);
      });
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
