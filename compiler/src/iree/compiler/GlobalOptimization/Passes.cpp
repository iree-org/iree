// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

static llvm::cl::opt<bool> clDemoteI64ToI32(
    "iree-global-opt-demote-i64-to-i32",
    llvm::cl::desc("Converts all i64 ops and values into i32 counterparts "
                   "unconditionally before global optimization."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clDemoteF32ToF16(
    "iree-global-opt-demote-f32-to-f16",
    llvm::cl::desc("Converts all f32 ops and values into f16 counterparts "
                   "unconditionally before global optimization."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clPromoteBF16ToF32(
    "iree-global-opt-promote-bf16-to-f32",
    llvm::cl::desc("Converts all bf16 ops and values into f32 counterparts "
                   "unconditionally before global optimization."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clPromoteF16ToF32(
    "iree-global-opt-promote-f16-to-f32",
    llvm::cl::desc("Converts all f16 ops and values into f32 counterparts "
                   "unconditionally before global optimization."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clDemoteF64ToF32(
    "iree-global-opt-demote-f64-to-f32",
    llvm::cl::desc("Converts all f64 ops and values into f32 counterparts "
                   "unconditionally before global optimization."),
    llvm::cl::init(true));

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {


using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions) {
  // ML frontends have very uneven support for user-controlled types _and_ users
  // tend to use types not well suited for the work they are doing. These
  // demotions/promotions allow users to change the types after lowering out of
  // the frontends. It'll always be better to do this higher up in the stack
  // as these kind of blanket conversions have corner cases and potential
  // accuracy/precision losses beyond what the user may expect.
  if (clDemoteF64ToF32) {
    mainPassManager.addPass(IREE::Util::createDemoteF64ToF32Pass());
  }
  if (clDemoteF32ToF16) {
    mainPassManager.addPass(IREE::Util::createDemoteF32ToF16Pass());
  }
  if (clPromoteF16ToF32) {
    mainPassManager.addPass(IREE::Util::createPromoteF16ToF32Pass());
  }
  if (clDemoteI64ToI32) {
    mainPassManager.addPass(IREE::Util::createDemoteI64ToI32Pass());
  }
  if (clPromoteBF16ToF32) {
    mainPassManager.addPass(IREE::Util::createPromoteBF16ToF32Pass());
  }

  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Flow::createRemoveZeroExtentTensorsPass)
      .addPass(IREE::Flow::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(IREE::Flow::createConvert1X1FilterConv2DToMatmulPass);
  mainPassManager.addPass(IREE::Flow::createEraseUnusedLinalgOperands());

  // Expand tensor shapes into SSA values and optimize the whole program.
  // The more we are able to equate shape dimensions at this level the better
  // our fusions will be.
  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Flow::createTopLevelSCFToCFGPass);
  mainPassManager.addPass(IREE::Flow::createExpandTensorShapesPass());

  OpPassManager pipeline(ModuleOp::getOperationName());
  FunctionLikeNest(pipeline)
      // Simplify util.global accesses early on; this can help with dispatch
      // region formation as redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other util
  // ops).
  pipeline.addPass(IREE::Util::createApplyPatternsPass());
  pipeline.addPass(IREE::Util::createFoldGlobalsPass());
  pipeline.addPass(IREE::Util::createIPOPass());

  if (transformOptions.constExprHoisting) {
    pipeline.addPass(IREE::Util::createHoistIntoGlobalsPass());
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(pipeline);
  }

  if (transformOptions.numericPrecisionReduction) {
    pipeline.addPass(IREE::Flow::createInferNumericNarrowingPass());
    pipeline.addPass(IREE::Flow::createOptimizeNumericsPass());
    pipeline.addPass(IREE::Flow::createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(pipeline)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Add the whole fixed point iterator.
  mainPassManager.addPass(
      IREE::Util::createFixedPointIteratorPass(std::move(pipeline)));
}

void registerGlobalOptimizationPipeline() {
  PassPipelineRegistration<TransformOptions>
      globalOptimizationTransformPassPipeline(
          "iree-global-optimization-transformation-pipeline",
          "Runs the IREE global optimization transformation pipeline",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildGlobalOptimizationPassPipeline(passManager, transformOptions);
          });
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
