// Copyright 2023 The IREE Authors //
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Modules/IO/Parameters/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

static llvm::cl::opt<bool> clEnableQuantizedMatmulReassociation(
    "iree-global-opt-enable-quantized-matmul-reassociation",
    llvm::cl::desc(
        "Enables reassociation of quantized matmul ops (experimental)."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clEnableFuseSiluHorizontalMatmul(
    "iree-global-opt-enable-fuse-silu-horizontal-matmul",
    llvm::cl::desc(
        "Enables fusing specifically structured matmuls (experimental)."),
    llvm::cl::init(false));
// TODO(#15973): Make default to true after fixing the CPU DT regression.
static llvm::cl::opt<bool> clEnableTransposePropagation(
    "iree-global-opt-propagate-transposes",
    llvm::cl::desc(
        "Enables propagation of transpose ops to improve fusion chances."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableDemoteContractionInputsToBF16(
    "iree-global-opt-enable-demote-contraction-inputs-to-bf16",
    llvm::cl::desc(
        "Demote inputs (LHS, RHS) of linalg matmul-like ops from f32 to bf16."),
    llvm::cl::init(false));

void buildGlobalOptExprHoistingPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  IREE::Util::ExprHoistingOptions options;
  options.maxSizeIncreaseThreshold =
      transformOptions.options.constExprMaxSizeIncreaseThreshold;
  options.registerDependentDialectsFn = [](DialectRegistry &registry) {
    registry.insert<IREE::Flow::FlowDialect>();
  };
  passManager.addPass(IREE::Util::createHoistIntoGlobalsPass(options));
}

void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions) {
  // ML frontends have very uneven support for user-controlled types _and_ users
  // tend to use types not well suited for the work they are doing. These
  // demotions/promotions allow users to change the types after lowering out of
  // the frontends. It'll always be better to do this higher up in the stack
  // as these kind of blanket conversions have corner cases and potential
  // accuracy/precision losses beyond what the user may expect.
  if (transformOptions.options.demoteF64ToF32) {
    mainPassManager.addPass(IREE::Util::createDemoteF64ToF32Pass());
  }
  if (transformOptions.options.demoteF32ToF16) {
    mainPassManager.addPass(IREE::Util::createDemoteF32ToF16Pass());
  }
  if (transformOptions.options.promoteF16ToF32) {
    mainPassManager.addPass(IREE::Util::createPromoteF16ToF32Pass());
  }
  if (transformOptions.options.promoteBF16ToF32) {
    mainPassManager.addPass(IREE::Util::createPromoteBF16ToF32Pass());
  }
  if (transformOptions.options.demoteI64ToI32) {
    mainPassManager.addPass(IREE::Util::createDemoteI64ToI32Pass());
  }

  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(mainPassManager)
      .addPass(createRemoveZeroExtentTensorsPass)
      .addPass(createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(createConvert1X1FilterConv2DToMatmulPass);
  mainPassManager.addPass(createEraseUnusedLinalgOperands());

  // Expand tensor shapes into SSA values and optimize the whole program.
  // The more we are able to equate shape dimensions at this level the
  // better our fusions will be.
  mainPassManager.addPass(createExpandTensorShapesPass());

  FunctionLikeNest(mainPassManager)
      // Preprocess the input to a form more amenable for fusion
      // - Convert all elementwise ops to Linalg
      // - Remove unit-extent dimensions.
      .addPass(mlir::createConvertElementwiseToLinalgPass)
      // RaiseSpecialOps, by virtue of implementing various peephole
      // optimizations, is sensitive to surrounding IR structure. Thus we run
      // this pass both before unit dim folding + consteval, as well as after.
      .addPass(createRaiseSpecialOps)
      // We decompose and transpose concatenations immediately before folding
      // unit extent dims because this allows decoupling unit dims in the
      // concatenation from the transposes that are introduced.
      .addPass([&]() {
        return createDecomposeConcatPass(
            transformOptions.options.outerDimConcat);
      })
      // We generalize certain named ops immediately before folding unit extent
      // dims as the unit dim folding pass updates indexing maps and is better
      // at working with generics. By this point we have already done any
      // specialized raising and the op names are no longer useful.
      .addPass(createGeneralizeLinalgNamedOpsPass);
  mainPassManager.addPass(IREE::Flow::createFoldUnitExtentDimsPass());
  FunctionLikeNest(mainPassManager)
      .addPredicatedPass(clEnableFuseSiluHorizontalMatmul,
                         createFuseSiluHorizontalMatmulPass)
      .addPredicatedPass(clEnableDemoteContractionInputsToBF16,
                         createDemoteContractionInputsToBF16Pass)
      .addPass([&]() {
        return createFuseDequantizationMatmulPass(
            clEnableQuantizedMatmulReassociation);
      })
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      // Propagate transposes immediately before set encoding/data tiling
      // because transpose propagation cannot take an opinion on the preferred
      // layout of various operations. This simplifies local propagation
      // decisions as SetEncoding is expected to pick the ideal layout for
      // that operation anyway, and this way we only need to make such a
      // decision once.
      .addPredicatedPass(
          clEnableTransposePropagation,
          [&]() {
            return createPropagateLinalgTransposePass(
                transformOptions.options.aggressiveTransposePropagation);
          })
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Enable data tiling after they are in a canonical form.
  if (transformOptions.options.dataTiling) {
    mainPassManager.addPass(createSetEncodingPass());
    mainPassManager.addPass(createMaterializeHomogeneousEncodingsPass());
    mainPassManager.addPass(createCanonicalizerPass());
    mainPassManager.addPass(createCSEPass());
    mainPassManager.addPass(createSimplifyPackUnpackPass());
    if (transformOptions.options.propagateGlobalLayout) {
      mainPassManager.addPass(createPropagateDataLayoutPass());
      mainPassManager.addPass(createCSEPass());
    }
  }
  // Generalize transposes and any other remaining named linalg ops that can
  // now be represented as generics.
  FunctionLikeNest(mainPassManager).addPass(createGeneralizeLinalgNamedOpsPass);

  // Hoist loop invariants (e.g. from scf loops) with zero-trip-check.
  FunctionLikeNest(mainPassManager)
      .addPass(createGlobalLoopInvariantCodeMotionPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  OpPassManager pipeline(ModuleOp::getOperationName());
  FunctionLikeNest(pipeline)
      // Simplify util.global accesses early on; this can help with dispatch
      // region formation as redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other
  // util ops).
  pipeline.addPass(IREE::Util::createApplyPatternsPass());
  pipeline.addPass(IREE::Util::createFoldGlobalsPass());
  pipeline.addPass(IREE::Util::createIPOPass());
  pipeline.addPass(createCanonicalizerPass());
  pipeline.addPass(createCSEPass());

  if (transformOptions.options.constExprHoisting) {
    buildGlobalOptExprHoistingPassPipeline(pipeline, transformOptions);
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(pipeline);
  }

  // Export after const-eval. If the user wants to keep the input constants
  // as is in the final parameter archive, they will probably want to disable
  // const-eval, or could run this pass as preprocessing. There might be a
  // configuration in the future where users want to limit const-eval to smaller
  // constants that aren't exported and skip it for larger parameters, but this
  // is a sensible place for the common case of wanting const-eval in the final
  // artifact + archive.
  if (!transformOptions.options.parameterArchiveExportPath.empty()) {
    IREE::IO::Parameters::ExportParametersPassOptions exportParametersOptions;
    exportParametersOptions.archivePath =
        transformOptions.options.parameterArchiveExportPath;
    exportParametersOptions.parameterScope =
        transformOptions.options.parameterExportScope;
    exportParametersOptions.minimumSize =
        transformOptions.options.minimumParameterExportSize;
    pipeline.addPass(IREE::IO::Parameters::createExportParametersPass(
        exportParametersOptions));
  }

  if (!transformOptions.options.splatParameterArchiveExportPath.empty()) {
    IREE::IO::Parameters::GenerateSplatParameterArchivePassOptions
        generateSplatOptions;
    generateSplatOptions.archivePath =
        transformOptions.options.splatParameterArchiveExportPath;
    pipeline.addPass(
        IREE::IO::Parameters::createGenerateSplatParameterArchivePass(
            generateSplatOptions));
  }

  if (transformOptions.options.numericPrecisionReduction) {
    pipeline.addPass(createInferNumericNarrowingPass());
    pipeline.addPass(createOptimizeNumericsPass());
    pipeline.addPass(createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(pipeline)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Add the whole fixed point iterator.
  mainPassManager.addPass(
      IREE::Util::createFixedPointIteratorPass(std::move(pipeline)));

  FunctionLikeNest(mainPassManager)
      // After running const-eval to a fixed point and folding unit extent dims,
      // try any new raising opportunities.
      .addPass(createRaiseSpecialOps)
      // Strip std.assert & co after we perform optimizations; prior to this we
      // may use the assertions to derive information during analysis.
      .addPredicatedPass(transformOptions.options.stripAssertions,
                         IREE::Util::createStripDebugOpsPass);
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerGlobalOptimizationPipeline() {
  registerPasses();

  PassPipelineRegistration<TransformOptions>
      globalOptimizationTransformPassPipeline(
          "iree-global-optimization-transformation-pipeline",
          "Runs the IREE global optimization transformation pipeline",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildGlobalOptimizationPassPipeline(passManager, transformOptions);
          });
  PassPipelineRegistration<TransformOptions>
      globalOptimizationConstantHoistingPassPipeline(
          "iree-global-optimization-hoist-constant-expressions",
          "Hoists constant expressions with the preferred storage types for "
          "global optimization",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildGlobalOptExprHoistingPassPipeline(passManager,
                                                   transformOptions);
          });
}

} // namespace mlir::iree_compiler::GlobalOptimization
