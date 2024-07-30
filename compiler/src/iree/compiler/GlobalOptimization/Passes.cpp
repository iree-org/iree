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

// TODO(hanchung): Remove the flag. We don't want to do early materialization by
// default. Because it won't work for heterogeneous computing. This is not the
// right layer for handling such information.
static llvm::cl::opt<bool> clEnableEarlyMaterialization(
    "iree-global-opt-enable-early-materialization",
    llvm::cl::desc(
        "Enables early materialization on encodings. Note, this flag should be "
        "false eventually. This does not work for heterogeneous computing."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableFuseHorizontalContractions(
    "iree-global-opt-enable-fuse-horizontal-contractions",
    llvm::cl::desc(
        "Enables horizontal fusion of contractions with one common operand"),
    llvm::cl::init(false));

static llvm::cl::opt<DemotionOption> clDemoteContractionInputsToBF16Strategy(
    "iree-global-opt-enable-demote-contraction-inputs-to-bf16",
    llvm::cl::desc("Demotes inputs (LHS, RHS) of contraction ops to BF16. "
                   "Selects types of contraction ops to demote."),
    llvm::cl::values(
        clEnumValN(DemotionOption::All, "all", "Demote all contraction ops."),
        clEnumValN(DemotionOption::Conv, "conv",
                   "Only demote convolution ops."),
        clEnumValN(DemotionOption::Matmul, "matmul", "Only demote matmul ops."),
        clEnumValN(DemotionOption::None, "none", "Demote no contraction ops.")),
    llvm::cl::init(DemotionOption::None));

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
  // Import parameters before any global optimization passes so that the inlined
  // parameters are available for folding.
  if (!transformOptions.options.parameterImportPaths.empty()) {
    IREE::IO::Parameters::ImportParametersPassOptions importParametersOptions;
    importParametersOptions.scopePaths =
        transformOptions.options.parameterImportPaths;
    importParametersOptions.keys = transformOptions.options.parameterImportKeys;
    importParametersOptions.maximumSize =
        transformOptions.options.parameterImportMaximumSize;
    mainPassManager.addPass(IREE::IO::Parameters::createImportParametersPass(
        importParametersOptions));
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
      .addPass([&]() {
        return createDemoteContractionInputsToBF16Pass(
            clDemoteContractionInputsToBF16Strategy);
      })
      .addPass([&]() {
        return createFuseDequantizationMatmulPass(
            clEnableQuantizedMatmulReassociation);
      })
      .addPass(IREE::Flow::createCanonicalizerPass)
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
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  if (clEnableFuseHorizontalContractions) {
    FunctionLikeNest(mainPassManager)
        .addPass(createFuseHorizontalContractionsPass)
        .addPass(IREE::Flow::createCanonicalizerPass)
        .addPass(mlir::createCSEPass);
  }

  // Enable data tiling after they are in a canonical form.
  if (transformOptions.options.dataTiling) {
    // TODO(hanchung): Make data-tiling passes be FunctionOpInterface pass, so
    // we can use `FunctionLikNest` here.
    // TODO(hanchung): Make it controlable through flags. It is fine for now
    // because it is an experimental path.
    const int64_t kPadFactor = clEnableEarlyMaterialization ? 0 : 16;
    mainPassManager.addPass(createSetEncodingPass(kPadFactor));
    if (clEnableEarlyMaterialization) {
      mainPassManager.addPass(createMaterializeHomogeneousEncodingsPass());
    }
    mainPassManager.addPass(IREE::Flow::createCanonicalizerPass());
    mainPassManager.addPass(createCSEPass());
    mainPassManager.addPass(createSimplifyPackUnpackPass());
    FunctionLikeNest(mainPassManager).addPass(createDataLayoutPropagationPass);
  }
  // Generalize transposes and any other remaining named linalg ops that can
  // now be represented as generics.
  FunctionLikeNest(mainPassManager).addPass(createGeneralizeLinalgNamedOpsPass);

  // Hoist loop invariants (e.g. from scf loops) with zero-trip-check.
  FunctionLikeNest(mainPassManager)
      .addPass(createGlobalLoopInvariantCodeMotionPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Simplify util.global accesses early on; this can help with dispatch
  // region formation as redundant store-loads are removed.
  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other
  // util ops).
  mainPassManager.addPass(IREE::Util::createApplyPatternsPass());
  mainPassManager.addPass(IREE::Util::createFoldGlobalsPass());
  mainPassManager.addPass(IREE::Util::createIPOPass());
  mainPassManager.addPass(IREE::Flow::createCanonicalizerPass());
  mainPassManager.addPass(createCSEPass());

  if (transformOptions.options.constExprHoisting) {
    buildGlobalOptExprHoistingPassPipeline(mainPassManager, transformOptions);
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(mainPassManager);
  }

  if (transformOptions.options.numericPrecisionReduction) {
    mainPassManager.addPass(createInferNumericNarrowingPass());
    mainPassManager.addPass(createOptimizeNumericsPass());
    mainPassManager.addPass(createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  FunctionLikeNest(mainPassManager)
      // After running const-eval to a fixed point and folding unit extent dims,
      // try any new raising opportunities.
      .addPass(createRaiseSpecialOps)
      // Strip std.assert & co after we perform optimizations; prior to this we
      // may use the assertions to derive information during analysis.
      .addPredicatedPass(transformOptions.options.stripAssertions,
                         IREE::Util::createStripDebugOpsPass);

  // Export after const-eval. If the user wants to keep the input constants
  // as is in the final parameter archive, they will probably want to disable
  // const-eval, or could run this pass as preprocessing. There might be a
  // configuration in the future where users want to limit const-eval to smaller
  // constants that aren't exported and skip it for larger parameters, but this
  // is a sensible place for the common case of wanting const-eval in the final
  // artifact + archive.
  if (!transformOptions.options.parameterExportPath.empty()) {
    IREE::IO::Parameters::ExportParametersPassOptions exportParametersOptions;
    exportParametersOptions.scopePath =
        transformOptions.options.parameterExportPath;
    exportParametersOptions.minimumSize =
        transformOptions.options.parameterExportMinimumSize;
    mainPassManager.addPass(IREE::IO::Parameters::createExportParametersPass(
        exportParametersOptions));
  }

  if (!transformOptions.options.parameterSplatExportFile.empty()) {
    IREE::IO::Parameters::GenerateSplatParameterArchivePassOptions
        generateSplatOptions;
    generateSplatOptions.filePath =
        transformOptions.options.parameterSplatExportFile;
    mainPassManager.addPass(
        IREE::IO::Parameters::createGenerateSplatParameterArchivePass(
            generateSplatOptions));
  }
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
