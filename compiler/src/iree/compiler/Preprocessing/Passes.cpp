// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Preprocessing/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-preprocessing-pass-pipeline"

namespace mlir::iree_compiler::Preprocessing {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

namespace {

void extendWithTextPipeline(OpPassManager &passManager,
                            StringRef textPipeline) {
  StringRef orig = textPipeline;
  // Strip the `builtin.module(...)` that surrounds the pass pipeline
  // description. On failure an assertion is triggered, but in release builds
  // it just will silently return and not raise an error. There is no
  // way to handle the error in caller currently.
  size_t pos = textPipeline.find_first_of("(");
  if (pos == StringRef::npos) {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << orig << "`\n";
    return;
  }
  if (textPipeline.substr(0, pos) != "builtin.module") {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << orig << "`\n";
    return;
  }
  if (textPipeline.back() != ')') {
    llvm::errs() << "ERROR: mismatched parenthesis in pass pipeline `" << orig
                 << "`\n";
    return;
  }
  textPipeline = textPipeline.substr(pos + 1);
  if (failed(parsePassPipeline(textPipeline.drop_back(), passManager))) {
    llvm::errs() << "ERROR: mismatched parenthesis in pass pipeline `" << orig
                 << "`\n";
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Preprocessing pass pipeline : ";
    passManager.printAsTextualPipeline(llvm::dbgs());
  });
}

} // namespace

/// Adds passes to `passManager` based on command line options.
/// Returns `true` if passes were added, `false` otherwise.
static void buildPreprocessingPassPipelineFromCommandLine(
    OpPassManager &passManager,
    const PreprocessingOptions &preprocessingOptions) {
  auto pipelineStr = preprocessingOptions.preprocessingPassPipeline;
  // First preference is explicit pass pipeline.
  if (!pipelineStr.empty()) {
    extendWithTextPipeline(passManager, pipelineStr);
  }
  // Second preference is for transform spec file as a preprocessing recipe.
  if (!preprocessingOptions.preprocessingTransformSpecFilename.empty()) {
    Preprocessing::InterpreterPassOptions interpreterOptions;
    interpreterOptions.transformSpecPath =
        preprocessingOptions.preprocessingTransformSpecFilename;
    passManager.addPass(
        Preprocessing::createInterpreterPass(interpreterOptions));
  }
  // Third preference is for PDL spec file as a preprocessing recipe.
  if (!preprocessingOptions.preprocessingPDLSpecFilename.empty()) {
    Preprocessing::ApplyPDLPatternsPassOptions applyPDLPatternsOptions;
    applyPDLPatternsOptions.patternsFile =
        preprocessingOptions.preprocessingPDLSpecFilename;
    passManager.addPass(
        Preprocessing::createApplyPDLPatternsPass(applyPDLPatternsOptions));
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
  }
}

void buildPreprocessingPassPipeline(
    OpPassManager &passManager,
    const PreprocessingOptions &preprocessingOptions,
    PipelineExtensions *pipelineExtensions) {

  // 1. Highest priority given to command line options.
  buildPreprocessingPassPipelineFromCommandLine(passManager,
                                                preprocessingOptions);

  // 2. Run pre-processing pipelines specified through plugin extensions
  // (when provided).
  if (pipelineExtensions) {
    pipelineExtensions->extendPreprocessingPassPipeline(passManager);
  }

  // 3. Run any pass pipelines specified through the use of
  //    `preprocessing_pipeline` attribute.
  FunctionLikeNest(passManager).addPass(createAttrBasedPipelinePass);
}

static void
buildTransposeConvolutionPassPipeline(OpPassManager &passManager,
                                      const TransformOptions &options) {
  FunctionLikeNest(passManager)
      .addPass(GlobalOptimization::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(GlobalOptimization::createConvert1X1FilterConv2DToMatmulPass)
      .addPass(createConvertConvToChannelsLastPass)
      .addPass(createConvertConvFilterToChannelsLastPass);
  passManager.addPass(DispatchCreation::createFoldUnitExtentDimsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

/// Pass pipeline to make the computation within a function a single dispatch.
/// Note that this expects a `OpPassManager` nested on `FunctionOpInterface`
/// ops.
static void
buildMakeSingleDispatchPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &options) {
  // We generalize certain named ops immediately before folding unit extent
  // dims as the unit dim folding pass updates indexing maps and is better
  // at working with generics.
  passManager.addPass(GlobalOptimization::createGeneralizeLinalgNamedOpsPass());
  passManager.addPass(DispatchCreation::createFoldUnitExtentDimsForFuncPass());
  GlobalOptimization::PropagateLinalgTransposePassOptions transposeOptions;
  transposeOptions.enableConvolutionPropagation = true;
  transposeOptions.enableAggressivePropagation = true;
  passManager.addPass(
      GlobalOptimization::createPropagateLinalgTransposePass(transposeOptions));
  // Generalize transposes and any other remaining named linalg ops that can
  // now be represented as generics.
  passManager.addPass(GlobalOptimization::createGeneralizeLinalgNamedOpsPass());
  passManager.addPass(
      GlobalOptimization::createConvertStridedContractionToContractionPass());
  passManager.addPass(DispatchCreation::createFusionPreprocessingPass());
  passManager.addPass(mlir::createCSEPass());
  DispatchCreation::BubbleUpExpandShapesPassOptions bubbleOptions;
  bubbleOptions.enableBubbleUpExpandShapesAcrossReductionOps = true;
  passManager.addPass(
      DispatchCreation::createBubbleUpExpandShapesPass(bubbleOptions));
  passManager.addPass(DispatchCreation::createElementwiseOpFusionPass(
      DispatchCreation::ElementwiseOpFusionPassOptions{
          /*enableElementWiseFuseMultiReduction=*/true}));
  // After elementwise operation fusion sink reshapes that block
  // producer-consumer fusion.
  passManager.addPass(DispatchCreation::createSinkReshapesPass());
  passManager.addPass(createMakeSingleDispatchForFunctionPass());
}

void registerPreprocessingPasses() {
  registerCommonPreprocessingPasses();

  PassPipelineRegistration<TransformOptions>
      preprocessingTransposeConvolutionPassPipeline(
          "iree-preprocessing-transpose-convolution-pipeline",
          "Runs a pass pipeline for transposing and canonicalizing "
          "convolutions",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildTransposeConvolutionPassPipeline(passManager,
                                                  transformOptions);
          });

  PassPipelineRegistration<TransformOptions>
      preprocessingMakeSingleDispatchPassPipeline(
          "iree-preprocessing-make-single-dispatch",
          "Runs passes to get a single dispatch for a function",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildMakeSingleDispatchPassPipeline(passManager, transformOptions);
          });
}

} // namespace mlir::iree_compiler::Preprocessing
