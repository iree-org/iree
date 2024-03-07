// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Preprocessing/Passes.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
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

void buildPreprocessingPassPipeline(
    OpPassManager &passManager,
    const PreprocessingOptions &preprocessingOptions,
    PipelineExtensions *pipelineExtensions) {
  auto pipelineStr = preprocessingOptions.preprocessingPassPipeline;
  if (!pipelineStr.empty()) {
    extendWithTextPipeline(passManager, pipelineStr);
  }

  if (pipelineExtensions) {
    pipelineExtensions->extendPreprocessingPassPipeline(passManager);
  }

  if (!preprocessingOptions.preprocessingTransformSpecFilename.empty()) {
    Preprocessing::InterpreterPassOptions interpreterOptions;
    interpreterOptions.transformSpecPath =
        preprocessingOptions.preprocessingTransformSpecFilename;
    passManager.addPass(
        Preprocessing::createInterpreterPass(interpreterOptions));
  }
}

static void
buildTransposeConvolutionPassPipeline(OpPassManager &passManager,
                                      const TransformOptions &options) {
  FunctionLikeNest(passManager)
      .addPass(GlobalOptimization::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(GlobalOptimization::createConvert1X1FilterConv2DToMatmulPass)
      .addPass(createConvertConvToChannelsLastPass);
  passManager.addPass(IREE::Flow::createFoldUnitExtentDimsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void registerPreprocessingPasses() {
  registerCommonPreprocessingPasses();

  PassPipelineRegistration<TransformOptions>
      globalOptimizationTransformPassPipeline(
          "iree-preprocessing-transpose-convolution-pipeline",
          "Runs a pass pipeline for transposing and canonicalizing "
          "convolutions",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildTransposeConvolutionPassPipeline(passManager,
                                                  transformOptions);
          });
}

} // namespace mlir::iree_compiler::Preprocessing
