// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::TFLite {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

void buildTransformPassPipeline(OpPassManager &passManager) {
  // Wraps the entry points in a "_tflite_xx" function and adds shape support.
  passManager.addPass(createWrapEntryPointsPass());

  // Cleanup the IR after manipulating it.
  passManager.addPass(createInlinerPass());
  FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
  FunctionLikeNest(passManager).addPass(createCSEPass);
  passManager.addPass(createSymbolDCEPass());
}

void registerTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-tflite-transform-pipeline",
      "Runs the TFLite bindings support pipeline",
      [](OpPassManager &passManager) {
        buildTransformPassPipeline(passManager);
      });
}

} // namespace mlir::iree_compiler::IREE::TFLite
