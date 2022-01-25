// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"

#include <memory>

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TFLite {

void buildTransformPassPipeline(OpPassManager &passManager) {
  // Wraps the entry points in a "_tflite_xx" function and adds shape support.
  passManager.addPass(createWrapEntryPointsPass());

  // Cleanup the IR after manipulating it.
  passManager.addPass(createInlinerPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
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

}  // namespace TFLite
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
