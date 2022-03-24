// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"

#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

void PYDM::buildPostImportPassPipeline(OpPassManager &passManager) {
  passManager.addNestedPass<PYDM::FuncOp>(createVariablesToSSAPass());
  passManager.addNestedPass<PYDM::FuncOp>(createLocalPropagateTypesPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void PYDM::buildLowerToIREEPassPipeline(OpPassManager &passManager,
                                        const LowerToIREEOptions &options) {
  // TODO: Needs to be iterative, support optimization passes, etc.
  passManager.addPass(createLowerIREEPyDMToRTLPass());
  if (options.linkRtlSource) {
    passManager.addPass(createLinkIREEPyDMRTLPass(options.linkRtlSource));
  }
  // TODO: Optimization passes need to be their own pipeline.
  passManager.addPass(createFixateWeakNumericPass());
  passManager.addPass(createCanonicalizerPass());

  // Lowering passes.
  passManager.addPass(createConvertIREEPyDMToIREEPass());

  // Cleanup.
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createSymbolDCEPass());
  passManager.addPass(createCSEPass());
}

namespace PYDM_generated {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h.inc"
} // namespace
} // namespace PYDM_generated

void PYDM::registerPasses() {
  PYDM_generated::registerPasses();
  PassPipelineRegistration<> postImportPassPipeline(
      "pydm-post-import-pipeline",
      "Runs passes to cleanup PyDM immediately post-import",
      [](OpPassManager &passManager) {
        buildPostImportPassPipeline(passManager);
      });

  PassPipelineRegistration<> lowerToIREEPipeline(
      "pydm-lower-to-iree-pipeline",
      "Runs passes to lower PyDM to IREE's input dialects",
      [](OpPassManager &passManager) {
        LowerToIREEOptions options;
        buildLowerToIREEPassPipeline(passManager, options);
      });
}
