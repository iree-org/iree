// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

void buildCodegenConfigurationPreProcessingPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  variantPassManager.addPass(createCreateDispatchConfigPass());
}

void buildCodegenTranslationPostProcessingPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createPropagateDispatchConfigPass());
}

void addCommonTargetExecutablePreprocessingPasses(
    FunctionLikeNest &funcPassManager, bool useDecomposeSoftmaxFusion) {
  funcPassManager.addPass(createTypePropagationPass)
      .addPass(createBubbleUpOrdinalOpsPass)
      .addPass(createBufferizeCopyOnlyDispatchesPass)
      .addPass([&]() {
        return createDecomposeSoftmaxPass(useDecomposeSoftmaxFusion);
      })
      .addPass(IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass);
}

//===---------------------------------------------------------------------===//
// Register Common Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace

void registerCodegenCommonPasses() {
  // Generated.
  registerPasses();
}
} // namespace mlir::iree_compiler
