// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/CodegenStrategy.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

#define DEBUG_TYPE "linalg-codegen-strategy"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

void CodegenStrategy::configurePassPipeline(OpPassManager &pm,
                                            MLIRContext *context,
                                            bool addEnablePass) const {
  for (unsigned stepCount = 0, e = transformationSequence.size(); stepCount < e;
       ++stepCount) {
    const std::unique_ptr<Transformation> &t =
        transformationSequence[stepCount];
    std::string currentStr = std::to_string(stepCount);
    auto currentState = StringAttr::get(context, currentStr);
    std::string nextStr = std::to_string(stepCount + 1);
    auto nextState = StringAttr::get(context, nextStr);
    auto filter = (currentState.str() == std::to_string(0))
                      ? LinalgExt::LinalgTransformationFilter(
                            t->filter, ArrayRef<StringAttr>{}, nextState)
                      : LinalgExt::LinalgTransformationFilter(
                            t->filter, currentState, nextState);
    t->addToPassPipeline(pm, filter);
    if (addEnablePass)
      pm.addPass(createLinalgStrategyEnablePass(linalgEnablingOptions));
  }
  pm.addPass(createLinalgStrategyRemoveMarkersPass());
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
