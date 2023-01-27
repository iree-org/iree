// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Preprocessing/Passes.h"

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Pass/PassRegistry.h"

static llvm::cl::opt<std::string> clPreprocessingPassPipeline(
    "iree-preprocessing-pass-pipeline",
    llvm::cl::desc(
        "Passes to run before IREE's Flow pipeline for program pre-processing"),
    llvm::cl::init(""));

namespace mlir {
namespace iree_compiler {
namespace IREE {

void buildPreprocessingPassPipeline(OpPassManager &mainPassManager) {
  if (clPreprocessingPassPipeline.empty()) {
    return;
  }
  (void)parsePassPipeline(clPreprocessingPassPipeline, mainPassManager);
}

void registerPreprocessingPasses() { registerCommonPreprocessingPasses(); }

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
