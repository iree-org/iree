// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Preprocessing/Passes.h"

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

void buildPreprocessingPassPipeline(
    OpPassManager &passManager,
    const PreprocessingOptions &preprocessingOptions) {
  if (preprocessingOptions.preprocessingPassPipeline.empty()) {
    return;
  }
  (void)parsePassPipeline(preprocessingOptions.preprocessingPassPipeline,
                          passManager);
}

void registerPreprocessingPasses() { registerCommonPreprocessingPasses(); }

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
