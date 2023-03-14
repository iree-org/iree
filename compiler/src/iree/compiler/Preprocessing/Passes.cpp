// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Preprocessing/Passes.h"

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-preprocessing-pass-pipeline"

namespace mlir {
namespace iree_compiler {
namespace IREE {

void buildPreprocessingPassPipeline(
    OpPassManager &passManager,
    const PreprocessingOptions &preprocessingOptions) {
  auto pipelineStr = preprocessingOptions.preprocessingPassPipeline;
  if (pipelineStr.empty()) {
    return;
  }

  // Strip the `builtin.module(...)` that surrounds the pass pipeline
  // description. On failure an assertion is triggered, but in release builds
  // it just will silently return and not raise an error. There is no
  // way to handle the error in caller currently.
  StringRef text(pipelineStr);
  size_t pos = text.find_first_of("(");
  if (pos == StringRef::npos) {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << pipelineStr << "`\n";
    return;
  }
  if (text.substr(0, pos) != "builtin.module") {
    llvm::errs() << "ERROR: expected preprocessing pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << pipelineStr << "`\n";
    return;
  }
  if (text.back() != ')') {
    llvm::errs() << "ERROR: mismatched parenthesis in pass pipeline `"
                 << pipelineStr << "`\n";
    return;
  }
  text = text.substr(pos + 1);
  if (failed(parsePassPipeline(text.drop_back(), passManager))) {
    llvm::errs() << "ERROR: mismatched parenthesis in pass pipeline `"
                 << pipelineStr << "`\n";
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Preprocessing pass pipeline : ";
    passManager.printAsTextualPipeline(llvm::dbgs());
  });
}

void registerPreprocessingPasses() { registerCommonPreprocessingPasses(); }

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
