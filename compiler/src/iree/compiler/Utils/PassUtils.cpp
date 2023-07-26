// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/PassUtils.h"

#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-utils"

namespace mlir {
namespace iree_compiler {

void signalFixedPointModified(Operation *rootOp) {
  MLIRContext *context = rootOp->getContext();
  if (!rootOp->hasAttr("iree.fixedpoint.iteration")) {
    LLVM_DEBUG(llvm::dbgs() << "Not signaling fixed-point modification: not "
                               "running under fixed point iterator");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Signalling fixed-point iterator modification");
  rootOp->setAttr("iree.fixedpoint.modified", UnitAttr::get(context));
}

// Extends the pass manager with the given textual pipeline, following standard
// MLIR `--pass-pipeline` syntax.
void extendWithTextPipeline(OpPassManager &passManager, StringRef textPipeline,
                            StringRef pipelineDebugType) {
  StringRef orig = textPipeline;
  // Strip the `builtin.module(...)` that surrounds the pass pipeline
  // description. On failure an assertion is triggered, but in release builds
  // it just will silently return and not raise an error. There is no
  // way to handle the error in caller currently.
  size_t pos = textPipeline.find_first_of("(");
  if (pos == StringRef::npos) {
    llvm::errs() << "ERROR: expected " << pipelineDebugType
                 << " pass pipeline string to be "
                    "nested within `builtin.module(..)`; got `"
                 << orig << "`\n";
    return;
  }
  if (textPipeline.substr(0, pos) != "builtin.module") {
    llvm::errs() << "ERROR: expected " << pipelineDebugType
                 << " pass pipeline string to be "
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
    llvm::dbgs() << pipelineDebugType << " pass pipeline : ";
    passManager.printAsTextualPipeline(llvm::dbgs());
  });
}

} // namespace iree_compiler
} // namespace mlir
