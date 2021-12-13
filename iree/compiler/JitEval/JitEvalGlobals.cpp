// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/JitEval/PassDetail.h"
#include "iree/compiler/JitEval/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace JitEval {

namespace {

struct JitEvalGlobalsPass : public JitEvalGlobalsBase<JitEvalGlobalsPass> {
  void runOnOperation() override {
    //
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createJitEvalGlobalsPass() {
  return std::make_unique<JitEvalGlobalsPass>();
}

}  // namespace JitEval
}  // namespace iree_compiler
}  // namespace mlir
