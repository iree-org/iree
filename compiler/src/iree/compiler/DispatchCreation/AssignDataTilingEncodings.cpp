// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/MaterializedLayoutTarget.h"
#include "iree/compiler/DispatchCreation/Passes.h"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_ASSIGNDATATILINGENCODINGSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct AssignDataTilingEncodingsPass final
    : impl::AssignDataTilingEncodingsPassBase<AssignDataTilingEncodingsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildDataTilingEncodingPassPipeline(pipeline, getEncodingOptions());
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Assigning encodings again would reinterpret materialized layouts.
    if (getMaterializedLayoutTarget(moduleOp)) {
      return;
    }
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildDataTilingEncodingPassPipeline(pipeline, getEncodingOptions());
    if (failed(runPipeline(pipeline, moduleOp))) {
      signalPassFailure();
    }
  }

private:
  DataTilingEncodingOptions getEncodingOptions() const {
    return {llvm::to_vector(opTypes), encodingOption};
  }
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
