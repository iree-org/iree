// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMOVEINDEXHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Pass to remove all iree_codegen.index_hint operations by replacing them
/// with their input values.
struct RemoveIndexHintsPass final
    : impl::RemoveIndexHintsPassBase<RemoveIndexHintsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<IREE::Codegen::IndexHintOp> indexHintOps;
    funcOp.walk([&](IREE::Codegen::IndexHintOp hintOp) {
      indexHintOps.push_back(hintOp);
    });

    for (auto hintOp : indexHintOps) {
      hintOp.getResult().replaceAllUsesWith(hintOp.getInput());
      rewriter.eraseOp(hintOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
