// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEAGGREGATEDOPPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {
struct DecomposeAggregatedOpPass final
    : impl::DecomposeAggregatedOpPassBase<DecomposeAggregatedOpPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::LinalgExt::IREELinalgExtDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;

private:
  FailureOr<StringSet<>> parseFilterOps();
};
} // namespace

void DecomposeAggregatedOpPass::runOnOperation() {
  FailureOr<StringSet<>> filterResult = parseFilterOps();
  if (failed(filterResult))
    return;
  StringSet<> filter = filterResult.value();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation().walk([&](linalg::AggregatedOpInterface aggregatedOp) {
    if (!filter.empty() &&
        !filter.contains(aggregatedOp->getName().getStringRef())) {
      return;
    }
    rewriter.setInsertionPoint(aggregatedOp);
    FailureOr<SmallVector<Value>> results =
        aggregatedOp.decomposeOperation(rewriter);
    if (failed(results)) {
      if (failed(results)) {
        aggregatedOp->emitOpError("Could not decompose ")
            << aggregatedOp->getName();
        return signalPassFailure();
      }
    }
    rewriter.replaceOp(aggregatedOp, results.value());
  });
}

FailureOr<StringSet<>> DecomposeAggregatedOpPass::parseFilterOps() {
  if (filterOps.empty()) {
    return llvm::StringSet{};
  }
  StringSet<> ret{llvm::from_range, llvm::split(filterOps, ',')};
  // todo: validate the string
  return ret;
}

} // namespace mlir::iree_compiler::IREE::LinalgExt