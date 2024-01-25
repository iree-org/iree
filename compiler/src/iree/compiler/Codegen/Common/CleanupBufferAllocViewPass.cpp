// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferAllocViewCleanUpPass.cpp -------------------------------------===//
//
// This pass performs canonicalizations/cleanups related to HAL interface/buffer
// allocations and views. We need a dedicated pass because patterns here involve
// multiple dialects.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Runs canonicalization patterns on interface load/store ops.
struct CleanupBufferAllocViewPass
    : public CleanupBufferAllocViewBase<CleanupBufferAllocViewPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateReshapeToInterfaceTensorPatterns(patterns);
    populateRemoveDeadMemAllocPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCleanupBufferAllocViewPass() {
  return std::make_unique<CleanupBufferAllocViewPass>();
}

} // namespace mlir::iree_compiler
