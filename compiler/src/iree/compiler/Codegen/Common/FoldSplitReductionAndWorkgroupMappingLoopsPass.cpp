// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOLDSPLITREDUCTIONANDWORKGROUPMAPPINGLOOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct FoldSplitReductionAndWorkgroupMappingLoopsPass
    : public impl::FoldSplitReductionAndWorkgroupMappingLoopsPassBase<
          FoldSplitReductionAndWorkgroupMappingLoopsPass> {
  using Base::Base;

  void runOnOperation() override;
};

void FoldSplitReductionAndWorkgroupMappingLoopsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  RewritePatternSet patterns(context);
  populateFoldSplitReductionAndWorkgroupMappingLoops(patterns);
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    op->emitOpError("failed to apply pattern to fold split reduction loop with "
                    "workgroup for all");
    return signalPassFailure();
  }
}

} // namespace

} // namespace mlir::iree_compiler
