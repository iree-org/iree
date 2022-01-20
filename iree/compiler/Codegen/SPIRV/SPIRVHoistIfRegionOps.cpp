// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-hoist-if-region-ops"

namespace mlir {
namespace iree_compiler {

struct HoistIfRegionOps final
    : public SPIRVHoistIfRegionOpsBase<HoistIfRegionOps> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    auto hoistInInnermostIfOp = [](Operation *candidate) {
      // Only hoist index calculation ops for now.
      if (!isa<arith::AddIOp, arith::MulIOp, arith::CmpIOp, arith::SelectOp>(
              candidate)) {
        return false;
      }

      // Only hoist if it is in the innermost scf.if op.
      auto ifOp = dyn_cast<scf::IfOp>(candidate->getParentOp());
      for (Operation &op : *ifOp.thenBlock()) {
        if (isa<scf::IfOp, scf::ForOp>(op)) return false;
      }
      if (ifOp.elseBlock()) {
        for (Operation &op : *ifOp.elseBlock()) {
          if (isa<scf::IfOp, scf::ForOp>(op)) return false;
        }
      }
      return true;
    };
    scf::populateIfRegionHoistingPatterns(patterns, hoistInInnermostIfOp);

    FuncOp funcOp = getOperation();
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createSPIRVHoistIfRegionOpsPass() {
  return std::make_unique<HoistIfRegionOps>();
}

}  // namespace iree_compiler
}  // namespace mlir
