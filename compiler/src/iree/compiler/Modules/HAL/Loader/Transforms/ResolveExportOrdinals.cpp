// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/PassDetail.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Loader {

struct ResolveExecutableDispatchSymbolOp
    : public OpRewritePattern<IREE::HAL::Loader::ExecutableDispatchSymbolOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::Loader::ExecutableDispatchSymbolOp op,
                  PatternRewriter &rewriter) const override {
    auto symbol = SymbolTable::lookupNearestSymbolFrom(op, op.getEntryPoint());
    assert(symbol && "missing ExecutableEntryPoint symbol");
    auto exportOp = cast<IREE::HAL::ExecutableExportOp>(symbol);
    rewriter.replaceOpWithNewOp<IREE::HAL::Loader::ExecutableDispatchOp>(
        op, op.getExecutable(), exportOp.getOrdinalAttr(), op.getWorkgroupX(),
        op.getWorkgroupY(), op.getWorkgroupZ(), op.getPushConstants(),
        op.getBindingBuffers(), op.getBindingOffsets(), op.getBindingLengths());
    return success();
  }
};

class ResolveExportOrdinalsPass
    : public ResolveExportOrdinalsBase<ResolveExportOrdinalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::Loader::HALLoaderDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ResolveExecutableDispatchSymbolOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createResolveExportOrdinalsPass() {
  return std::make_unique<ResolveExportOrdinalsPass>();
}

static PassRegistration<ResolveExportOrdinalsPass> pass;

} // namespace Loader
} // namespace mlir::iree_compiler::IREE::HAL
