// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVEEXPORTORDINALSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

struct ResolveCommandBufferDispatchOrdinals
    : public OpRewritePattern<IREE::HAL::CommandBufferDispatchSymbolOp> {
  using OpRewritePattern<
      IREE::HAL::CommandBufferDispatchSymbolOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::CommandBufferDispatchSymbolOp op,
                                PatternRewriter &rewriter) const override {
    auto symbol = SymbolTable::lookupNearestSymbolFrom(op, op.getEntryPoint());
    assert(symbol && "missing ExecutableEntryPoint symbol");
    auto exportOp = cast<IREE::HAL::ExecutableExportOp>(symbol);

    // Lookup the device for our command buffer, then the executable from the
    // exports nested reference.
    auto device = rewriter.createOrFold<IREE::HAL::CommandBufferDeviceOp>(
        op.getLoc(), IREE::HAL::DeviceType::get(rewriter.getContext()),
        op.getCommandBuffer());
    auto executableOp = dyn_cast<IREE::HAL::ExecutableOp>(
        exportOp->getParentOp()->getParentOp());
    auto executable = rewriter.createOrFold<IREE::HAL::ExecutableLookupOp>(
        op.getLoc(), device, executableOp.getSymName());

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferDispatchOp>(
        op, op.getCommandBuffer(), executable, exportOp.getOrdinalAttr(),
        op.getWorkgroupX(), op.getWorkgroupY(), op.getWorkgroupZ());
    return success();
  }
};

struct ResolveCommandBufferDispatchIndirectOrdinals
    : public OpRewritePattern<
          IREE::HAL::CommandBufferDispatchIndirectSymbolOp> {
  using OpRewritePattern<
      IREE::HAL::CommandBufferDispatchIndirectSymbolOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::CommandBufferDispatchIndirectSymbolOp op,
                  PatternRewriter &rewriter) const override {
    auto symbol = SymbolTable::lookupNearestSymbolFrom(op, op.getEntryPoint());
    assert(symbol && "missing ExecutableEntryPoint symbol");
    auto exportOp = cast<IREE::HAL::ExecutableExportOp>(symbol);

    // Lookup the device for our command buffer, then the executable from the
    // exports nested reference.
    auto device = rewriter.createOrFold<IREE::HAL::CommandBufferDeviceOp>(
        op.getLoc(), IREE::HAL::DeviceType::get(rewriter.getContext()),
        op.getCommandBuffer());
    auto executableOp = dyn_cast<IREE::HAL::ExecutableOp>(
        exportOp->getParentOp()->getParentOp());
    auto executable = rewriter.createOrFold<IREE::HAL::ExecutableLookupOp>(
        op.getLoc(), device, executableOp.getSymName());

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferDispatchIndirectOp>(
        op, op.getCommandBuffer(), executable, exportOp.getOrdinalAttr(),
        op.getWorkgroupsBuffer(), op.getWorkgroupsOffset());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-export-ordinals
//===----------------------------------------------------------------------===//

struct ResolveExportOrdinalsPass
    : public IREE::HAL::impl::ResolveExportOrdinalsPassBase<
          ResolveExportOrdinalsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ResolveCommandBufferDispatchOrdinals>(context);
    patterns.insert<ResolveCommandBufferDispatchIndirectOrdinals>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
