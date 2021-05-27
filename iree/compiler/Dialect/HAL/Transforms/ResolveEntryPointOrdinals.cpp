// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class ResolveCommandBufferDispatchOrdinals
    : public OpRewritePattern<IREE::HAL::CommandBufferDispatchSymbolOp> {
 public:
  using OpRewritePattern<
      IREE::HAL::CommandBufferDispatchSymbolOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::CommandBufferDispatchSymbolOp op,
                                PatternRewriter &rewriter) const override {
    auto entryPointOp = dyn_cast<IREE::HAL::ExecutableEntryPointOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.entry_point()));

    // Lookup the device for our command buffer, then the executable from the
    // entry point's nested reference.
    auto device = rewriter.createOrFold<IREE::HAL::CommandBufferDeviceOp>(
        op.getLoc(), IREE::HAL::DeviceType::get(rewriter.getContext()),
        op.command_buffer());
    auto executableOp = dyn_cast<IREE::HAL::ExecutableOp>(
        entryPointOp->getParentOp()->getParentOp());
    auto executable = rewriter.createOrFold<IREE::HAL::ExecutableLookupOp>(
        op.getLoc(), device, executableOp.sym_name());

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferDispatchOp>(
        op, op.command_buffer(), executable, entryPointOp.ordinalAttr(),
        op.workgroup_x(), op.workgroup_y(), op.workgroup_z());
    return success();
  }
};

class ResolveCommandBufferDispatchIndirectOrdinals
    : public OpRewritePattern<
          IREE::HAL::CommandBufferDispatchIndirectSymbolOp> {
 public:
  using OpRewritePattern<
      IREE::HAL::CommandBufferDispatchIndirectSymbolOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::CommandBufferDispatchIndirectSymbolOp op,
      PatternRewriter &rewriter) const override {
    auto entryPointOp = dyn_cast<IREE::HAL::ExecutableEntryPointOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.entry_point()));

    // Lookup the device for our command buffer, then the executable from the
    // entry point's nested reference.
    auto device = rewriter.createOrFold<IREE::HAL::CommandBufferDeviceOp>(
        op.getLoc(), IREE::HAL::DeviceType::get(rewriter.getContext()),
        op.command_buffer());
    auto executableOp = dyn_cast<IREE::HAL::ExecutableOp>(
        entryPointOp->getParentOp()->getParentOp());
    auto executable = rewriter.createOrFold<IREE::HAL::ExecutableLookupOp>(
        op.getLoc(), device, executableOp.sym_name());

    rewriter.replaceOpWithNewOp<IREE::HAL::CommandBufferDispatchIndirectOp>(
        op, op.command_buffer(), executable, entryPointOp.ordinalAttr(),
        op.workgroups_buffer(), op.workgroups_offset());
    return success();
  }
};

class ResolveEntryPointOrdinalsPass
    : public PassWrapper<ResolveEntryPointOrdinalsPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ResolveCommandBufferDispatchOrdinals>(context);
    patterns.insert<ResolveCommandBufferDispatchIndirectOrdinals>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createResolveEntryPointOrdinalsPass() {
  return std::make_unique<ResolveEntryPointOrdinalsPass>();
}

static PassRegistration<ResolveEntryPointOrdinalsPass> pass(
    "iree-hal-resolve-entry-point-ordinals",
    "Resolves hal.executable.entry_point references to ordinals");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
