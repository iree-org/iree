// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-set-num-workgroups-from-lianlg-ext"

static const unsigned kNumMaxParallelDims = 3;

static constexpr auto kFakeHALEntryPointRegionOpName =
    "fake_hal_entry_point_region";
static constexpr auto kFakeHALReturnOpName = "fake_hal_return";
static constexpr auto kFakeHALWorkgroupIdOpName = "fake_hal_workgroup_id";
static constexpr auto kFakeHALWorkgroupCountOpName = "fake_hal_workgroup_count";

namespace mlir {
namespace iree_compiler {

namespace {
class SetNumWorkgroupsFromLinalgExtPass
    : public SetNumWorkgroupsFromLinalgExtBase<
          SetNumWorkgroupsFromLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::HAL::HALDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void SetNumWorkgroupsFromLinalgExtPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp module = variantOp.getInnerModule();

  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPoints =
      getAllEntryPoints(module);
  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPoints.lookup(funcOp.getName());
    if (!entryPointOp) continue;

    funcOp->walk([](Operation *op) {
      if (op->getName().getStringRef() == kFakeHALWorkgroupIdOpName) {
        ImplicitLocOpBuilder b(op->getLoc(), op);
        Value idx = b.create<IREE::HAL::InterfaceWorkgroupIDOp>(
            op->getAttr("idx").cast<IntegerAttr>().getInt());
        op->getResult(0).replaceAllUsesWith(idx);
        op->erase();
        return;
      }
      if (op->getName().getStringRef() == kFakeHALWorkgroupCountOpName) {
        ImplicitLocOpBuilder b(op->getLoc(), op);
        Value idx = b.create<IREE::HAL::InterfaceWorkgroupCountOp>(
            op->getAttr("idx").cast<IntegerAttr>().getInt());
        op->getResult(0).replaceAllUsesWith(idx);
        op->erase();
        return;
      }
      if (op->getName().getStringRef() == kFakeHALReturnOpName) {
        ImplicitLocOpBuilder b(op->getLoc(), op);
        b.create<IREE::HAL::ReturnOp>(op->getOperands());
        op->erase();
        return;
      }
    });

    bool numWorkgroupIsSet = false;
    assert(entryPointOp.workgroup_count_region().empty() &&
           "Expected a single entryPoint op with no regions");

    funcOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() == kFakeHALEntryPointRegionOpName) {
        assert(op->getNumRegions() == 1 && "Expected single region op");
        assert(!numWorkgroupIsSet);
        numWorkgroupIsSet = true;
        IRRewriter rewriter(op->getContext());
        rewriter.setInsertionPoint(entryPointOp);
        auto clonedEntryPointOp =
            rewriter.create<IREE::HAL::ExecutableEntryPointOp>(
                entryPointOp.getLoc(), entryPointOp.sym_nameAttr(),
                entryPointOp.ordinalAttr(), entryPointOp.layoutAttr(),
                entryPointOp.workgroup_sizeAttr(),
                entryPointOp.workgroup_local_memoryAttr(), 1);
        Block &block =
            clonedEntryPointOp.workgroup_count_region().front().emplaceBlock();
        rewriter.mergeBlocks(&op->getRegion(0).front(), &block);
        // TODO: Don't add args post-hoc and instead replace them when
        // kFakeHALEntryPointRegionOpName supports taking bbArgs itself.
        block.addArgument(rewriter.getIndexType(), op->getLoc());
        block.addArgument(rewriter.getIndexType(), op->getLoc());
        block.addArgument(rewriter.getIndexType(), op->getLoc());
        op->erase();
        entryPointOp.erase();
      }
    });
  }

  // Apply post distribution canonicalization passes.
  RewritePatternSet canonicalization(context);
  AffineMinOp::getCanonicalizationPatterns(canonicalization, context);
  populateAffineMinSCFCanonicalizationPattern(canonicalization);
  IREE::Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                           context);
  if (failed(
          applyPatternsAndFoldGreedily(module, std::move(canonicalization)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSetNumWorkgroupsFromLinalgExtPass() {
  return std::make_unique<SetNumWorkgroupsFromLinalgExtPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
