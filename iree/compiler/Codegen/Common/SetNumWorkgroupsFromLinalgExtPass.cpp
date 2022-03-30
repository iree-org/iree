// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
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

#define DEBUG_TYPE "iree-codegen-set-num-workgroups-from-linalg-ext"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

namespace mlir {
namespace iree_compiler {

namespace {
class SetNumWorkgroupsFromLinalgExtPass
    : public SetNumWorkgroupsFromLinalgExtBase<
          SetNumWorkgroupsFromLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::HAL::HALDialect, IREELinalgExtDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

int64_t getBuilderArgs(HALInterfaceWorkgroupIDOp op) {
  return op.dimension().getZExtValue();
}

int64_t getBuilderArgs(HALInterfaceWorkgroupCountOp op) {
  return op.dimension().getZExtValue();
}

ValueRange getBuilderArgs(HALReturnOp op) { return op.getOperands(); }

/// Generic implementation of one-to-one conversion from "SourceOp" to
/// "TargetOp".
template <typename SourceOp, typename TargetOp>
class OneToOneRewritePattern : public OpRewritePattern<SourceOp> {
 public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, getBuilderArgs(op));
    return success();
  }
};

}  // namespace

void SetNumWorkgroupsFromLinalgExtPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp module = variantOp.getInnerModule();

  // Perform 1-1 rewrites first: after the ExecutableEntryPointOp is
  // modified this will be more annoying to track.
  RewritePatternSet oneToOneRewrites(context);
  oneToOneRewrites
      .insert<OneToOneRewritePattern<HALInterfaceWorkgroupIDOp,
                                     IREE::HAL::InterfaceWorkgroupIDOp>,
              OneToOneRewritePattern<HALInterfaceWorkgroupCountOp,
                                     IREE::HAL::InterfaceWorkgroupCountOp>,
              OneToOneRewritePattern<HALReturnOp, IREE::HAL::ReturnOp>>(
          context);
  if (failed(
          applyPatternsAndFoldGreedily(module, std::move(oneToOneRewrites)))) {
    return signalPassFailure();
  }

  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPoints =
      getAllEntryPoints(module);
  for (auto funcOp : module.getOps<FuncOp>()) {
    auto entryPointOp = entryPoints.lookup(funcOp.getName());
    if (!entryPointOp) continue;

    bool numWorkgroupIsSet = false;
    assert(entryPointOp.workgroup_count_region().empty() &&
           "Expected a single entryPoint op with no regions");

    funcOp->walk([&](HALExecutableEntryPointOp op) {
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
      rewriter.mergeBlocks(&op.workgroup_count_region().front(), &block);
      // TODO: Don't add args post-hoc and instead replace them during
      // `mergeBlocks`.
      block.addArgument(rewriter.getIndexType(), op->getLoc());
      block.addArgument(rewriter.getIndexType(), op->getLoc());
      block.addArgument(rewriter.getIndexType(), op->getLoc());
      op->erase();
      entryPointOp.erase();
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
