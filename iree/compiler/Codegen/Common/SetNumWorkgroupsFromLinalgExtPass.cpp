// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Utils.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
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

/// Forward LinalgExt::InParallel -> Tensor::InsertSlice -> Flow::TensorStore.
/// This pattern is necessary for correctness, it accounts for the fact that
/// InParallel is distributed across multiple workgroups when lowering to HAL
/// but it then connects to a sequential tensor.insert_slice and then to
/// flow.dispatch.tensor_store.
///
// TODO: All the rewrites in this file this should be done as part of InParallel
// -> HAL rewrite. But because of dialect dependencies and layering, we have
// some phase ordering that prevents it atm.
class ForwardInParallelResultToFlow
    : public OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
 public:
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto insertSliceOp = op.value().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) return failure();

    // TODO: this should be done as part of InParallel -> HAL rewrite.
    // But because of dialect dependencies and layering, we have some phase
    // ordering that prevents it atm. It does not make sense to move the pattern
    // because of this temporary layering problem, so we just ignore the
    // condition for now.
    //
    // auto inParallelOp =
    //     insertSliceOp.source().getDefiningOp<IREE::LinalgExt::InParallelOp>();
    // if (!inParallelOp) return failure();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // `tensor.insert_slice` (i.e. the producer) folds **into**
    // `flow.dispatch.tensor.store` (i.e. the consumer).
    if (failed(foldOffsetsSizesAndStrides(rewriter, op.getLoc(), insertSliceOp,
                                          op, offsets, sizes, strides)))
      return failure();
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        op, insertSliceOp.source(), op.target(), op.target_dims(), offsets,
        sizes, strides);

    return success();
  }
};

}  // namespace

void SetNumWorkgroupsFromLinalgExtPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp module = variantOp.getInnerModule();

  // Perform forwarding patterns to bridge the tensor / flow gap.
  // This is necessary for correctness.
  // TODO: given existing bufferization tricks, this may trigger unnecessary
  // copies that need to be further investigated.
  RewritePatternSet forwardPatterns(context);
  forwardPatterns.insert<ForwardInParallelResultToFlow>(context);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(forwardPatterns))))
    return signalPassFailure();

  // Apply post-distribution canonicalization passes.
  RewritePatternSet canonicalization(context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalization, context);
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
