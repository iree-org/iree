// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GeneralizeLinalgOps.cpp - Pass to generalize named LinalgOps -------==//
//
// The pass is to generalize Linalg named operations that are better off being
// represented as `linalg.generic` operations in IREE.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-global-opt-generalize-linalg-named-ops"

namespace mlir::iree_compiler::GlobalOptimization {

// TODO(#21955): Adapts the convolution transformations to work with generalized
// linalg.generic form, but not only when they are named ops. E.g.,
// DownscaleSizeOneWindowed2DConvolution patterns.
static llvm::cl::opt<bool> clDisableConvGeneralization(
    "iree-global-opt-experimental-disable-conv-generalization",
    llvm::cl::desc("Disable generalization for some conv ops (experimental)."),
    llvm::cl::init(false));

#define GEN_PASS_DEF_GENERALIZELINALGNAMEDOPSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {
struct GeneralizeLinalgNamedOpsPass
    : public impl::GeneralizeLinalgNamedOpsPassBase<
          GeneralizeLinalgNamedOpsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void GeneralizeLinalgNamedOpsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  SmallVector<linalg::LinalgOp> namedOpCandidates;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp) ||
        isa<linalg::GenericOp>(linalgOp)) {
      return;
    }
    if (enableGeneralizeMatmul && linalg::isaContractionOpInterface(linalgOp)) {
      namedOpCandidates.push_back(linalgOp);
      return;
    }
    bool generalizeConvOps = linalg::isaConvolutionOpInterface(linalgOp);
    if (clDisableConvGeneralization &&
        isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
            linalg::PoolingNhwcSumOp, linalg::PoolingNhwcMaxOp,
            linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
            linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNchwSumOp,
            linalg::PoolingNchwMaxOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            linalgOp)) {
      generalizeConvOps = false;
    }
    if (isa_and_nonnull<linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp,
                        linalg::CeilOp, linalg::CopyOp, linalg::DivOp,
                        linalg::DivUnsignedOp, linalg::ExpOp, linalg::FloorOp,
                        linalg::LogOp, linalg::MapOp, linalg::MaxOp,
                        linalg::MulOp, linalg::NegFOp, linalg::ReduceOp,
                        linalg::SubOp, linalg::TransposeOp>(
            linalgOp.getOperation()) ||
        generalizeConvOps) {
      namedOpCandidates.push_back(linalgOp);
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto linalgOp : namedOpCandidates) {
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<linalg::GenericOp> generalizedOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize operation");
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::GlobalOptimization
