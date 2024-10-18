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

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_GENERALIZELINALGNAMEDOPSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {
struct GeneralizeLinalgNamedOpsPass
    : public impl::GeneralizeLinalgNamedOpsPassBase<
          GeneralizeLinalgNamedOpsPass> {
  void runOnOperation() override;
};
} // namespace

static bool is1x1FilterConv2dOp(linalg::LinalgOp convOp) {
  const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
  const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
  if (!isNCHW & !isNHWC)
    return false;

  auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
      convOp.getDpsInputOperand(1)->get().getType());
  if (!filterShapeType)
    return false;

  // Adjusting dimension indices based on Conv2DOpType.
  const int khIndex = isNHWC ? 0 : 2;
  const int kwIndex = isNHWC ? 1 : 3;
  auto filterShape = filterShapeType.getShape();
  return filterShape[khIndex] == 1 && filterShape[kwIndex] == 1;
}

void GeneralizeLinalgNamedOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::LinalgOp> namedOpCandidates;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
      return;
    }
    if (isa_and_nonnull<linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp,
                        linalg::CeilOp, linalg::CopyOp, linalg::DivOp,
                        linalg::DivUnsignedOp, linalg::ElemwiseBinaryOp,
                        linalg::ElemwiseUnaryOp, linalg::ExpOp, linalg::FloorOp,
                        linalg::LogOp, linalg::MapOp, linalg::MaxOp,
                        linalg::MulOp, linalg::NegFOp, linalg::ReduceOp,
                        linalg::SubOp, linalg::TransposeOp>(
            linalgOp.getOperation()) ||
        is1x1FilterConv2dOp(linalgOp)) {
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
