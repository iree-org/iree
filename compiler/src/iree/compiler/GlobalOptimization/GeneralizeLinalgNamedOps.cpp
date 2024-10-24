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

/// Returns true of `linalgOp` is a Conv2DNchwFchwOp or Conv2DNhwcHwcfOp with
/// all strides equal to 1 and with a kernel height and width of 1
static bool isConvFoldableToContraction(linalg::LinalgOp linalgOp) {
  auto NCHWOp = dyn_cast<linalg::Conv2DNchwFchwOp>(linalgOp.getOperation());
  auto NHWCOp = dyn_cast<linalg::Conv2DNhwcHwcfOp>(linalgOp.getOperation());

  if (!NCHWOp && !NHWCOp)
    return false;

  DenseIntElementsAttr strides =
      NCHWOp ? NCHWOp.getStrides() : NHWCOp.getStrides();
  if (!llvm::all_of(
          strides, [](APInt element) { return element.getSExtValue() == 1; })) {
    return false;
  }

  auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
      linalgOp.getDpsInputOperand(1)->get().getType());
  if (!filterShapeType)
    return false;

  // Adjusting dimension indices based on Conv2DOpType.
  const int khIndex = NHWCOp ? 0 : 2;
  const int kwIndex = NHWCOp ? 1 : 3;
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
        isConvFoldableToContraction(linalgOp)) {
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
