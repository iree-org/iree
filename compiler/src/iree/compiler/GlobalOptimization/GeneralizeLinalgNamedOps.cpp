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
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler::GlobalOptimization {

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

/// Returns true if `linalgOp` can be simplified to a basic GEMM.
static bool isConvFoldableToContraction(linalg::LinalgOp linalgOp) {
  auto convDimsOrFailure = linalg::inferConvolutionDims(linalgOp);
  if (failed(convDimsOrFailure)) {
    return false;
  }
  auto &convDims = *convDimsOrFailure;

  if (!llvm::all_of(convDims.strides,
                    [](int64_t element) { return element == 1; })) {
    LDBG("conv not foldable: non-unit strides");
    return false;
  }

  // Dont generalize pooling operations or depthwise convolutions. For pooling
  // ops, the input/output channel size will be categorized as the additional
  // batch dimension.
  if (convDims.outputChannel.empty() || convDims.inputChannel.empty()) {
    LDBG("conv not foldable: missing input or output channel dims");
    return false;
  }

  // Check if all filter dimensions are size 1.
  const int64_t kFilterInputIdx = 1;
  auto filterShapeType = llvm::dyn_cast<RankedTensorType>(
      linalgOp.getDpsInputOperand(kFilterInputIdx)->get().getType());
  if (!filterShapeType) {
    LDBG("conv not foldable: filter shape not ranked tensor");
    return false;
  }
  auto filterShape = filterShapeType.getShape();
  AffineMap filterMap = linalgOp.getIndexingMapsArray()[kFilterInputIdx];
  for (auto filterLoop : convDims.filterLoop) {
    std::optional<int64_t> maybeDim = filterMap.getResultPosition(
        getAffineDimExpr(filterLoop, filterMap.getContext()));
    if (!maybeDim || filterShape[*maybeDim] != 1) {
      LDBG("conv not foldable: non-unit filter dim");
      return false;
    }
  }

  return true;
}

void GeneralizeLinalgNamedOpsPass::runOnOperation() {
  auto funcOp = getOperation();
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
    if (isa_and_nonnull<linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp,
                        linalg::CeilOp, linalg::CopyOp, linalg::DivOp,
                        linalg::DivUnsignedOp, linalg::ExpOp, linalg::FloorOp,
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
