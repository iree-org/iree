// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::iree_compiler {

namespace {

class HoistRedundantVectorTransfersPass
    : public HoistRedundantVectorTransfersBase<
          HoistRedundantVectorTransfersPass> {
public:
  using HoistRedundantVectorTransfersBase::HoistRedundantVectorTransfersBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void HoistRedundantVectorTransfersPass::runOnOperation() {
  auto funcOp = getOperation();
  linalg::hoistRedundantVectorTransfers(funcOp);
  IRRewriter rewriter(funcOp->getContext());
  // Hoist redundant vector transfers on tensors.
  // TODO: walking in some reverse / inside-out order would be more efficient
  // and would capture more cases.
  funcOp.walk(
      [&](scf::ForOp forOp) { hoistLoopInvariantSubsets(rewriter, forOp); });
  vector::transferOpflowOpt(rewriter, funcOp);
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistRedundantVectorTransfersPass() {
  return std::make_unique<HoistRedundantVectorTransfersPass>();
}

} // namespace mlir::iree_compiler
