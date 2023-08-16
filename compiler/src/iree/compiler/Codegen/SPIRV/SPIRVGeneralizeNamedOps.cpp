// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVGeneralizeNamedOps.cpp - Pass to generalize ---------===//
//===- linalg.matmul_transpose_b LinalgOps with unit dimensions -----------===//
//
// The pass is to generalize linalg.matmul_transpose_b ops that are equivalent
// to a transposed matvec op
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct SPIRVGeneralizeNamedOpsPass
    : public SPIRVGeneralizeNamedOpsBase<
          SPIRVGeneralizeNamedOpsPass> {

  void runOnOperation() override;
};
} // namespace

// Check if any of the input dimensions are static 1
static bool hasUnitDims(linalg::MatmulTransposeBOp linalgOp) {
  auto initDims =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType())
          .getShape();
  return (llvm::any_of(initDims, [](auto dim) {
    return !ShapedType::isDynamic(dim) && dim == 1;
  }));
}

void SPIRVGeneralizeNamedOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::MatmulTransposeBOp> namedOpCandidates;
  funcOp.walk([&](linalg::MatmulTransposeBOp linalgOp) {
    if (isa_and_nonnull<linalg::MatmulTransposeBOp>(linalgOp.getOperation())) {
      if (hasUnitDims(linalgOp))
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

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVGeneralizeNamedOpsPass() {
  return std::make_unique<SPIRVGeneralizeNamedOpsPass>();
}

} // namespace iree_compiler
} // namespace mlir
