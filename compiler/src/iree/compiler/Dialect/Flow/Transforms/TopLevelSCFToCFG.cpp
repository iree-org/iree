// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct TopLevelSCFToCFGPass
    : public TopLevelSCFToCFGBase<TopLevelSCFToCFGPass> {
  void runOnOperation() override;
};

} // namespace

void TopLevelSCFToCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSCFToControlFlowConversionPatterns(patterns);
  // Configure conversion to lower out scf.for, scf.if, scf.parallel and
  // scf.while. Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  // For nested, opaque ops that we support, mark them recursively legal.
  // Otherwise, SCF within them will be processed by this pass.
  // It would be nice to be able to set this for the whole dialect, but
  // upstream does not support that yet.
  target.addLegalOp<linalg::GenericOp>();
  target.markOpRecursivelyLegal<linalg::GenericOp>();

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTopLevelSCFToCFGPass() {
  return std::make_unique<TopLevelSCFToCFGPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
