// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::Preprocessing {

namespace {

struct MakeSingleDispatchForFunctionPass
    : public MakeSingleDispatchForFunctionBase<
          MakeSingleDispatchForFunctionPass> {
  void runOnOperation() override;
};
} // namespace

void MakeSingleDispatchForFunctionPass::runOnOperation() {
  auto funcOp = getOperation();

  // Abort if there are any operations that prevent moving all operations
  // into a single dispatch.
  auto walkResult = funcOp.walk([](mlir::CallOpInterface op) -> WalkResult {
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted()) {
    funcOp->emitOpError("unhandled operation in function body prevents moving "
                        "body into a single dispatch");
  }

  // Currently this can only be done for static shapes cause
  // there is no way of getting the tied dynamic shapes for
  // a function.
  auto resultTypes = funcOp.getResultTypes();
  if (llvm::any_of(resultTypes, [&](Type t) {
        auto shapedType = t.dyn_cast<ShapedType>();
        return shapedType && !shapedType.hasStaticShape();
      })) {
    return;
  }

  IRRewriter rewriter(&getContext());
  Location loc = funcOp.getLoc();
  Region &funcBody = funcOp.getFunctionBody();

  // Split the function entry block to create a new entry block into which the
  // new operations will be added.
  Block &entryBlock = funcBody.front();
  Block *funcBodyStart = rewriter.splitBlock(&entryBlock, entryBlock.begin());

  // Create an empty `flow.dispatch.region` operation with same result type as
  // the function.
  rewriter.setInsertionPointToEnd(&entryBlock);
  auto dispatchRegionOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      loc, resultTypes, /*result_dims=*/ValueRange{},
      /*workload=*/ValueRange{});

  // Move the body of the function into the region.
  Region &region = dispatchRegionOp.getBody();
  region.getBlocks().splice(region.begin(), funcBody.getBlocks(),
                            Region::iterator(funcBodyStart), funcBody.end());

  // Replace all `func.return` with `flow.return`.
  SmallVector<func::ReturnOp> returnOps =
      llvm::to_vector(region.getOps<func::ReturnOp>());
  for (auto returnOp : returnOps) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(returnOp);
    rewriter.replaceOpWithNewOp<IREE::Flow::ReturnOp>(returnOp,
                                                      returnOp.getOperands());
  }

  // Return the results of the `flow.dispatch.region`.
  rewriter.setInsertionPointAfter(dispatchRegionOp);
  rewriter.create<func::ReturnOp>(loc, dispatchRegionOp.getResults());
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createMakeSingleDispatchForFunctionPass() {
  return std::make_unique<MakeSingleDispatchForFunctionPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
