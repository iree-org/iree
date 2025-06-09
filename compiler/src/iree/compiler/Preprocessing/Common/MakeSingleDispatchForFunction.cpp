// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_MAKESINGLEDISPATCHFORFUNCTIONPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

struct MakeSingleDispatchForFunctionPass
    : public iree_compiler::Preprocessing::impl::
          MakeSingleDispatchForFunctionPassBase<
              MakeSingleDispatchForFunctionPass> {
  void runOnOperation() override;
};
} // namespace

void MakeSingleDispatchForFunctionPass::runOnOperation() {
  auto funcOp = getOperation();

  Region &body = funcOp.getFunctionBody();
  if (!llvm::hasSingleElement(body)) {
    // Do nothing.
    return;
  }

  Block &block = body.front();
  auto whitelistedOps = [&](Operation *op) {
    auto dialect = op->getDialect();
    if (isa<IREE::LinalgExt::IREELinalgExtDialect, linalg::LinalgDialect,
            tensor::TensorDialect>(dialect)) {
      return true;
    }
    if (isa<arith::ArithDialect>(dialect)) {
      return !isa<arith::ConstantOp>(op);
    }
    return false;
  };

  // Find the region to outline by doing two slices.

  // 1. The first slice removes any ABI related operations at the return.
  BackwardSliceOptions firstSliceOptions;
  firstSliceOptions.omitUsesFromAbove = false;
  firstSliceOptions.inclusive = true;
  // Filter returns true for any dialect not allowed.
  firstSliceOptions.filter = [&](Operation *op) { return !whitelistedOps(op); };
  llvm::SetVector<Operation *> firstSlice;
  [[maybe_unused]] LogicalResult ret =
      getBackwardSlice(block.getTerminator(), &firstSlice, firstSliceOptions);
  assert(ret.succeeded());

  // 2. Do the second slice starting from the first slice to remove any ABI
  // related operations on the argument.
  BackwardSliceOptions secondSliceOptions;
  secondSliceOptions.omitUsesFromAbove = false;
  secondSliceOptions.inclusive = true;
  secondSliceOptions.filter = whitelistedOps;
  llvm::SetVector<Operation *> secondSlice;
  for (Operation *op : firstSlice) {
    for (Value operand : op->getOperands()) {
      ret = getBackwardSlice(operand, &secondSlice, secondSliceOptions);
      assert(ret.succeeded());
    }
  }
  if (secondSlice.empty()) {
    return;
  }

  // 3. Sort  the operations.
  mlir::topologicalSort(secondSlice);

  // Move the second slice into a `flow.dispatch.region`.
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  Operation *insertionPoint = secondSlice.front();
  rewriter.setInsertionPoint(insertionPoint);

  // Go over the slice and find the return values, i.e. values defined by ops in
  // the slice and used outside of the slice.
  SetVector<Value> results;
  for (Operation *op : secondSlice) {
    for (OpOperand &use : op->getUses()) {
      Operation *user = use.getOwner();
      if (!secondSlice.contains(user)) {
        results.insert(use.get());
      }
    }
  }

  // Getting the dynamic dimensions of result values in terms of values passed
  // into the slice should be possible, but is involved. Ignore this case for
  // now.
  for (auto result : results) {
    auto shapedType = dyn_cast<ShapedType>(result.getType());
    if (!shapedType.hasStaticShape()) {
      emitError(result.getLoc())
          << "unhandled dynamic dimensions for created dispatch region";
      return signalPassFailure();
    }
  }

  // Find the values captures by the slice.
  SetVector<Value> capturedValues;
  for (Operation *op : secondSlice) {
    for (OpOperand &use : op->getOpOperands()) {
      Operation *definingOp = use.get().getDefiningOp();
      if (!definingOp || secondSlice.contains(definingOp)) {
        continue;
      }
      capturedValues.insert(use.get());
    }
  }
  if (failed(moveValueDefinitions(rewriter, capturedValues.getArrayRef(),
                                  insertionPoint))) {
    funcOp.emitOpError(
        "failed to move definitions of captured values before region op");
    return signalPassFailure();
  }

  auto resultTypes =
      llvm::map_to_vector(results, [](Value v) -> Type { return v.getType(); });
  auto dispatchRegionOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      funcOp.getLoc(), resultTypes,
      /*result_dims=*/ValueRange{}, /*workload=*/ValueRange{});
  Region &regionOpBody = dispatchRegionOp.getBody();
  Block *newBlock = rewriter.createBlock(&regionOpBody, regionOpBody.begin());
  for (Operation *op : secondSlice) {
    rewriter.moveOpBefore(op, newBlock, newBlock->end());
  }
  rewriter.setInsertionPointToEnd(newBlock);
  rewriter.create<IREE::Flow::ReturnOp>(dispatchRegionOp.getLoc(),
                                        results.getArrayRef());
  rewriter.replaceUsesWithIf(
      results.getArrayRef(), dispatchRegionOp->getResults(),
      [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user->getParentOfType<IREE::Flow::DispatchRegionOp>() !=
               dispatchRegionOp;
      });
}

} // namespace mlir::iree_compiler::Preprocessing
