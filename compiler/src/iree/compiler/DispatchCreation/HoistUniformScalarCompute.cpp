// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_HOISTUNIFORMSCALARCOMPUTEPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

static bool isUniformScalarForDispatch(Operation *op, Operation *dispatch) {
  assert(op->getParentOp() == dispatch &&
         "hoist target is not direct child of dispatch");
  if (!mlir::isPure(op) || op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }

  auto isIntOrIndex = [](Type t) { return t.isIntOrIndex(); };
  auto isOutsideDispatch = [&](Value v) {
    return v.getParentRegion()->getParentOp() != dispatch;
  };

  // Check that all operands are defined outside the dispatch and all operand
  // and result types are scalars or vectors.
  return llvm::all_of(op->getOperands(), isOutsideDispatch) &&
         llvm::all_of(op->getOperandTypes(), isIntOrIndex) &&
         llvm::all_of(op->getResultTypes(), isIntOrIndex);
}

namespace {

struct HoistUniformScalarComputePass
    : public DispatchCreation::impl::HoistUniformScalarComputePassBase<
          HoistUniformScalarComputePass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    WalkResult walkResult =
        funcOp.walk([&](IREE::Flow::DispatchRegionOp dispatch) {
          for (Block &body : dispatch.getBody()) {
            SmallVector<Operation *> ops;
            // Restrict to arith ops to avoid unexpected hoisting of
            // flow/stream/hal.dispatch.workgroups.count/id ops.
            // TODO: Add an op trait to tie count/id ops to region ops.
            for (Operation &op : body.getOperations()) {
              if (isa<arith::ArithDialect>(op.getDialect())) {
                ops.push_back(&op);
              }
            }
            for (Operation *op : ops) {
              if (isUniformScalarForDispatch(op, dispatch)) {
                op->moveBefore(dispatch);
              }
            }
          }

          llvm::SetVector<Operation *> constantsToClone;
          mlir::visitUsedValuesDefinedAbove(
              dispatch.getBody(), dispatch.getBody(), [&](OpOperand *operand) {
                Value v = operand->get();
                auto constant = v.getDefiningOp<arith::ConstantOp>();
                if (!constant || !v.getType().isIntOrIndex()) {
                  return;
                }
                constantsToClone.insert(constant);
              });
          for (Operation *constant : constantsToClone) {
            if (constant->hasOneUse()) {
              constant->moveBefore(&dispatch.getBody().front().front());
            } else {
              if (failed(IREE::Flow::clonePrecedingOpIntoDispatchRegion(
                      rewriter, constant, dispatch))) {
                return WalkResult::interrupt();
              }
            }
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) {
      funcOp->emitError("Failed to clone constant into dispatch region.");
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
