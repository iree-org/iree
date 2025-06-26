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
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_HOISTUNIFORMSCALARCOMPUTEPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

static bool isUniformScalarForDispatch(Operation *op, Operation *dispatch) {
  assert(op->getParentOp() == dispatch &&
         "hoist target is not direct child of dispatch");
  if (!mlir::isPure(op)) {
    return false;
  }

  auto isScalarOrVector = [](Type t) {
    return t.isIntOrIndexOrFloat() || isa<VectorType>(t);
  };
  auto isOutsideDispatch = [&](Value v) {
    return v.getParentRegion()->getParentOp() != dispatch;
  };

  // Check that all operands are defined outside the dispatch and all operand
  // and result types are scalars or vectors.
  return llvm::all_of(op->getOperands(), isOutsideDispatch) &&
         llvm::all_of(op->getOperandTypes(), isScalarOrVector) &&
         llvm::all_of(op->getResultTypes(), isScalarOrVector);
}

namespace {

struct HoistUniformScalarComputePass
    : public DispatchCreation::impl::HoistUniformScalarComputePassBase<
          HoistUniformScalarComputePass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    funcOp.walk([&](IREE::Flow::DispatchRegionOp dispatch) {
      for (Block &body : dispatch.getBody()) {
        SmallVector<Operation *> ops = llvm::map_to_vector(
            body.getOperations(), [](Operation &op) { return &op; });
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
            if (!constant || (!v.getType().isIntOrIndexOrFloat() &&
                              !isa<VectorType>(v.getType()))) {
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
      return WalkResult::interrupt();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
