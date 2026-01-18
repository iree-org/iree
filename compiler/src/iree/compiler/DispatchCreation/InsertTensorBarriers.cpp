// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_INSERTTENSORBARRIERSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Check if an operation is a compute operation
/// (linalg/linalgext/tensor/tensor_ext).
static bool isComputeOp(Operation *op) {
  if (!op) {
    return false;
  }
  auto *dialect = op->getDialect();
  if (!dialect) {
    return false;
  }
  return isa<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect,
             tensor::TensorDialect, IREE::TensorExt::IREETensorExtDialect>(
      dialect);
}

// Traverse forward along use-def chains starting from `val` to identify values
// that flow into compute operations. These values are candidates for inserting
// compute_barrier.start operations.
static void collectInputsToComputeRegion(Value val,
                                         llvm::SetVector<Value> &inputValues,
                                         llvm::DenseSet<Value> &visited) {
  if (!visited.insert(val).second) {
    return;
  }

  for (OpOperand &use : val.getUses()) {
    Operation *userOp = use.getOwner();
    if (!userOp) {
      continue;
    }
    if (isComputeOp(userOp)) {
      inputValues.insert(val);
    } else {
      for (Value result : userOp->getResults()) {
        collectInputsToComputeRegion(result, inputValues, visited);
      }
    }
  }
}

// Traverse backward along use-def chains starting from `val` to identify values
// produced by compute operations. These values are candidates for inserting
// compute_barrier.end operations.
static void
collectOutputsFromComputeRegion(Value val, llvm::SetVector<Value> &outputValues,
                                llvm::DenseSet<Value> &visited) {
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp || !visited.insert(val).second) {
    return;
  }

  if (isComputeOp(definingOp)) {
    outputValues.insert(val);
  } else {
    llvm::for_each(definingOp->getOperands(), [&](Value operand) {
      collectOutputsFromComputeRegion(operand, outputValues, visited);
    });
  }
}

struct InsertTensorBarriersPass final
    : public impl::InsertTensorBarriersPassBase<InsertTensorBarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Insert compute_barrier.start operations for values that flow into compute
    // ops.
    llvm::SetVector<Value> needsStartBarrier;
    llvm::DenseSet<Value> visited;
    llvm::for_each(funcOp.getArguments(), [&](BlockArgument arg) {
      collectInputsToComputeRegion(arg, needsStartBarrier, visited);
    });

    for (Value val : needsStartBarrier) {
      if (!isa<RankedTensorType>(val.getType())) {
        continue;
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfterValue(val);
      auto startOp = IREE::TensorExt::ComputeBarrierStartOp::create(
          builder, val.getLoc(), val);
      val.replaceUsesWithIf(startOp.getResult(), [&](OpOperand &use) {
        return use.getOwner() != startOp && isComputeOp(use.getOwner()) &&
               !isa<tensor::DimOp>(use.getOwner());
      });
    }

    // Insert compute_barrier.end operations for values that flow out of compute
    // ops.
    llvm::SetVector<Value> needsEndBarrier;
    visited.clear();
    funcOp.walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::ReturnLike>()) {
        return;
      }
      for (Value operand : op->getOperands()) {
        collectOutputsFromComputeRegion(operand, needsEndBarrier, visited);
      }
    });

    for (Value val : needsEndBarrier) {
      if (!isa<RankedTensorType>(val.getType())) {
        continue;
      }
      Operation *definingOp = val.getDefiningOp();
      if (!definingOp) {
        continue;
      }
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(definingOp);
      auto endOp = IREE::TensorExt::ComputeBarrierEndOp::create(
          builder, val.getLoc(), val);
      val.replaceUsesWithIf(endOp.getResult(), [&](OpOperand &use) {
        return !isComputeOp(use.getOwner()) && use.getOwner() != endOp &&
               !isa<tensor::DimOp>(use.getOwner());
      });
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
