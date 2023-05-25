// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

void populateEscapingProducers(Operation *parentOp, ConstExprOpInfo &info) {
  SmallPtrSet<Operation *, 8> containedOps;
  parentOp->walk<WalkOrder::PreOrder>([&](Operation *itOp) {
    containedOps.insert(parentOp);
    // For the outer-most op, consider that all operands escape.
    if (itOp == parentOp) {
      info.producers.insert(itOp->getOperands().begin(),
                            itOp->getOperands().end());
      return;
    }

    // For nested operations, only consider that they escape if they are
    // defined outside of the parent.
    for (Value operand : itOp->getOperands()) {
      Block *block = operand.getParentBlock();
      if (!containedOps.contains(block->getParentOp())) {
        info.producers.insert(operand);
      }
    }
  });
}

ConstExprOpInfo getInfoForDefaultConstExprOp(Operation *op) {
  ConstExprOpInfo info;
  info.isEligible = true;
  populateEscapingProducers(op, info);
  return info;
}

}  // namespace

void registerConstExprDependentDialects(DialectRegistry &registry) {
  registry.insert<IREE::Util::UtilDialect>();
  registry.insert<linalg::LinalgDialect>();
}

ConstExprOpInfo ConstExprOpInfo::getForOp(Operation *op) {
  // Special carve-out for unregistered testing ops.
  if (!op->isRegistered()) {
    // Reject.
    if (op->getName().getStringRef() == "iree_unregistered.var_expr") {
      return {};
    }
    // Accept.
    if (op->getName().getStringRef() ==
            "iree_unregistered.non_leaf_const_expr" ||
        op->getName().getStringRef() == "iree_unregistered.const_expr") {
      return getInfoForDefaultConstExprOp(op);
    }
    return {};
  }

  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  if (op->getDialect() ==
      op->getContext()->getOrLoadDialect<linalg::LinalgDialect>()) {
    // Structured op implementations and a handful of pure ops are included.
    // Notably: IndexOp is not included because it establishes a hidden
    // dependency to the iterator and is non-const.
    if (llvm::isa<linalg::LinalgOp>(op) || llvm::isa<tensor::PadOp>(op) ||
        llvm::isa<tensor::EmptyOp>(op)) {
      return getInfoForDefaultConstExprOp(op);
    }

    return {};
  }

  // By default any effects make it non const-expr.
  if (!isMemoryEffectFree(op)) {
    return {};
  }

  // By default, ops without results are not const-expr.
  if (op->getNumResults() == 0) {
    return {};
  }

  // Forbid if part of a parent that should be treated atomically.
  if (op->getParentOfType<linalg::LinalgOp>()) {
    return {};
  }

  return getInfoForDefaultConstExprOp(op);
}

bool isHoistableConstExprLeaf(const ConstExprAnalysis::ConstValueInfo *info) {
  Operation *op = info->getOperation();
  if (!op->isRegistered()) {
    if (op->getName().getStringRef() ==
        "iree_unregistered.non_leaf_const_expr") {
      return false;
    }
  }

  // Never hoist sub-byte aligned values: in legal programs, these will be
  // cast or packed in some successor.
  if (auto integerType = llvm::dyn_cast<IntegerType>(
          getElementTypeOrSelf(info->constValue.getType()))) {
    if (integerType.getWidth() % 8 != 0) {
      return false;
    }
  }

  // Generally, we prefer to not hoist broadcasts.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    // Detect op that only broadcast input as fusing them makes the new
    // op cheaper.
    if (genericOp.getNumParallelLoops() == genericOp.getNumLoops() &&
        isa<linalg::YieldOp>(genericOp.getBody()->front())) {
      for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
        AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
        if (indexingMap.isProjectedPermutation() &&
            indexingMap.getNumDims() != indexingMap.getNumResults()) {
          return false;
        }
      }
    }
  }

  // Never hoist empty. These are sometimes used for pure shape metadata
  // and must not be separated from their consumers.
  if (isa<tensor::EmptyOp>(op)) {
    return false;
  }

  return true;
}

bool isHoistableConstExprConsumingOperand(OpOperand *operand) {
  Operation *op = operand->getOwner();
  // For linalg ops, we only want to hoist inputs.
  if (auto structuredOp = dyn_cast<linalg::LinalgOp>(op)) {
    return operand->getOperandNumber() < structuredOp.getNumDpsInputs();
  }

  // Fallback to yes.
  return true;
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
