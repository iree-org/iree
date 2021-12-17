// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

void registerConstExprDependentDialects(DialectRegistry &registry) {
  registry.insert<IREE::Util::UtilDialect>();
  registry.insert<linalg::LinalgDialect>();
}

bool isEligibleConstExprOp(Operation *op) {
  // Special carve-out for unregistered testing ops.
  if (!op->isRegistered()) {
    if (op->getName().getStringRef() ==
        "iree_unregistered.non_leaf_const_expr") {
      return true;
    }
    if (op->getName().getStringRef() == "iree_unregistered.const_expr") {
      return true;
    }
    if (op->getName().getStringRef() == "iree_unregistered.var_expr") {
      return false;
    }
    return false;
  }

  // Allow linalg ops, even though they are not effect annotated.
  if (op->getDialect() ==
      op->getContext()->getOrLoadDialect<linalg::LinalgDialect>()) {
    return true;
  }

  // By default any effects make it non const-expr.
  if (!MemoryEffectOpInterface::hasNoEffect(op)) {
    return false;
  }

  // By default, ops without results are not const-expr.
  if (op->getNumResults() == 0) {
    return false;
  }

  return true;
}

bool isHoistableConstExprLeaf(const ConstExprAnalysis::ConstValueInfo *info) {
  if (!info->getOperation()->isRegistered()) {
    if (info->getOperation()->getName().getStringRef() ==
        "iree_unregistered.non_leaf_const_expr") {
      return false;
    }
  }

  // Generally, we prefer to not hoist broadcasts.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(info->getOperation())) {
    // Detect op that only broadcast input as fusing them makes the new
    // op cheaper.
    if (genericOp.getNumParallelLoops() == genericOp.getNumLoops() &&
        isa<linalg::YieldOp>(genericOp.getBody()->front())) {
      for (OpOperand *opOperand : genericOp.getInputOperands()) {
        AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
        if (indexingMap.isProjectedPermutation() &&
            indexingMap.getNumDims() != indexingMap.getNumResults()) {
          return false;
        }
      }
    }
  }

  return true;
}

bool isHoistableConstExprConsumingOperand(OpOperand *operand) {
  Operation *op = operand->getOwner();
  // For linalg ops, we only want to hoist inputs.
  if (auto structuredOp = dyn_cast<linalg::LinalgOp>(op)) {
    return operand->getOperandNumber() < structuredOp.getNumInputs();
  }

  // Fallback to yes.
  return true;
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
