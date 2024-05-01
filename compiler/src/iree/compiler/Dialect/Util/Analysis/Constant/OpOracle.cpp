// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler::IREE::Util {

void registerConstExprDependentDialects(DialectRegistry &registry) {
  registry.insert<IREE::Util::UtilDialect>();
}

static void populateEscapingProducers(Operation *parentOp,
                                      ConstExprOpInfo &info) {
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

// Enforce a limited allow-list of types that are legal to consider
// constexpr operand or result types. Given MLIR's open type system,
// it is best to be conservative here, and we limit to known value
// types.
static bool isLegalConstExprType(Type t) {
  // If implementing the hoistable interface just return what the interface
  // says.
  if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(t)) {
    return hoistableType.isHoistableType();
  } else if (t.isIntOrIndexOrFloat()) {
    return true;
  } else if (auto tensorType = dyn_cast<TensorType>(t)) {
    return isLegalConstExprType(tensorType.getElementType());
  }
  return false;
}

bool isLegalConstExprRootType(Type t) { return isLegalConstExprType(t); }

// Check if the op can be an eligible const expr.
static bool isEligibleConstExpr(Operation *op) {
  // Optimization barriers cannot be folded.
  if (isa<IREE::Util::OptimizationBarrierOp>(op)) {
    return false;
  }

  // By default, ops without results are not const-expr.
  if (op->getNumResults() == 0) {
    return false;
  }

  // Forbid if illegal result types. It is sufficient to verify result
  // types since all constexpr values must come from a result somewhere
  // in the analyzed tree.
  if (!llvm::all_of(op->getResultTypes(), isLegalConstExprType)) {
    return false;
  }

  // If implementing the HoistableOpInterface, just use the decision made by
  // the interface.
  if (auto hoistableOp = dyn_cast<IREE::Util::HoistableOpInterface>(op)) {
    if (hoistableOp.isHoistableOp()) {
      return true;
    }
    return false;
  }

  // Forbid if part of a parent that should be treated atomically.
  Operation *parent = op;
  while (auto hoistableParent =
             parent->getParentOfType<IREE::Util::HoistableOpInterface>()) {
    if (hoistableParent.isAtomicallyHoistableOp())
      return false;
    parent = hoistableParent;
  }

  // Special carve-out for unregistered testing ops.
  // Since these are unregistered, we have to make this carve-out
  // after any verification on structure but before any verification
  // of traits (like effects). These specific unregistered ops
  // override default traits.
  if (!op->isRegistered()) {
    // Reject.
    if (op->getName().getStringRef() == "iree_unregistered.var_expr") {
      return false;
    }
    // Accept.
    if (op->getName().getStringRef() ==
            "iree_unregistered.non_leaf_const_expr" ||
        op->getName().getStringRef() == "iree_unregistered.const_expr") {
      return true;
    }
  }

  // By default any effects make it non const-expr.
  if (!isMemoryEffectFree(op)) {
    return false;
  }

  return true;
}

ConstExprOpInfo ConstExprOpInfo::getForOp(Operation *op) {
  ConstExprOpInfo info;
  info.isEligible = isEligibleConstExpr(op);

  // Populate the producers for both eligible and ineligible cases, as we need
  // the producers of ineligible op to identify hoistable constant producers.
  populateEscapingProducers(op, info);

  return info;
}

bool isHoistableConstExprLeaf(const ConstExprAnalysis::ConstValueInfo *info) {
  Operation *op = info->getOperation();
  if (!op->isRegistered()) {
    if (op->getName().getStringRef() ==
        "iree_unregistered.non_leaf_const_expr") {
      return false;
    }
  }

  // First check whether we should hoist this kind of operation. Type local
  // decisions should always come last.

  // If implementing the HoistableOpInterface, check whether the op is legal to
  // hoist. We still need to check for type legality afterwards though.
  if (auto hoistableOp = dyn_cast<IREE::Util::HoistableOpInterface>(op)) {
    if (!hoistableOp.isHoistableLeafOp())
      return false;
  }

  // If implementing the HoistableTypeInterface, at this point we can just
  // return what the interface says.
  if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(
          info->constValue.getType())) {
    return hoistableType.isHoistableLeafType();
  }

  // Never hoist sub-byte aligned values: in legal programs, these will be
  // cast or packed in some successor.
  if (auto integerType = dyn_cast<IntegerType>(
          getElementTypeOrSelf(info->constValue.getType()))) {
    if (integerType.getWidth() % 8 != 0) {
      return false;
    }
  }

  return true;
}

bool isHoistableConstExprConsumingOperand(OpOperand *operand) {
  Operation *op = operand->getOwner();
  if (auto hoistableOp = dyn_cast<IREE::Util::HoistableOpInterface>(op)) {
    return hoistableOp.isOperandHoistable(operand);
  }

  // Fallback to yes.
  return true;
}

} // namespace mlir::iree_compiler::IREE::Util
