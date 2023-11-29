// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTTransferFunctions.h"

#define DEBUG_TYPE "iree-simt-transfer-functions"

using namespace llvm;
using namespace mlir;
using namespace mlir::iree_compiler;

/// Get OpOperand from an operation and the lattice index, which is basically
/// the x^th operand of vector type.
static OpOperand &getOpOperand(Operation *op, unsigned operandLatticeIndex) {
  unsigned operandIndex = 0;
  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get().getType().isa<VectorType>()) {
      if (operandIndex == operandLatticeIndex) {
        return operand;
      }
      operandIndex++;
    }
  }
  llvm_unreachable("No vector operand found");
}

/// Get a layout if all the given layouts are same. If all layouts are not same,
/// return nullptr.
static DistributionLayout *
getAgreedLayout(ArrayRef<DistributionLayout *> layouts) {
  if (layouts.size() == 0)
    return nullptr;

  // Check if all layouts are same.
  for (unsigned i = 1, e = layouts.size(); i < e; ++i) {
    if (*layouts[i] != *layouts[0]) {
      return nullptr;
    }
  }

  return layouts[0];
}

/// Get a layout if all the given layouts are same. If all layouts are not same,
/// return nullptr.
static const DistributionLayout *
getAgreedLayout(ArrayRef<const DistributionLayout *> layouts) {
  if (layouts.size() == 0)
    return nullptr;

  // Check if all layouts are same.
  for (unsigned i = 1, e = layouts.size(); i < e; ++i) {
    if (*layouts[i] != *layouts[0]) {
      return nullptr;
    }
  }

  return layouts[0];
}

/// Given a list of layouts, enforce a single layout for all of them.
/// The layout chosen is a heuristic that choses the first enforced layout.
/// TODO: Use the most common layout to minimize the number of conflicts.
static void enforceSameLayoutForOperands(
    Operation *op, ArrayRef<DistributionLayout *> operands,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Get any enforced layout.
  DistributionLayout *chosenOperandLayout = nullptr;
  for (DistributionLayout *lattice : operands) {
    if (lattice->hasLayout()) {
      chosenOperandLayout = lattice;
      break;
    }
  }

  // Enforce the layout to other operands.
  if (chosenOperandLayout) {
    // Note that the operand lattice is not updated. So using the operand
    // lattice again can cause bugs.
    for (auto [index, lattice] : llvm::enumerate(operands)) {
      OpOperand &opOperand = getOpOperand(op, index);
      ChangeResult changed =
          lattice->resolveWithPossibleConflict(chosenOperandLayout, opOperand);
      update(lattice, changed);
    }
  }
}

/// =========================
///        PROPAGATION
/// =========================

static void propagateLayoutToElementwiseOp(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  DistributionLayout *result = resultLattices[0];

  // If result lattice already has a layout, we cannot do
  // anything. We do not impose layout conflicts on results.
  // TODO: Explore if this is actually needed.
  if (result->hasLayout()) {
    return;
  }

  // Check if all vector operands agree on the same layout.
  const DistributionLayout *chosenOperandLayout =
      getAgreedLayout(operandLattices);
  if (chosenOperandLayout == nullptr) {
    return;
  }

  ChangeResult changed = result->resolve(chosenOperandLayout);
  update(result, changed);
}

static void propagateLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Multi reduce has only one vector result.
  DistributionLayout *result = resultLattices[0];
  // Multi reduce has first vector operands as the value being reduced.
  const DistributionLayout *vector = operandLattices[0];
  // Multi reduce has second operand as init.
  const DistributionLayout *init = operandLattices[1];

  // If result lattice already has a layout, we cannot do anything. We do not
  // impose layout conflicts on results.
  if (result->hasLayout()) {
    return;
  }

  // If the vector begin reduced has a layout, then propagate it to the result.
  // by projecting
  if (vector->hasLayout()) {
    SmallVector<bool> reductionMask = multiReduce.getReductionMask();
    ChangeResult changed =
        result->resolve(vector->getInnerLayout().project(reductionMask));
    update(result, changed);
    return;
  }

  ChangeResult changed = result->resolve(init);
  update(result, changed);
}

static void propagateLayoutToTransposeOp(
    vector::TransposeOp transpose,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Transpose has only one vector result.
  DistributionLayout *result = resultLattices[0];
  // Transpose has only one vector operand.
  const DistributionLayout *value = operandLattices[0];

  // If result lattice already has a layout, we cannot do anything. We do not
  // impose layout conflicts on results.
  if (result->hasLayout()) {
    return;
  }

  // Cannot propagate layout if value is uninitialized.
  if (value->isUninitialized()) {
    return;
  }

  // Build a transposed layout.
  SmallVector<unsigned> permutation;
  ArrayRef<int64_t> perm = transpose.getPermutation();
  VectorLayoutInterface permutedLayout = result->getInnerLayout().permute(perm);

  // Try to resolve with the transposed layout.
  ChangeResult changed = result->resolve(permutedLayout);
  update(result, changed);
}

void iree_compiler::propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    propagateLayoutToElementwiseOp(op, operandLattices, resultLattices, update);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    propagateLayoutToMultiReductionOp(multiReduce, operandLattices,
                                      resultLattices, update);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    propagateLayoutToTransposeOp(transpose, operandLattices, resultLattices,
                                 update);
    return;
  }

  return;
}

/// =========================
///        ENFORCEMENT
/// =========================

static void enforceLayoutToElementwiseOp(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  // Try to enforce the layout of the result on operands.
  const DistributionLayout *result = resultLattices[0];
  if (result->hasLayout()) {
    // Note that the operand lattice is not updated. So using the operand
    // lattice again can cause bugs.
    for (auto [index, operandLattice] : llvm::enumerate(operandLattices)) {
      ChangeResult changed = operandLattice->resolveWithPossibleConflict(
          result, getOpOperand(op, index));
      update(operandLattice, changed);
    }
  } else {
    // Enforce the same layout on all operands.
    enforceSameLayoutForOperands(op, operandLattices, update);
  }
}

static void enforceLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Reductions should always propagate value layout to result. Result can
  // enforce it's layout on init.
  const DistributionLayout *result = resultLattices[0];
  DistributionLayout *init = operandLattices[1];

  // Enforce the result layout on init.
  ChangeResult changedDueToResult =
      init->resolveWithPossibleConflict(result, getOpOperand(multiReduce, 1));
  update(init, changedDueToResult);
}

static void enforceLayoutToTransposeOp(
    vector::TransposeOp transpose,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Transpose has only one vector result.
  const DistributionLayout *result = resultLattices[0];
  // Transpose has only one vector operand.
  DistributionLayout *value = operandLattices[0];

  // Cannot enforce layout if result is uninitialized.
  if (result->isUninitialized()) {
    return;
  }

  // Build a transposed layout.
  SmallVector<unsigned> permutation;
  ArrayRef<int64_t> perm = transpose.getPermutation();
  VectorLayoutInterface permutedLayout = result->getInnerLayout().permute(perm);

  // Try to resolve with the transposed layout.
  ChangeResult changed = value->resolveWithPossibleConflict(
      permutedLayout, getOpOperand(transpose, 0));
  update(value, changed);
}

static void enforceLayoutToBroadcastOp(
    vector::BroadcastOp broadcast,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Broadcast has only one vector result.
  const DistributionLayout *result = resultLattices[0];
  // Broadcast has only one vector operand.
  DistributionLayout *value = operandLattices[0];

  // Cannot enforce layout if result is uninitialized.
  if (result->isUninitialized()) {
    return;
  }

  // Build broadcasted layout, essentially a reduced layout along the trailing
  // dimensions.

  // Ensure that there are no broadcasted unit dims as we do not know how to
  // handle them as of now.
  assert(broadcast.computeBroadcastedUnitDims().size() == 0 &&
         "Streching in broadcasting not implemented yet.");
  // The starting k dimensions of the result are the ones that need to be
  // projected out.

  auto resultShape = broadcast.getResultVectorType().getShape();
  auto inputType = broadcast.getSourceType();
  assert(inputType.isa<VectorType>() &&
         "Scalar broadcast not supported for now.");
  auto inputShape = inputType.cast<VectorType>().getShape();

  SmallVector<bool> reductionMask(resultShape.size(), false);
  // Set the trailing dimensions to be reduced.
  int64_t resultDiff = resultShape.size() - inputShape.size();
  assert(resultDiff >= 0 && "Result shape cannot be smaller than input shape");
  for (int64_t i = 0; i < resultDiff; ++i) {
    reductionMask[i] = true;
  }

  VectorLayoutInterface resultLayout =
      result->getInnerLayout().project(reductionMask);
  ChangeResult changed = value->resolveWithPossibleConflict(
      resultLayout, getOpOperand(broadcast, 0));
  update(value, changed);
}

void iree_compiler::enforcementTransferFunction(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    enforceLayoutToElementwiseOp(op, operandLattices, resultLattices, update);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    enforceLayoutToMultiReductionOp(multiReduce, operandLattices,
                                    resultLattices, update);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    enforceLayoutToTransposeOp(transpose, operandLattices, resultLattices,
                               update);
    return;
  }

  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
    enforceLayoutToBroadcastOp(broadcast, operandLattices, resultLattices,
                               update);
    return;
  }
}
