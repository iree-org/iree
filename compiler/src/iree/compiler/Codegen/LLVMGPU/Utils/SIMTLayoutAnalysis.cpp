// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/SIMTLayoutAnalysis.h"
#include "SIMTLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

#define DEBUG_TYPE "iree-simt-layout-analysis"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::VectorExt;

/// ==========================================================================
///        DistributionLayout
/// ==========================================================================

DistributionLayout::ResolutionResult
DistributionLayout::doResolution(const VectorLayoutInterface &rhs) {
  VectorLayoutInterface &lhs = vectorLayout;

  // Ignore if the layout to resolve with is empty.
  if (!rhs) {
    return ResolutionResult::NoChange;
  }

  // If both layouts are same, do nothing.
  if (lhs == rhs) {
    return ResolutionResult::NoChange;
  }

  // Take the other layout if the current layout is empty.
  if (!lhs && rhs) {
    setInnerLayout(rhs);
    return ResolutionResult::Change;
  }

  // Layouts conflict and need to be resolved.
  return ResolutionResult::Conflict;
}

ChangeResult DistributionLayout::resolveWithPossibleConflict(
    const VectorLayoutInterface &rhs, OpOperand &opOperand) {
  ResolutionResult result = doResolution(rhs);

  // If there is no conflict, simply return.
  if (result == ResolutionResult::NoChange) {
    return ChangeResult::NoChange;
  } else if (result == ResolutionResult::Change) {
    return ChangeResult::Change;
  }

  // Resolve conflict by create an operation that takes the input the conflicted
  // value and returns the resolved value.
  OpBuilder builder(opOperand.getOwner());
  Value input = opOperand.get();
  // Create a resolution operation. This conflict should be handeled later by
  // someone else, not this analysis.
  Operation *resolveOp =
      builder.create<IREE ::VectorExt::LayoutConflictResolutionOp>(
          input.getLoc(), input.getType(), input, vectorLayout, rhs);
  Value resolvedValue = resolveOp->getResult(0);
  opOperand.set(resolvedValue);

  // Create a new value for the resolved value and subscribe it to propagation
  // and enforcement.
  // We possibly don't need to subscribe this since this value has already
  // reached the top of the lattice and shouldn't do anything else. But it's
  // nicer to do it to have consistency.
  DistributionLayout *resolvedLayout =
      propagation->getLatticeElement(resolvedValue);
  resolvedLayout->subscribeEnforcement(enforcement);

  // We can now resolve this resolved value to the required layout.
  resolvedLayout->resolve(rhs);

  // No change actually needs to be propagated after a conflict resolution.
  // TODO: Ideally, there should be another state in the lattice which says
  // "Fixed", which would say that there is no way you can change this layout
  // anymore, and it should be override any other layout used.
  return ChangeResult::NoChange;
}

ChangeResult
DistributionLayout::resolveWithPossibleConflict(const DistributionLayout *rhs,
                                                OpOperand &opOperand) {
  assert(rhs && "layout to resolve with should not be null");
  return resolveWithPossibleConflict(rhs->vectorLayout, opOperand);
}

ChangeResult DistributionLayout::resolve(const VectorLayoutInterface &rhs) {
  ResolutionResult result = doResolution(rhs);

  switch (result) {
  case ResolutionResult::NoChange:
    return ChangeResult::NoChange;
  case ResolutionResult::Change:
    return ChangeResult::Change;
  case ResolutionResult::Conflict: {
    llvm::errs() << "Layout conflict at: " << *this << "\n";
    llvm::errs() << "With: " << rhs << "\n";
    llvm_unreachable("Layout conflict should have been handled with "
                     "resolveWithPossibleConflict instead");
  }
  }
}

ChangeResult DistributionLayout::resolve(const DistributionLayout *rhs) {
  assert(rhs && "layout to resolve with should not be null");
  return resolve(rhs->vectorLayout);
}

void DistributionLayout::print(raw_ostream &os) const {
  if (vectorLayout) {
    os << " " << vectorLayout;
  } else {
    os << "Uninitialized";
  }
}

void DistributionLayout::onUpdate(DataFlowSolver *solver) const {
  AnalysisState::onUpdate(solver);

  Value value = point.get<Value>();

  if (propagation) {
    // Make propagation run again on all users of this value.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({user, propagation});
    }
    // TODO: Maybe we need to run it on the parent operation as well to give
    // layout to other results? Seems unlikely though as results usually
    // don't need the same layout?
  }

  if (enforcement) {
    // Make enforcement run on the parent.
    if (Operation *definingOp = value.getDefiningOp()) {
      solver->enqueue({definingOp, enforcement});
    } else {
      // TODO: This is not always correct. Ideally, we should enqueue all
      // predecessors of these block arguements.
      solver->enqueue({value.getParentBlock()->getParentOp(), enforcement});
    }

    // Enforce users of this value also, as some other operands may need to
    // be updated.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({user, enforcement});
    }
  }
}

/// ==========================================================================
///        TRANSFER FUNCTIONS UTILITIES
/// ==========================================================================

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

/// ==========================================================================
///        PROPAGATION TRANSFER FUNCTIONS
/// ==========================================================================

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

  // Otherwise, try resolving with init.
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

void propagationTransferFunction(
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

/// ==========================================================================
///        ENFORCEMENT TRANSFER FUNCTIONS
/// ==========================================================================

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

void enforcementTransferFunction(
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

/// ==========================================================================
///        PropagateLayout
/// ==========================================================================

LogicalResult PropagateLayout::initialize(Operation *root) {
  // Set layout for anchor ops.
  for (auto [val, layout] : anchors) {
    DistributionLayout *latticeEl = getLatticeElement(val);
    ChangeResult changed = latticeEl->resolve(layout);
    propagateIfChanged(latticeEl, changed);
  }

  root->walk([&](Operation *traversed) { visitOperation(traversed); });

  return success();
}

LogicalResult PropagateLayout::visit(ProgramPoint point) {
  if (Operation *op = dyn_cast_or_null<Operation *>(point)) {
    visitOperation(op);
    return success();
  }

  // Do not expect anything other than an operation.
  return failure();
}

void PropagateLayout::visitOperation(Operation *op) {
  // Handle region branching control flow.
  // TODO: Write more about what we are doing here.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionSuccessors(branch, RegionBranchPoint::parent(),
                          branch->getOperands());
    return;
  }

  if (auto yield = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(yield->getParentOp())) {
      visitRegionSuccessors(branch, RegionBranchPoint(yield->getParentRegion()),
                            yield->getOperands());
      return;
    }
  }

  // TODO: Handle BranchOpInterface also.

  // Grab the lattice elements of the operands.
  SmallVector<const DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (!isa<VectorType>(operand.getType())) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(operand);
    operandLattices.push_back(operandLattice);
  }

  // Get the result lattices.
  SmallVector<DistributionLayout *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    if (!isa<VectorType>(result.getType())) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(result);
    resultLattices.push_back(resultLattice);
  }

  // Exit early on operations with no results.
  if (resultLattices.size() == 0) {
    return;
  }

  auto changeFunc = [&](DistributionLayout *lattice, ChangeResult changed) {
    this->propagateIfChanged(lattice, changed);
  };

  propagationTransferFunction(op, operandLattices, resultLattices, changeFunc);
}

void PropagateLayout::visitRegionSuccessors(RegionBranchOpInterface branch,
                                            RegionBranchPoint branchPoint,
                                            OperandRange operands) {
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(branchPoint, successors);
  for (RegionSuccessor successor : successors) {
    ValueRange inputs = successor.getSuccessorInputs();

    // Get vector layouts for forwarded operands.
    SmallVector<const DistributionLayout *> forwardedLattices;
    for (Value operand : operands) {
      if (isa<VectorType>(operand.getType())) {
        forwardedLattices.push_back(getLatticeElement(operand));
      }
    }

    // Get vector layouts for input operands.
    SmallVector<DistributionLayout *> inputLattices;
    for (Value operand : inputs) {
      if (isa<VectorType>(operand.getType())) {
        inputLattices.push_back(getLatticeElement(operand));
      }
    }

    // Both should have same number of vector operands.
    assert(forwardedLattices.size() == inputLattices.size() &&
           "Number of forwarded operands and inputs should match");

    // Propagate the layouts.
    for (auto [forwardedLattice, inputLattice] :
         llvm::zip(forwardedLattices, inputLattices)) {
      inputLattice->resolve(forwardedLattice);
    }
  }
}

DistributionLayout *PropagateLayout::getLatticeElement(Value val) {
  // Add dependency of operation on the analysis state.
  assert(isa<VectorType>(val.getType()) && "Lattice value should be a vector");
  DistributionLayout *layout =
      DataFlowAnalysis::getOrCreate<DistributionLayout>(val);
  // Subscribe this analysis to updates of the lattice.
  layout->subscribePropagation(this);
  return layout;
}

/// ==========================================================================
///        Enforce Layout
/// ==========================================================================

LogicalResult EnforceLayout::initialize(Operation *root) {
  root->walk([&](Operation *traversed) { visitOperation(traversed); });
  return success();
}

LogicalResult EnforceLayout::visit(ProgramPoint point) {
  if (Operation *op = dyn_cast_or_null<Operation *>(point)) {
    visitOperation(op);
    return success();
  }

  // Do not expect anything else.
  return failure();
}

void EnforceLayout::visitOperation(Operation *op) {
  // Handle region branching control flow.
  // TODO: Write more about what we are doing here.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    visitRegionSuccessors(branch, RegionBranchPoint::parent(),
                          branch->getOpOperands());
    return;
  }

  if (auto yield = dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(yield->getParentOp())) {
      visitRegionSuccessors(branch, RegionBranchPoint(yield->getParentRegion()),
                            yield->getOpOperands());
      return;
    }
  }

  // TODO: Handle BranchOpInterface also.

  // Grab the lattice elements of the operands.
  SmallVector<DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (!isa<VectorType>(operand.getType())) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(operand);
    operandLattices.push_back(operandLattice);
  }

  // Exit early on operations with no results.
  if (operandLattices.size() == 0) {
    return;
  }

  // Get the result lattices.
  SmallVector<const DistributionLayout *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    if (!isa<VectorType>(result.getType())) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(result);
    resultLattices.push_back(resultLattice);
  }

  auto changeFunc = [&](DistributionLayout *lattice, ChangeResult changed) {
    this->propagateIfChanged(lattice, changed);
  };

  enforcementTransferFunction(op, operandLattices, resultLattices, changeFunc);
}

void EnforceLayout::visitRegionSuccessors(RegionBranchOpInterface branch,
                                          RegionBranchPoint branchPoint,
                                          MutableArrayRef<OpOperand> operands) {
  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(branchPoint, successors);
  for (RegionSuccessor successor : successors) {
    ValueRange inputs = successor.getSuccessorInputs();

    // Get vector layouts for forwarded operands.
    SmallVector<DistributionLayout *> forwardedLattices;
    SmallVector<OpOperand *> forwardedOperands;
    for (OpOperand &use : operands) {
      Value operand = use.get();
      if (isa<VectorType>(operand.getType())) {
        forwardedLattices.push_back(getLatticeElement(operand));
        forwardedOperands.push_back(&use);
      }
    }

    // Get vector layouts for input operands.
    SmallVector<const DistributionLayout *> inputLattices;
    for (Value operand : inputs) {
      if (isa<VectorType>(operand.getType())) {
        inputLattices.push_back(getLatticeElement(operand));
      }
    }

    // Both should have same number of vector operands.
    assert(forwardedLattices.size() == inputLattices.size() &&
           "Number of forwarded operands and inputs should match");

    // Propagate the layouts.
    int64_t curr = 0;
    for (auto [forwardedLattice, inputLattice] :
         llvm::zip(forwardedLattices, inputLattices)) {
      forwardedLattice->resolveWithPossibleConflict(inputLattice,
                                                    *forwardedOperands[curr]);
      curr++;
    }
  }
}

DistributionLayout *EnforceLayout::getLatticeElement(Value val) {
  // Add dependency of operation on the analysis state.
  assert(isa<VectorType>(val.getType()) && "Lattice value should be a vector");
  DistributionLayout *layout =
      DataFlowAnalysis::getOrCreate<DistributionLayout>(val);
  // Subscribe this analysis to updates of the lattice.
  layout->subscribeEnforcement(this);
  return layout;
}

LogicalResult VectorLayoutAnalysis::run() {
  // The order of loading matters here, because propagateLayout does anchoring
  // initialization which needs the lattice to know both enforcement and
  // propagation.
  solver.load<EnforceLayout>(root->getContext());
  solver.load<PropagateLayout>(anchors, root->getContext());
  return solver.initializeAndRun(root);
}

VectorLayoutInterface VectorLayoutAnalysis::getLayout(Value val) {
  const DistributionLayout *layout =
      solver.lookupState<DistributionLayout>(val);
  if (!layout) {
    return VectorLayoutInterface();
  }
  return layout->getInnerLayout();
}
