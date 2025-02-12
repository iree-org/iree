// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"

#include <cassert>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

#define DEBUG_TYPE "iree-vector-layout-analysis"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::VectorExt;

// Forward declarations.
class DistributionLayout;
class PropagateLayout;
class EnforceLayout;

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  TypedValue<VectorType> getValue() const {
    auto anchor = getAnchor();
    assert(isa<Value>(anchor) && "expected anchor to be a value");
    Value val = cast<Value>(anchor);
    assert(isa<VectorType>(val.getType()) &&
           "expected value to be of vector type");
    return cast<TypedValue<VectorType>>(val);
  }

  /// TODO: This currently, creates a new value but doesn't replace it with the
  /// current value, because that would be wrong. Find a way to take care of
  /// that better.
  ChangeResult resolveWithPossibleConflict(const DistributionLayout *rhs,
                                           OpOperand &operand);
  ChangeResult resolveWithPossibleConflict(const VectorLayoutInterface &rhs,
                                           OpOperand &operand);

  ChangeResult resolve(const DistributionLayout *rhs, bool force = false);
  ChangeResult resolve(const VectorLayoutInterface &rhs, bool force = false);

  VectorLayoutInterface getLayout() const { return vectorLayout; }

  bool isUninitialized() const { return !vectorLayout; }
  bool hasLayout() const { return !isUninitialized(); }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return vectorLayout == rhs.vectorLayout;
  }
  bool operator!=(const DistributionLayout &rhs) const {
    return !(*this == rhs);
  }

  void print(raw_ostream &os) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice
  /// changes, subscribed analyses are re-invoked. This is more efficient than
  /// relying on the dependency map.
  void subscribePropagation(PropagateLayout *analysis) {
    propagation = analysis;
  }
  void subscribeEnforcement(EnforceLayout *analysis) { enforcement = analysis; }

private:
  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// The result of a resolution.
  /// Change: The layout was changed.
  /// Conflict: The layout was not changed because there was a conflict.
  /// NoChange: The layout was not changed because it was already the same.
  enum ResolutionResult {
    Change,
    Conflict,
    NoChange,
  };

  /// Attempt to resolve the current lattice with the given lattice. Returns if
  /// the current layout was not changed, changed or if there was a layout
  /// conflict.
  ResolutionResult doResolution(const VectorLayoutInterface &rhs);

  /// Set the layout for this lattice element to the given layout. This function
  /// should only be used when you know there will be no layout conflicts.
  /// Otherwise, the resolve-like functions should be used.
  void setInnerLayout(const VectorLayoutInterface &layout) {
    assert(layout &&
           layout.isValidLayout(getValue().getType(), getValue().getLoc())
               .succeeded());
    vectorLayout = layout;
  }

  /// The layout of the vector SSA Value.
  VectorLayoutInterface vectorLayout;

  /// Each lattice element stores a pointer to the analysis that work on it so
  /// it can notify them when it changes.
  PropagateLayout *propagation = nullptr;
  EnforceLayout *enforcement = nullptr;
};

class EnforceLayout : public DataFlowAnalysis {
public:
  explicit EnforceLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *root) override;

  LogicalResult visit(ProgramPoint *point) override;

  void registerNewValue(Value val, const VectorLayoutInterface &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             MutableArrayRef<OpOperand> operands);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

class PropagateLayout : public DataFlowAnalysis {
public:
  explicit PropagateLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *root) override;

  LogicalResult visit(ProgramPoint *point) override;

  /// Register a new value to be part of the dataflow analysis. The value should
  /// not be part of the analysis already. This is used for new values that are
  /// created.
  void registerNewValue(Value val, const VectorLayoutInterface &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             OperandRange operands);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

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

  IRRewriter builder(opOperand.getOwner());
  // Handle case where constantOp may have multiple consumers with different
  // layouts by creating a copy of constOp for other users.
  if (!opOperand.get().hasOneUse() && !vectorLayout &&
      llvm::isa_and_nonnull<arith::ConstantOp, vector::StepOp>(
          opOperand.get().getDefiningOp())) {
    builder.setInsertionPoint(opOperand.get().getDefiningOp());
    Operation *copiedConstOp = builder.clone(*opOperand.get().getDefiningOp());
    Value copiedConst = copiedConstOp->getResult(0);
    builder.replaceAllUsesExcept(opOperand.get(), copiedConst,
                                 opOperand.getOwner());
  }

  ResolutionResult result = doResolution(rhs);

  // If there is no conflict, simply return.
  if (result == ResolutionResult::NoChange) {
    return ChangeResult::NoChange;
  }
  if (result == ResolutionResult::Change) {
    return ChangeResult::Change;
  }

  // Resolve conflict by create an operation that takes the input the conflicted
  // value and returns the resolved value.
  Value input = opOperand.get();
  // Create a resolution operation. This conflict should be handeled later by
  // someone else, not this analysis.
  Operation *resolveOp =
      builder.create<IREE::VectorExt::ToLayoutOp>(input.getLoc(), input, rhs);
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
  (void)resolvedLayout->resolve(rhs);

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

ChangeResult DistributionLayout::resolve(const VectorLayoutInterface &rhs,
                                         bool force) {
  // If forced, set the layout regardless of a possible conflict.
  if (force) {
    bool changed = (vectorLayout != rhs);
    setInnerLayout(rhs);
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ResolutionResult result = doResolution(rhs);

  switch (result) {
  case ResolutionResult::NoChange:
    return ChangeResult::NoChange;
  case ResolutionResult::Change:
    return ChangeResult::Change;
  case ResolutionResult::Conflict: {
    llvm::errs() << "Layout conflict at: " << *this << "\n";
    llvm::errs() << "With: " << rhs << "\n";
    llvm::report_fatal_error("Layout conflict should have been handled with "
                             "resolveWithPossibleConflict instead");
  }
  }

  // This return will never be reached, but it's here to make the compiler
  // happy.
  return ChangeResult::NoChange;
}

ChangeResult DistributionLayout::resolve(const DistributionLayout *rhs,
                                         bool force) {
  assert(rhs && "layout to resolve with should not be null");
  return resolve(rhs->vectorLayout, force);
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

  Value value = cast<Value>(anchor);

  if (propagation) {
    // Make propagation run again on all users of this value.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({solver->getProgramPointAfter(user), propagation});
    }
    // TODO: Maybe we need to run it on the parent operation as well to give
    // layout to other results? Seems unlikely though as results usually
    // don't need the same layout?
  }

  if (enforcement) {
    // Make enforcement run on the parent.
    if (Operation *definingOp = value.getDefiningOp()) {
      solver->enqueue({solver->getProgramPointAfter(definingOp), enforcement});
    } else {
      // TODO: This is not always correct. Ideally, we should enqueue all
      // predecessors of these block arguements.
      solver->enqueue(
          {solver->getProgramPointAfter(value.getParentBlock()->getParentOp()),
           enforcement});
    }

    // Enforce users of this value also, as some other operands may need to
    // be updated.
    for (Operation *user : value.getUsers()) {
      solver->enqueue({solver->getProgramPointAfter(user), enforcement});
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
    if (isa<VectorType>(operand.get().getType())) {
      if (operandIndex == operandLatticeIndex) {
        return operand;
      }
      ++operandIndex;
    }
  }
  llvm::report_fatal_error("No vector operand found");
}

/// Get a layout if all the given initialized layouts are same.
/// If any initialized layouts are not same, return nullptr.
static const DistributionLayout *
getAgreedLayout(ArrayRef<const DistributionLayout *> layouts) {
  SmallVector<const DistributionLayout *> initializedLayouts;
  for (auto layout : layouts) {
    if (layout->isUninitialized())
      continue;
    initializedLayouts.push_back(layout);
  }

  if (initializedLayouts.empty())
    return nullptr;

  // Check if all layouts are same.
  if (!llvm::all_equal(llvm::make_pointee_range(initializedLayouts)))
    return nullptr;

  return initializedLayouts[0];
}

/// Hueristic to use to choose the best layout when enforcing the same layout
/// to all operands. Current hueristic is to simply choose the first operand
/// which has a layout.
/// TODO: Use a better hueristic.
static DistributionLayout *
enforceSameLayoutHueristic(ArrayRef<DistributionLayout *> operands) {
  DistributionLayout *chosenOperandLayout = nullptr;
  for (DistributionLayout *lattice : operands) {
    if (lattice->hasLayout()) {
      chosenOperandLayout = lattice;
      break;
    }
  }
  return chosenOperandLayout;
}

/// Given a list of layouts for operands, enforce a single layout for all of
/// them.
static void enforceSameLayoutForOperands(
    Operation *op, ArrayRef<DistributionLayout *> operands,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  DistributionLayout *chosenOperandLayout =
      enforceSameLayoutHueristic(operands);

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

static void propagateLayoutToLayoutOp(
    ToLayoutOp toLayout, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  DistributionLayout *result = resultLattices[0];

  // ToLayout operation propagates layout even if the result already has a
  // layout.

  ChangeResult changed = result->resolve(toLayout.getLayout(), /*force=*/true);
  update(result, changed);
}

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
        result->resolve(vector->getLayout().project(reductionMask));
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
  VectorLayoutInterface permutedLayout = value->getLayout().permute(perm);

  // Try to resolve with the transposed layout.
  ChangeResult changed = result->resolve(permutedLayout);
  update(result, changed);
}

static void propagateLayoutToContractionOp(
    vector::ContractionOp contraction,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Contraction has only one vector result.
  DistributionLayout *result = resultLattices[0];
  // Get the init value of the contraction.
  const DistributionLayout *init = operandLattices[2];

  // If result lattice already has a layout, we cannot do anything. We do not
  // impose layout conflicts on results.
  if (result->hasLayout()) {
    return;
  }

  // True to resolve result with init.
  ChangeResult changed = result->resolve(init);
  update(result, changed);
}

static void propagateLayoutToGatherOp(
    vector::GatherOp gather,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  DistributionLayout *result = resultLattices[0];

  const DistributionLayout *indicesLayout = operandLattices[0];

  // If result lattice already has a layout, we cannot do anything. We do not
  // impose layout conflicts on results.
  if (result->hasLayout()) {
    return;
  }

  ChangeResult changed = result->resolve(indicesLayout);
  update(result, changed);
}

void propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
    propagateLayoutToLayoutOp(toLayout, operandLattices, resultLattices,
                              update);
    return;
  }

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

  if (auto contraction = dyn_cast<vector::ContractionOp>(op)) {
    propagateLayoutToContractionOp(contraction, operandLattices, resultLattices,
                                   update);
    return;
  }

  if (auto gather = dyn_cast<vector::GatherOp>(op)) {
    propagateLayoutToGatherOp(gather, operandLattices, resultLattices, update);
    return;
  }

  return;
}

/// ==========================================================================
///        ENFORCEMENT TRANSFER FUNCTIONS
/// ==========================================================================

static void enforceLayoutToLayoutOp(
    ToLayoutOp toLayout, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  DistributionLayout *input = operandLattices[0];

  // If the operand already has a layout, we don't do anything. The result
  // will already have the layout desired by this operation.
  if (input->hasLayout()) {
    return;
  }

  // Enforce the result layout on init.
  ChangeResult changed = input->resolveWithPossibleConflict(
      toLayout.getLayout(), getOpOperand(toLayout, 0));
  update(input, changed);
}

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
  if (resultLattices.empty()) {
    return;
  }
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
  VectorLayoutInterface permutedLayout =
      result->getLayout().permute(invertPermutationVector(perm));

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
  assert(broadcast.computeBroadcastedUnitDims().empty() &&
         "Streching in broadcasting not implemented yet.");
  // The starting k dimensions of the result are the ones that need to be
  // projected out.

  auto resultShape = broadcast.getResultVectorType().getShape();
  auto inputType = broadcast.getSourceType();

  VectorType inputVectorType = dyn_cast<VectorType>(inputType);
  if (!inputVectorType)
    return;

  auto inputShape = inputVectorType.getShape();

  SmallVector<bool> reductionMask(resultShape.size(), false);
  // Set the trailing dimensions to be reduced.
  int64_t resultDiff = resultShape.size() - inputShape.size();
  assert(resultDiff >= 0 && "Result shape cannot be smaller than input shape");
  for (int64_t i = 0; i < resultDiff; ++i) {
    reductionMask[i] = true;
  }

  VectorLayoutInterface resultLayout =
      result->getLayout().project(reductionMask);
  ChangeResult changed = value->resolveWithPossibleConflict(
      resultLayout, getOpOperand(broadcast, 0));
  update(value, changed);
}

static void enforceLayoutToContractionOp(
    vector::ContractionOp contraction,
    ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  if (resultLattices.empty())
    return;
  // Contraction has only one vector result.
  const DistributionLayout *result = resultLattices[0];
  // Contraction has init value at position 2.
  DistributionLayout *value = operandLattices[2];

  // Cannot enforce layout if result is uninitialized.
  if (result->isUninitialized()) {
    return;
  }

  // True to resolve the init value with the result layout.
  ChangeResult changed =
      value->resolveWithPossibleConflict(result, getOpOperand(contraction, 2));
  update(value, changed);
}

static void enforceLayoutToGatherOp(
    vector::GatherOp gather, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {
  // Gather has only one vector result.
  const DistributionLayout *result = resultLattices[0];

  if (result->hasLayout()) {
    // Note that the operand lattice is not updated. So using the operand
    // lattice again can cause bugs.
    for (auto [index, operandLattice] : llvm::enumerate(operandLattices)) {
      ChangeResult changed = operandLattice->resolveWithPossibleConflict(
          result, getOpOperand(gather, index));
      update(operandLattice, changed);
    }
  } else {
    // Enforce the same layout on all operands.
    enforceSameLayoutForOperands(gather, operandLattices, update);
  }
}

void enforcementTransferFunction(
    Operation *op, ArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices,
    std::function<void(DistributionLayout *, ChangeResult)> update) {

  if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
    enforceLayoutToLayoutOp(toLayout, operandLattices, resultLattices, update);
  }

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

  if (auto gather = dyn_cast<vector::GatherOp>(op)) {
    enforceLayoutToGatherOp(gather, operandLattices, resultLattices, update);
    return;
  }

  if (auto contraction = dyn_cast<vector::ContractionOp>(op)) {
    enforceLayoutToContractionOp(contraction, operandLattices, resultLattices,
                                 update);
    return;
  }
}

/// ==========================================================================
///        PropagateLayout
/// ==========================================================================

LogicalResult PropagateLayout::initialize(Operation *root) {
  // Initialize/set anchor layouts.
  root->walk([&](IREE::VectorExt::ToLayoutOp toLayout) {
    Value anchorVal = toLayout.getResult();
    DistributionLayout *latticeEl = getLatticeElement(anchorVal);
    ChangeResult changed =
        latticeEl->resolve(toLayout.getLayout(), /*force=*/true);
    propagateIfChanged(latticeEl, changed);
  });

  root->walk([&](Operation *traversed) { visitOperation(traversed); });

  return success();
}

LogicalResult PropagateLayout::visit(ProgramPoint *point) {
  if (point->isBlockStart())
    return success();

  if (auto op = point->getPrevOp()) {
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
  if (resultLattices.empty()) {
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
      ChangeResult changed = inputLattice->resolve(forwardedLattice);
      propagateIfChanged(inputLattice, changed);
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

LogicalResult EnforceLayout::visit(ProgramPoint *point) {
  if (point->isBlockStart())
    return success();

  if (auto op = point->getPrevOp()) {
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
  if (operandLattices.empty()) {
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
      ChangeResult changed = forwardedLattice->resolveWithPossibleConflict(
          inputLattice, *forwardedOperands[curr]);
      propagateIfChanged(forwardedLattice, changed);
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

/// ==========================================================================
///        VectorLayoutAnalysis
/// ==========================================================================

LogicalResult VectorLayoutAnalysis::run() {
  // The order of loading matters here, because propagateLayout does anchoring
  // initialization which needs the lattice to know both enforcement and
  // propagation.
  solver.load<PropagateLayout>(root->getContext());
  solver.load<EnforceLayout>(root->getContext());
  return solver.initializeAndRun(root);
}

VectorLayoutInterface VectorLayoutAnalysis::getLayout(Value val) {
  const DistributionLayout *layout =
      solver.lookupState<DistributionLayout>(val);
  if (!layout) {
    return VectorLayoutInterface();
  }
  return layout->getLayout();
}

void VectorLayoutAnalysis::debugAnnotateLayouts() {
  // Annotate each operation with the layout of it's result.
  root->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }

    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (!isa<VectorType>(result.getType())) {
        continue;
      }

      // Do not annotate to_layout operations since they already have
      // this information in their attributes.
      if (isa<IREE::VectorExt::ToLayoutOp>(op)) {
        continue;
      }

      Attribute layout = getLayout<Attribute>(result);
      if (!layout) {
        continue;
      }

      op->setAttr("layout_result_" + std::to_string(index), layout);
    }
  });
}

void VectorLayoutAnalysis::print(raw_ostream &os) {
  debugAnnotateLayouts();
  root->print(os);
}

void VectorLayoutAnalysis::dump() {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}
