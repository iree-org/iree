// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-vector-layout-analysis"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE::VectorExt;

// Forward declarations.
class DistributionLayout;
class PropagateLayout;
class EnforceLayout;
class VectorLayoutSolver;

using VectorValue = TypedValue<VectorType>;

/// The result of a resolution.
/// Change: The layout was changed.
/// Conflict: The layout was not changed because there was a conflict.
/// NoChange: The layout was not changed because it was already the same.
enum class ResolutionResult {
  Change,
  Conflict,
  NoChange,
};

enum class TransitionKind {
  Forward,
  Backward,
};

class DistributionLayout {
public:
  explicit DistributionLayout(VectorValue val, VectorLayoutSolver &solver)
      : val(val), solver(solver) {}

  VectorValue getValue() const { return val; }

  /// Resolve the current lattice with the given layout. Creates a conflict
  /// resolution if there is a conflict between the two layouts. When a conflict
  /// resolution is created, the layout of the current lattice is resolved,
  /// but all users of the current lattice are updated to use the old layout.
  void resolve(const DistributionLayout *rhs);
  void resolve(const VectorLayoutInterface &rhs);

  /// Resolve the current lattice with the given layout. Creates a conflict
  /// resolution if there is a conflict between the two layouts. When a conflict
  /// resolution is created, the given "use" is updated to use the resolved
  /// layout.
  ///
  /// Returns the lattice element for the use after resolution.
  [[nodiscard]] DistributionLayout *resolveValUse(const DistributionLayout *rhs,
                                                  OpOperand &use);
  [[nodiscard]] DistributionLayout *
  resolveValUse(const VectorLayoutInterface &rhs, OpOperand &use);

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

  void print(raw_ostream &os) const;
  void dump() const;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const DistributionLayout &layout) {
    layout.print(os);
    return os;
  }

private:
  /// When the lattice gets updated, broadcast it's changes to all potential
  /// users of this lattice.
  void broadcastChanges() const;

  /// Attempt to resolve the current lattice with the given lattice. Returns if
  /// the current layout was not changed, changed or if there was a layout
  /// conflict.
  ResolutionResult doResolution(const VectorLayoutInterface &rhs);

  /// Set the layout for this lattice element to the given layout. This function
  /// should only be used when you know there will be no layout conflicts.
  /// Otherwise, the resolve-like functions should be used.
  void setInnerLayout(const VectorLayoutInterface &layout) {
    assert(!layout || layout.isValidLayout(getValue().getType().getShape()));
    vectorLayout = layout;
  }

  /// The vector value that this lattice element represents.
  VectorValue val;

  /// The layout of the vector SSA Value.
  VectorLayoutInterface vectorLayout;

  VectorLayoutSolver &solver;
};

class EnforceLayout {
public:
  explicit EnforceLayout(VectorLayoutSolver &solver) : solver(solver) {}

  void visit(Operation *op);

private:
  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             MutableArrayRef<OpOperand> operands);

  DistributionLayout *getLatticeElement(VectorValue val);

  VectorLayoutSolver &solver;
};

class PropagateLayout {
public:
  explicit PropagateLayout(VectorLayoutSolver &solver) : solver(solver) {}

  void visit(Operation *op);

private:
  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             OperandRange operands);

  DistributionLayout *getLatticeElement(VectorValue val);

  VectorLayoutSolver &solver;
};

class VectorLayoutSolver {
public:
  VectorLayoutSolver(MLIRContext *ctx)
      : ctx(ctx), propagation(*this), enforcement(*this) {}

  DistributionLayout *getLatticeElement(VectorValue val);

  void enqueueLowPriority(Operation *op, TransitionKind kind) {
    worklist.emplace_back(op, kind);
  }

  void enqueueHighPriority(Operation *op, TransitionKind kind) {
    worklist.emplace_front(op, kind);
  }

  void solve();

private:
  MLIRContext *ctx;

  PropagateLayout propagation;
  EnforceLayout enforcement;

  /// The lattice for each value.
  DenseMap<VectorValue, std::unique_ptr<DistributionLayout>> lattices;

  /// The solver's work queue. Work items can be inserted to the front of the
  /// queue to be processed greedily.
  std::deque<std::pair<Operation *, TransitionKind>> worklist;
};

/// ==========================================================================
///        DistributionLayout
/// ==========================================================================

ResolutionResult
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
    // Broadcast changes to all users of this lattice.
    broadcastChanges();
    return ResolutionResult::Change;
  }

  // Layouts conflict and need to be resolved.
  return ResolutionResult::Conflict;
}

DistributionLayout *
DistributionLayout::resolveValUse(const VectorLayoutInterface &rhs,
                                  OpOperand &opOperand) {
  ResolutionResult result = doResolution(rhs);

  // If there is no conflict, simply return.
  if (result != ResolutionResult::Conflict) {
    return this;
  }

  // Resolve conflict by create an operation that takes the input the
  // conflicted value and returns the resolved value.
  OpBuilder builder(opOperand.getOwner());
  Value input = opOperand.get();
  // Create a resolution operation. This conflict should be handeled later by
  // someone else, not this analysis.
  Operation *resolveOp =
      builder.create<IREE::VectorExt::LayoutConflictResolutionOp>(
          input.getLoc(), input.getType(), input, vectorLayout, rhs);
  VectorValue resolvedValue = cast<VectorValue>(resolveOp->getResult(0));
  opOperand.set(resolvedValue);

  // We can now resolve this resolved value to the required layout.
  DistributionLayout *resolvedLayout = solver.getLatticeElement(resolvedValue);
  resolvedLayout->resolve(rhs);

  return resolvedLayout;
}

DistributionLayout *
DistributionLayout::resolveValUse(const DistributionLayout *rhs,
                                  OpOperand &opOperand) {
  assert(rhs && "layout to resolve with should not be null");
  return resolveValUse(rhs->vectorLayout, opOperand);
}

void DistributionLayout::resolve(const VectorLayoutInterface &rhs) {
  ResolutionResult result = doResolution(rhs);

  if (result != ResolutionResult::Conflict) {
    return;
  }

  // Resolve conflict by create an operation that takes the input the
  // resolved value and returns the conflicted value.
  OpBuilder builder(getValue().getContext());

  // Save the conflicted layout for the users.
  VectorLayoutInterface conflictedLayout = getLayout();

  // Set the layout of this value to the resolved layout.
  setInnerLayout(rhs);

  // Create a resolution operation. This conflict should be handeled later by
  // someone else, not this analysis.
  VectorValue resolvedValue = getValue();
  Operation *resolveOp =
      builder.create<IREE::VectorExt::LayoutConflictResolutionOp>(
          resolvedValue.getLoc(), resolvedValue.getType(), resolvedValue, rhs,
          conflictedLayout);
  VectorValue conflictedValue = cast<VectorValue>(resolveOp->getResult(0));

  // Replace all uses of the resolved value with the conflicted value.
  resolvedValue.replaceAllUsesWith(conflictedValue);

  // We can now give the conflicted value the old layout.
  DistributionLayout *conflictedLattice =
      solver.getLatticeElement(conflictedValue);
  conflictedLattice->resolve(conflictedLayout);
}

void DistributionLayout::resolve(const DistributionLayout *rhs) {
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

void DistributionLayout::broadcastChanges() const {
  VectorValue value = getValue();

  // Forward analyse all users of this value.
  // We give high priority to this forward analysis.
  for (Operation *user : value.getUsers()) {
    solver.enqueueHighPriority(user, TransitionKind::Forward);
  }

  // Backward analyse all users of this value with atleast 2 operands. This is
  // done to enforce layout changes on other operands of the same operation.
  // We give high priority to this backward analysis.
  for (Operation *user : value.getUsers()) {
    if (user->getNumOperands() >= 2) {
      solver.enqueueHighPriority(user, TransitionKind::Backward);
    }
  }

  // Backward analyse the parent operation of this value.
  // We give low priority to this backward analysis, since this generates
  // conflicts and we prefer to resolve conflicts as late as possible, as
  // so that we have as much information as possible.
  if (Operation *definingOp = value.getDefiningOp()) {
    solver.enqueueLowPriority(definingOp, TransitionKind::Backward);
  } else {
    // TODO(Groverkss): This may not always be correct (I cannot prove it right
    // now). Ideally, we should enqueue all predecessors of these block
    // arguements. The expectation here is that the parent op will recursively
    // enqueue all its predecessors.
    solver.enqueueLowPriority(value.getParentBlock()->getParentOp(),
                              TransitionKind::Backward);
  }

  // TODO(Groverkss): Ideally, we should also enqueue forward analysis on
  // the parent operation of this value. But, usually, there are no constraints
  // on two results of the same operation. If in future we have such
  // constraints, we should add an enqueue here.
}

DistributionLayout *EnforceLayout::getLatticeElement(VectorValue val) {
  return solver.getLatticeElement(val);
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

/// Get a layout if all the given layouts are same. If all layouts are not
/// same, return nullptr.
static const DistributionLayout *
getAgreedLayout(ArrayRef<const DistributionLayout *> layouts) {
  if (layouts.empty())
    return nullptr;

  // Check if all layouts are same.
  if (!llvm::all_equal(llvm::make_pointee_range(layouts))) {
    return nullptr;
  }

  return layouts[0];
}

/// Hueristic to use to choose the best layout when enforcing the same layout
/// to all operands. Current hueristic is to simply choose the first operand
/// which has a layout.
/// TODO(Groverkss): Use a better hueristic.
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
static void
enforceSameLayoutForOperands(Operation *op,
                             MutableArrayRef<DistributionLayout *> operands) {
  DistributionLayout *chosenOperandLayout =
      enforceSameLayoutHueristic(operands);

  // Enforce the layout to other operands.
  if (chosenOperandLayout) {
    for (auto [index, lattice] : llvm::enumerate(operands)) {
      OpOperand &opOperand = getOpOperand(op, index);
      // Update the lattice with the resolved layout.
      lattice = lattice->resolveValUse(chosenOperandLayout, opOperand);
    }
  }
}

/// ==========================================================================
///        PROPAGATION TRANSFER FUNCTIONS
/// ==========================================================================

static void propagateLayoutToElementwiseOp(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  DistributionLayout *result = resultLattices[0];

  // If result lattice already has a layout, we cannot do
  // anything. We do not impose layout conflicts on results.
  // TODO(Groverkss): Explore if this is actually needed.
  if (result->hasLayout()) {
    return;
  }

  // Check if all vector operands agree on the same layout.
  const DistributionLayout *chosenOperandLayout =
      getAgreedLayout(operandLattices);
  if (chosenOperandLayout == nullptr) {
    return;
  }

  result->resolve(chosenOperandLayout);
}

static void propagateLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices) {
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

  // If the vector begin reduced has a layout, then propagate it to the
  // result. by projecting
  if (vector->hasLayout()) {
    SmallVector<bool> reductionMask = multiReduce.getReductionMask();
    result->resolve(vector->getLayout().project(reductionMask));
    return;
  }

  // Otherwise, try resolving with init.
  result->resolve(init);
}

static void propagateLayoutToTransposeOp(
    vector::TransposeOp transpose,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices) {
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
  result->resolve(permutedLayout);
}

static void propagateLayoutToContractionOp(
    vector::ContractionOp contraction,
    ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices) {
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
  result->resolve(init);
}

void propagationTransferFunction(
    Operation *op, ArrayRef<const DistributionLayout *> operandLattices,
    ArrayRef<DistributionLayout *> resultLattices) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    propagateLayoutToElementwiseOp(op, operandLattices, resultLattices);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    propagateLayoutToMultiReductionOp(multiReduce, operandLattices,
                                      resultLattices);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    propagateLayoutToTransposeOp(transpose, operandLattices, resultLattices);
    return;
  }

  if (auto contraction = dyn_cast<vector::ContractionOp>(op)) {
    propagateLayoutToContractionOp(contraction, operandLattices,
                                   resultLattices);
    return;
  }

  return;
}

/// ==========================================================================
///        ENFORCEMENT TRANSFER FUNCTIONS
/// ==========================================================================

static void enforceLayoutToElementwiseOp(
    Operation *op, MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {
  // All operands and results must agree on the same layout.

  // We do not support multiple results yet.
  if (resultLattices.size() != 1)
    return;

  // Try to enforce the layout of the result on operands.
  const DistributionLayout *result = resultLattices[0];
  if (result->hasLayout()) {
    for (auto [index, operandLattice] : llvm::enumerate(operandLattices)) {
      // Update the operand lattice with the resolved layout.
      operandLattice =
          operandLattice->resolveValUse(result, getOpOperand(op, index));
    }
  } else {
    // Enforce the same layout on all operands.
    enforceSameLayoutForOperands(op, operandLattices);
  }
}

static void enforceLayoutToMultiReductionOp(
    vector::MultiDimReductionOp multiReduce,
    MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {
  // Reductions should always propagate value layout to result. Result can
  // enforce it's layout on init.
  const DistributionLayout *result = resultLattices[0];
  DistributionLayout *init = operandLattices[1];

  // Enforce the result layout on init.
  init = init->resolveValUse(result, getOpOperand(multiReduce, 1));
  operandLattices[1] = init;
}

static void enforceLayoutToTransposeOp(
    vector::TransposeOp transpose,
    MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {
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
  VectorLayoutInterface permutedLayout = result->getLayout().permute(perm);

  // Try to resolve with the transposed layout.
  value = value->resolveValUse(permutedLayout, getOpOperand(transpose, 0));
  // Update the operand lattice.
  operandLattices[0] = value;
}

static void enforceLayoutToBroadcastOp(
    vector::BroadcastOp broadcast,
    MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {
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
  assert(isa<VectorType>(inputType) &&
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
      result->getLayout().project(reductionMask);
  value = value->resolveValUse(resultLayout, getOpOperand(broadcast, 0));
  operandLattices[0] = value;
}

static void enforceLayoutToContractionOp(
    vector::ContractionOp contraction,
    MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {
  // Contraction has only one vector result.
  const DistributionLayout *result = resultLattices[0];
  // Contraction has init value at position 2.
  DistributionLayout *value = operandLattices[2];

  // Cannot enforce layout if result is uninitialized.
  if (result->isUninitialized()) {
    return;
  }

  // True to resolve the init value with the result layout.
  value = value->resolveValUse(result, getOpOperand(contraction, 2));
  // Update the operand lattice.
  operandLattices[2] = value;
}

void enforcementTransferFunction(
    Operation *op, MutableArrayRef<DistributionLayout *> operandLattices,
    ArrayRef<const DistributionLayout *> resultLattices) {

  // Propagate layout to elementwise operations.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    enforceLayoutToElementwiseOp(op, operandLattices, resultLattices);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    enforceLayoutToMultiReductionOp(multiReduce, operandLattices,
                                    resultLattices);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    enforceLayoutToTransposeOp(transpose, operandLattices, resultLattices);
    return;
  }

  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
    enforceLayoutToBroadcastOp(broadcast, operandLattices, resultLattices);
    return;
  }

  if (auto contraction = dyn_cast<vector::ContractionOp>(op)) {
    enforceLayoutToContractionOp(contraction, operandLattices, resultLattices);
    return;
  }
}

/// ==========================================================================
///        PropagateLayout
/// ==========================================================================

void PropagateLayout::visit(Operation *op) {
  // Handle region branching control flow.
  // TODO(Groverkss): Write more about what we are doing here.
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

  // TODO(Groverkss): Handle BranchOpInterface also.

  // Grab the lattice elements of the operands.
  SmallVector<const DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto opValue = dyn_cast<VectorValue>(operand);
    if (!opValue) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(opValue);
    operandLattices.push_back(operandLattice);
  }

  // Get the result lattices.
  SmallVector<DistributionLayout *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    auto resValue = dyn_cast<VectorValue>(result);
    if (!resValue) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(resValue);
    resultLattices.push_back(resultLattice);
  }

  // Exit early on operations with no results.
  if (resultLattices.empty()) {
    return;
  }

  propagationTransferFunction(op, operandLattices, resultLattices);
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
      if (auto opValue = dyn_cast<VectorValue>(operand)) {
        forwardedLattices.push_back(getLatticeElement(opValue));
      }
    }

    // Get vector layouts for input operands.
    SmallVector<DistributionLayout *> inputLattices;
    for (Value operand : inputs) {
      if (auto opValue = dyn_cast<VectorValue>(operand)) {
        inputLattices.push_back(getLatticeElement(opValue));
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

DistributionLayout *PropagateLayout::getLatticeElement(VectorValue val) {
  return solver.getLatticeElement(val);
}

/// ==========================================================================
///        Enforce Layout
/// ==========================================================================

void EnforceLayout::visit(Operation *op) {
  // Handle region branching control flow.
  // TODO(Groverkss): Write more about what we are doing here.
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

  // TODO(Groverkss): Handle BranchOpInterface also.

  // Grab the lattice elements of the operands.
  SmallVector<DistributionLayout *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto opValue = dyn_cast<VectorValue>(operand);
    if (!opValue) {
      continue;
    }

    DistributionLayout *operandLattice = getLatticeElement(opValue);
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
    auto resValue = dyn_cast<VectorValue>(result);
    if (!resValue) {
      continue;
    }

    DistributionLayout *resultLattice = getLatticeElement(resValue);
    resultLattices.push_back(resultLattice);
  }

  enforcementTransferFunction(op, operandLattices, resultLattices);
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
      if (auto opValue = dyn_cast<VectorValue>(operand)) {
        forwardedLattices.push_back(getLatticeElement(opValue));
        forwardedOperands.push_back(&use);
      }
    }

    // Get vector layouts for input operands.
    SmallVector<const DistributionLayout *> inputLattices;
    for (Value operand : inputs) {
      if (auto opValue = dyn_cast<VectorValue>(operand)) {
        inputLattices.push_back(getLatticeElement(opValue));
      }
    }

    // Both should have same number of vector operands.
    assert(forwardedLattices.size() == inputLattices.size() &&
           "Number of forwarded operands and inputs should match");

    // Propagate the layouts.
    int64_t curr = 0;
    for (auto [forwardedLattice, inputLattice] :
         llvm::zip(forwardedLattices, inputLattices)) {
      // Update the lattice for the forwarded operand with the resolved layout.
      forwardedLattice = forwardedLattice->resolveValUse(
          inputLattice, *forwardedOperands[curr]);
      curr++;
    }
  }
}

/// ==========================================================================
///        VectorLayoutSolver
/// ==========================================================================

DistributionLayout *VectorLayoutSolver::getLatticeElement(VectorValue val) {
  auto it = lattices.find(val);
  if (it == lattices.end()) {
    // Create a new lattice element for the value.
    lattices[val] = std::make_unique<DistributionLayout>(val, *this);
    return lattices[val].get();
  }
  return it->second.get();
}

void VectorLayoutSolver::solve() {
  while (!worklist.empty()) {
    auto [op, kind] = worklist.front();
    worklist.pop_front();

    switch (kind) {
    case TransitionKind::Forward:
      propagation.visit(op);
      break;
    case TransitionKind::Backward:
      enforcement.visit(op);
      break;
    }
  }
}

/// ==========================================================================
///        VectorLayoutAnalysis
/// ==========================================================================

VectorLayoutAnalysis::VectorLayoutAnalysis(Operation *root) : root(root) {
  solver = new VectorLayoutSolver(root->getContext());
}

VectorLayoutAnalysis::~VectorLayoutAnalysis() {
  delete (VectorLayoutSolver *)(solver);
}

void VectorLayoutAnalysis::setAnchorForResult(VectorValue val,
                                              Attribute layout) {
  VectorLayoutSolver *layoutSolver = (VectorLayoutSolver *)solver;
  DistributionLayout *lattice = layoutSolver->getLatticeElement(val);
  VectorLayoutInterface vectorLayout =
      dyn_cast_or_null<VectorLayoutInterface>(layout);
  if (!vectorLayout) {
    llvm::report_fatal_error(
        "Layouts used in layout analysis must be vector layouts");
  }
  lattice->resolve(vectorLayout);
  layoutSolver->solve();
}

void VectorLayoutAnalysis::setAnchorForOperand(OpOperand &operand,
                                               Attribute layout) {
  VectorLayoutSolver *layoutSolver = (VectorLayoutSolver *)solver;
  VectorValue opValue = dyn_cast<VectorValue>(operand.get());
  if (!opValue) {
    llvm::report_fatal_error(
        "Operands used in layout analysis must be vector values");
  }
  DistributionLayout *lattice = layoutSolver->getLatticeElement(opValue);
  VectorLayoutInterface vectorLayout =
      dyn_cast_or_null<VectorLayoutInterface>(layout);
  if (!vectorLayout) {
    llvm::report_fatal_error(
        "Layouts used in layout analysis must be vector layouts");
  }
  (void)lattice->resolveValUse(vectorLayout, operand);
  layoutSolver->solve();
}

VectorLayoutInterface VectorLayoutAnalysis::getLayout(VectorValue val) {
  VectorLayoutSolver *layoutSolver = (VectorLayoutSolver *)solver;
  const DistributionLayout *layout = layoutSolver->getLatticeElement(val);
  return layout->getLayout();
}

void VectorLayoutAnalysis::debugAnnotateLayouts() {
  // Annotate each operation with the layout of it's result.
  root->walk([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }

    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      auto resVal = dyn_cast<VectorValue>(result);
      if (!resVal) {
        continue;
      }

      // Do not annotate resolve_conflict operations since they already have
      // this information in their attributes.
      if (isa<IREE::VectorExt::LayoutConflictResolutionOp>(op)) {
        continue;
      }

      auto layout = getLayout<VectorLayoutInterface>(resVal);
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
