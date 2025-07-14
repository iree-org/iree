// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#define DEBUG_TYPE "rocdl-use-buffer-instructions"
#define LDBGS(X)                                                               \
  LLVM_DEBUG((llvm::dbgs() << "[" << DEBUG_TYPE << "]") << X << "\n")
#define ADBGS(X) LLVM_DEBUG((llvm::dbgs() << "[thread-uniform]") << X << "\n")
#define DUMP_OP(x) '`' << x->getName() << "` [" << x << "]"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_ROCDLUSEBUFFERINSTRUCTIONSPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"
} // namespace mlir::iree_compiler

namespace {
//===----------------------------------------------------------------------===//
// ThreadUniform
//===----------------------------------------------------------------------===//
/// Class holding the state of the sparse lattice. There are three possible
/// states for the lattice:
/// `uninitialized`: the state is unknown.
/// `uniform`: the value was determined to be uniform across threads.
/// `dependent`: the value is thread dependent.
struct ThreadUniform {
  ThreadUniform() = default;
  /// Creates a dependent state.
  static ThreadUniform getDependent() { return ThreadUniform(dependent); }
  /// Creates a uniform state.
  static ThreadUniform getUniform() { return ThreadUniform(uniform); }
  /// Returns whether this state is uniform.
  bool isUniform() const { return state == uniform; }
  /// Compares two states.
  bool operator==(const ThreadUniform &other) const {
    return state == other.state;
  }
  /// Prints the state to a stream.
  void print(llvm::raw_ostream &s) const {
    s << (state == uninitialized
              ? "uninitialized"
              : (state == uniform ? "uniform" : "dependent"));
  }
  /// Joins two states, where `dependent` is the top state of dataflow.
  static ThreadUniform join(const ThreadUniform &lhs, const ThreadUniform &rhs);

private:
  typedef enum { uninitialized, uniform, dependent } State;
  ThreadUniform(State state) : state(state) {}
  State state = uninitialized;
};

//===----------------------------------------------------------------------===//
// ThreadUniformLattice
//===----------------------------------------------------------------------===//
/// Class holding a lattice for the sparse analysis.
class ThreadUniformLattice : public dataflow::Lattice<ThreadUniform> {
public:
  using Lattice::Lattice;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadUniformLattice);
};

//===----------------------------------------------------------------------===//
// ThreadUniformAnalysis
//===----------------------------------------------------------------------===//
/// The dataflow analysis computing whether a value is thread uniform or not.
class ThreadUniformAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ThreadUniformLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// Sets the lattice to a pessimistic state.
  void setToEntryState(ThreadUniformLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ThreadUniform::getDependent()));
  }

  /// Visits an operation and determines whether it's uniform or not.
  LogicalResult
  visitOperation(Operation *op, ArrayRef<const ThreadUniformLattice *> operands,
                 ArrayRef<ThreadUniformLattice *> results) override;

  /// Handles the uniformity of control-flow arguments and results.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<ThreadUniformLattice *> argLattices,
                               unsigned firstIndex) override;

  /// Override the default handling, this is necessary as control-flow with
  /// `scf.forall` / `scf.in_parallel` is broken, and the fix is large. TODO:
  /// Remove this once the ops have been fixed.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::AbstractSparseLattice *> lattices) override;
};

//===----------------------------------------------------------------------===//
// ROCDLUseBufferInstructionsPass
//===----------------------------------------------------------------------===//
struct ROCDLUseBufferInstructionsPass final
    : mlir::iree_compiler::impl::ROCDLUseBufferInstructionsPassBase<
          ROCDLUseBufferInstructionsPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ThreadUniform
//===----------------------------------------------------------------------===//

ThreadUniform ThreadUniform::join(const ThreadUniform &lhs,
                                  const ThreadUniform &rhs) {
  if (lhs.state == dependent || rhs.state == dependent)
    return getDependent();
  return (lhs.state == uniform || rhs.state == uniform) ? getUniform()
                                                        : ThreadUniform();
}

//===----------------------------------------------------------------------===//
// ThreadUniformAnalysis
//===----------------------------------------------------------------------===//
/// This function is an overly conservative estimate ops that are safe to assume
/// to be uniform. TODO: Encode this in an interface.
static bool maybeDefinitelyWorkgroupUniform(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>() ||
      op->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (isa<IREE::HAL::InterfaceBindingSubspanOp,
          IREE::HAL::InterfaceConstantLoadOp,
          IREE::TensorExt::DispatchTensorLoadOp, IREE::Util::AssumeIntOp,
          IREE::TensorExt::DispatchWorkloadOrdinalOp, LoopLikeOpInterface>(op))
    return true;
  if (isa<affine::AffineDialect>(op->getDialect()))
    return !isa<affine::AffineDmaStartOp, affine::AffineDmaWaitOp>(op);
  return isa<arith::ArithDialect>(op->getDialect()) &&
         op->getNumResults() == 1 &&
         op->getResult(0).getType().isIntOrIndexOrFloat();
}

LogicalResult ThreadUniformAnalysis::visitOperation(
    Operation *op, ArrayRef<const ThreadUniformLattice *> operands,
    ArrayRef<ThreadUniformLattice *> results) {
  // Early exit if any of the operands is already dependent.
  if (llvm::any_of(operands, [&](const ThreadUniformLattice *lattice) {
        return ThreadUniform::getDependent() == lattice->getValue();
      })) {
    for (ThreadUniformLattice *v : results) {
      propagateIfChanged(v, v->join(ThreadUniform::getDependent()));
    }
    ADBGS(" dependent op: " << *op);
    return success();
  }

  // Check if it's a uniform op.
  if (maybeDefinitelyWorkgroupUniform(op)) {
    ADBGS(" uniform op: " << *op);
    for (ThreadUniformLattice *v : results)
      propagateIfChanged(v, v->join(ThreadUniform::getUniform()));
    return success();
  }

  ADBGS(" pessimistic dependent op: " << *op);
  // Be pessimistic about all other ops.
  setAllToEntryStates(results);
  return success();
}

void ThreadUniformAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<ThreadUniformLattice *> argLattices, unsigned firstIndex) {
  auto loop = dyn_cast<LoopLikeOpInterface>(op);

  // Be pessimistic about non loop ops.
  if (!loop) {
    SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, argLattices, firstIndex);
    return;
  }

  // Get the induction variables, and be pessimistic if they cannot be
  // retrieved.
  std::optional<SmallVector<Value>> iV = loop.getLoopInductionVars();
  if (!iV) {
    SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, argLattices, firstIndex);
    return;
  }

  // Get the inits.
  OperandRange inits = loop.getInits();
  assert((iV->size() + inits.size() == argLattices.size() ||
          inits.size() == argLattices.size()) &&
         "unsupported loop-like op");

  // Get the loop structure.
  std::optional<SmallVector<OpFoldResult>> lb = loop.getLoopLowerBounds();
  std::optional<SmallVector<OpFoldResult>> ub = loop.getLoopUpperBounds();
  std::optional<SmallVector<OpFoldResult>> sv = loop.getLoopSteps();
  assert(lb && ub && sv && "unsupported loop-like op");
  assert((iV->front() == argLattices.front()->getAnchor() ||
          (inits.empty() ||
           op->getResult(0) == argLattices.front()->getAnchor())) &&
         "unsupported loop-like op");

  // Helper function to get the state of an op fold result.
  auto getState = [&](OpFoldResult ofr) {
    auto v = dyn_cast<Value>(ofr);
    if (!v)
      return ThreadUniform::getUniform();
    return getLatticeElement(v)->getValue();
  };

  // Get whether the loop is thread uniform based on the structure.
  ThreadUniform value;
  for (auto [lv, uv, s] : llvm::zip(*lb, *ub, *sv)) {
    value = ThreadUniform::join(value, getState(lv));
    value = ThreadUniform::join(value, getState(uv));
    value = ThreadUniform::join(value, getState(s));
  }

  // Handle `scf.forall` in a pessimistic manner.
  if (auto forAll = dyn_cast<scf::ForallOp>(op)) {
    bool isSafeUniform =
        llvm::all_of(forAll.getDeviceMappingAttrs(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr,
                     IREE::Codegen::WorkgroupMappingAttr>(attr);
        });
    value = isSafeUniform
                ? ThreadUniform::join(value, ThreadUniform::getUniform())
                : ThreadUniform::join(value, ThreadUniform::getDependent());
  }

  // This is needed because dataflow is broken, and will call this function to
  // check the results of a control-flow op. TODO: fix dataflow upstream.
  if (inits.size() == argLattices.size()) {
    for (auto [lattice, operand] :
         llvm::zip(argLattices, loop.getRegionIterArgs())) {
      join(lattice, *getLatticeElement(operand));
      propagateIfChanged(lattice, lattice->join(value));
    }
    return;
  }

  // Propagate the state.
  for (ThreadUniformLattice *lattice : argLattices.take_front(iV->size())) {
    propagateIfChanged(lattice, lattice->join(value));
  }
  for (auto [lattice, operand] :
       llvm::zip(argLattices.drop_front(iV->size()), inits)) {
    join(lattice, *getLatticeElement(operand));
    propagateIfChanged(lattice, lattice->join(value));
  }
}

// This method was copied verbatim from `AbstractSparseForwardDataFlowAnalysis`
// and modified to provide partial support for `scf.forall`. TODO: remove this
// method.
void ThreadUniformAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> latticesRaw) {
  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  ArrayRef<ThreadUniformLattice *> lattices(
      reinterpret_cast<ThreadUniformLattice *const *>(latticesRaw.begin()),
      latticesRaw.size());

  for (Operation *op : predecessors->getKnownPredecessors()) {
    // Get the incoming successor operands.
    std::optional<OperandRange> operands;

    // Check if the predecessor is the parent op.
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
      // Otherwise, try to deduce the operands from a region return-like op.
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    } else if (isa<scf::InParallelOp>(op)) {
      // This is a hack until upstream fixes control-flow semantics.
      operands = op->getOperands();
      // Be pessimistic about the inits as we cannot reason about them.
      auto forallOp = cast<scf::ForallOp>(op->getParentOp());
      if (lattices.size() >= forallOp.getNumDpsInits() &&
          forallOp.getNumDpsInits() > 0) {
        ArrayRef<ThreadUniformLattice *> initLattices =
            lattices.take_back(forallOp.getNumDpsInits());
        assert(
            (initLattices.front()->getAnchor() ==
                 forallOp.getTiedBlockArgument(forallOp.getDpsInitOperand(0)) ||
             initLattices.front()->getAnchor() == forallOp.getResult(0)) &&
            "ill-formed forall");
        setAllToEntryStates(initLattices);
      }
    }

    if (!operands) {
      // We can't reason about the data-flow.
      return setAllToEntryStates(lattices);
    }

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty())
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty())
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto it : llvm::zip(*operands, lattices.drop_front(firstIndex)))
      join(std::get<1>(it), *getLatticeElementFor(point, std::get<0>(it)));
  }
}

//===----------------------------------------------------------------------===//
// ROCDLUseBufferInstructionsPass
//===----------------------------------------------------------------------===//

/// Determine whether input source of a extract slice op should be handle by the
/// pass. Currently we only handle a very specific case to not interfere with
/// `ROCDLConfigureBufferInstructions`. TODO: unify these 2 passes.
static bool isValidSource(TypedValue<RankedTensorType> value) {
  auto loadOp = dyn_cast_or_null<
      mlir::iree_compiler::IREE::TensorExt::DispatchTensorLoadOp>(
      value.getDefiningOp());
  if (!loadOp)
    return false;
  auto subspanOp = dyn_cast_or_null<IREE::HAL::InterfaceBindingSubspanOp>(
      loadOp.getSource().getDefiningOp());
  if (!subspanOp)
    return false;
  return !subspanOp->hasAttr(
      IREE::GPU::IREEGPUDialect::UseRocdlBufferInstructionsAttrHelper::
          getNameStr());
}

/// Handles the slice op, possibly inserting a buffer resource cast operation if
/// it was determined buffer ops can be safely used.
static void handleOp(tensor::ExtractSliceOp slice) {
  SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
  // Exit if it is a trivial tensor.
  if (sizes.size() <= 1) {
    LDBGS("- the slice is trivial");
    return;
  }

  IRRewriter rewriter(slice);
  IntegerAttr maxVal =
      rewriter.getIndexAttr(std::numeric_limits<int32_t>::max());

  // Check if all the sizes in the slice are less than 2^31.
  if (!llvm::all_of(sizes, [&](OpFoldResult sz) {
        return ValueBoundsConstraintSet::compare(
            ValueBoundsConstraintSet::Variable(sz),
            ValueBoundsConstraintSet::ComparisonOperator::LT,
            ValueBoundsConstraintSet::Variable(OpFoldResult(maxVal)));
      })) {
    LDBGS("- failed to infer that the slice was within bounds");
    return;
  }

  Type eTy = slice.getResultType().getElementType();
  // Skip if the type is not trivial. TODO: use the data layout to get the size.
  if (!getElementTypeOrSelf(eTy).isIntOrIndexOrFloat()) {
    LDBGS("- this pass can only handle tensors of int, index or floats");
    return;
  }

  // Over approximate the size of index to not require using the data layout.
  unsigned byteSize =
      eTy.isIntOrFloat() ? std::max(eTy.getIntOrFloatBitWidth(), 8u) / 8 : 8;

  // Compute the total extent of the buffer.
  SmallVector<AffineExpr> dims(sizes.size());
  bindSymbolsList(slice.getContext(), MutableArrayRef<AffineExpr>(dims));
  AffineExpr result = dims[0];
  for (AffineExpr e : MutableArrayRef<AffineExpr>(dims).drop_front())
    result = result * e;
  result = result * byteSize;

  // Check if the total extent is less than 2^31.
  SmallVector<ValueBoundsConstraintSet::Variable> szVars =
      llvm::to_vector_of<ValueBoundsConstraintSet::Variable>(sizes);
  if (!ValueBoundsConstraintSet::compare(
          ValueBoundsConstraintSet::Variable(
              AffineMap::get(0, dims.size(), result), szVars),
          ValueBoundsConstraintSet::ComparisonOperator::LT,
          ValueBoundsConstraintSet::Variable(maxVal))) {
    LDBGS("- failed to infer that the slice was within bounds");
    return;
  }

  // Add the cast operation.
  rewriter.setInsertionPointAfter(slice);
  auto buffOp = rewriter.create<IREE::GPU::BufferResourceCastOp>(
      slice.getLoc(), slice.getType(), slice, Value());
  rewriter.replaceAllUsesExcept(slice, buffOp, buffOp);
  LDBGS("- success, added cast: " << buffOp);
}

void ROCDLUseBufferInstructionsPass::runOnOperation() {
  FunctionOpInterface func = getOperation();

  IREE::GPU::TargetAttr target = getGPUTargetAttr(func);
  if (!target || !target.isAMD())
    return;

  // Configure and run the dataflow analysis.
  DataFlowSolver solver;
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<ThreadUniformAnalysis>();

  if (failed(solver.initializeAndRun(func)))
    return signalPassFailure();

  func.walk([&](tensor::ExtractSliceOp slice) {
    LDBGS(" found slice op: " << slice);

    // Skip invalid sources.
    if (!isValidSource(slice.getSource())) {
      LDBGS("- the slice doesn't come from a valid source");
      return;
    }

    // Skip if the slice is not thread uniform.
    if (llvm::any_of(slice.getMixedOffsets(), [&](OpFoldResult operand) {
          auto v = dyn_cast<Value>(operand);
          if (!v)
            return false;
          auto opAnalysis = solver.lookupState<ThreadUniformLattice>(v);
          return !opAnalysis || !opAnalysis->getValue().isUniform();
        })) {
      LDBGS("- the slice offsets are not thread uniform");
      return;
    }

    // Try to handle op.
    handleOp(slice);
  });
}
