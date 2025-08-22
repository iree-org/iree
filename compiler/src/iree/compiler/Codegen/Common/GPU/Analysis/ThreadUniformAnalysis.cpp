// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Analysis/ThreadUniformAnalysis.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-codegen-thread-uniform-analysis"
#define LDBGS(X)                                                               \
  LLVM_DEBUG((llvm::dbgs() << "[" << DEBUG_TYPE << "]") << X << "\n")

namespace mlir::iree_compiler::dataflow {
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

void ThreadUniform::print(llvm::raw_ostream &s) const {
  s << (state == uninitialized ? "uninitialized"
                               : (state == uniform ? "uniform" : "dependent"));
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
    LDBGS(" dependent op: " << *op);
    return success();
  }

  // Check if it's a uniform op.
  if (maybeDefinitelyWorkgroupUniform(op)) {
    LDBGS(" uniform op: " << *op);
    for (ThreadUniformLattice *v : results)
      propagateIfChanged(v, v->join(ThreadUniform::getUniform()));
    return success();
  }

  LDBGS(" pessimistic dependent op: " << *op);
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
    ArrayRef<mlir::dataflow::AbstractSparseLattice *> latticesRaw) {
  const auto *predecessors =
      getOrCreateFor<mlir::dataflow::PredecessorState>(point, point);
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
} // namespace mlir::iree_compiler::dataflow

MLIR_DEFINE_EXPLICIT_TYPE_ID(
    mlir::iree_compiler::dataflow::ThreadUniformLattice)
