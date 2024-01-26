// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "llvm/ADT/SetOperations.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"

using namespace mlir::dataflow;

namespace mlir::iree_compiler {

namespace {

constexpr const char *kRaWBarrier = "__sync_barrier_raw";
constexpr const char *kWaRBarrier = "__sync_barrier_war";

class SetLattice final : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  virtual ~SetLattice() = default;

  ChangeResult join(const AbstractDenseLattice &rhs) override;

  ChangeResult join(const DenseSet<Attribute> &rhsReads,
                    const DenseSet<Attribute> &rhsWrites);

  void clear();

  void print(raw_ostream &os) const override;

  const DenseSet<Attribute> getPendingReads() const { return pendingReads; }
  const DenseSet<Attribute> getPendingWrites() const { return pendingWrites; }

private:
  DenseSet<Attribute> pendingReads;
  DenseSet<Attribute> pendingWrites;
};

class SynchronizationAnalysis final
    : public dataflow::DenseForwardDataFlowAnalysis<SetLattice> {
public:
  SynchronizationAnalysis(DataFlowSolver &solver)
      : DenseForwardDataFlowAnalysis(solver) {}

  virtual ~SynchronizationAnalysis() = default;

  void visitOperation(Operation *op, const SetLattice &before,
                      SetLattice *after) override;

private:
  std::tuple<DenseSet<Attribute>, DenseSet<Attribute>>
  getReadAndWriteSet(Operation *op);

  void setToEntryState(SetLattice *lattice) override;
};

}; // namespace

ChangeResult SetLattice::join(const AbstractDenseLattice &rhs) {
  auto &rhsSet = static_cast<const SetLattice &>(rhs);
  return join(rhsSet.getPendingReads(), rhsSet.getPendingWrites());
}

ChangeResult SetLattice::join(const DenseSet<Attribute> &rhsReads,
                              const DenseSet<Attribute> &rhsWrites) {
  bool changed = llvm::set_union(pendingReads, rhsReads);
  changed |= llvm::set_union(pendingWrites, rhsWrites);
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

void SetLattice::clear() {
  pendingReads.clear();
  pendingWrites.clear();
}

void SetLattice::print(raw_ostream &os) const {
  os << "pendingReads=[";
  llvm::interleaveComma(pendingReads, os, [&](Attribute attr) { os << attr; });
  os << "], ";
  os << "pendingWrites=[";
  llvm::interleaveComma(pendingWrites, os, [&](Attribute attr) { os << attr; });
  os << "]";
}

std::tuple<DenseSet<Attribute>, DenseSet<Attribute>>
SynchronizationAnalysis::getReadAndWriteSet(Operation *op) {
  DenseSet<Attribute> readSet;
  DenseSet<Attribute> writeSet;

  auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffects) {
    // This has a read/write effect on everything.
    readSet.insert(Attribute());
    writeSet.insert(Attribute());
    return std::make_tuple(readSet, writeSet);
  }

  for (OpOperand &operand : op->getOpOperands()) {
    auto memrefType = dyn_cast<MemRefType>(operand.get().getType());
    if (!memrefType) {
      continue;
    }

    // Check if this has a require sync memory space attribute.
    auto memorySpace = dyn_cast_or_null<IREE::LinalgExt::RequiresSyncAttr>(
        memrefType.getMemorySpace());
    if (!memorySpace) {
      continue;
    }

    // Check if this is a read or write.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    memEffects.getEffectsOnValue(operand.get(), effects);
    for (auto &effect : effects) {
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        readSet.insert(memorySpace.getBarrierToken());
      } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
        writeSet.insert(memorySpace.getBarrierToken());
      }
    }
  }

  return std::make_tuple(readSet, writeSet);
}

void SynchronizationAnalysis::visitOperation(Operation *op,
                                             const SetLattice &before,
                                             SetLattice *after) {
  auto [readSet, writeSet] = getReadAndWriteSet(op);

  DenseSet<Attribute> pendingReads = before.getPendingReads();
  DenseSet<Attribute> pendingWrites = before.getPendingWrites();

  // TODO: We should add a check for barriers that are already present in IR.
  // Currently, we are assuming the IR has no barriers in IR.

  if (!llvm::set_intersection(pendingWrites, readSet).empty()) {
    op->setAttr(kRaWBarrier, UnitAttr::get(op->getContext()));
    // Sync.
    pendingReads.clear();
    pendingWrites.clear();
    // TODO: Does this keep the analysis monotonic? The analysis should still
    // converge, but maybe we need to expand the lattice.
    after->clear();
  } else if (op->hasAttr(kRaWBarrier)) {
    op->removeAttr(kRaWBarrier);
  }

  if (!llvm::set_intersection(pendingReads, writeSet).empty()) {
    op->setAttr(kWaRBarrier, UnitAttr::get(op->getContext()));
    // Sync.
    pendingReads.clear();
    pendingWrites.clear();
    // TODO: Does this keep the analysis monotonic? The analysis should still
    // converge, but maybe we need to expand the lattice.
    after->clear();
  } else if (op->hasAttr(kWaRBarrier)) {
    op->removeAttr(kWaRBarrier);
  }

  llvm::set_union(pendingWrites, writeSet);
  llvm::set_union(pendingReads, readSet);

  // Propagate the change.
  propagateIfChanged(after, after->join(pendingReads, pendingWrites));

  // TODO: Put this in LLVM_DEBUG.
  // llvm::errs() << "\n\n";
  // before.print(llvm::errs());
  // llvm::errs() << "\nReadSet=";
  // interleaveComma(readSet, llvm::errs(),
  //                 [&](Attribute attr) { llvm::errs() << attr; });
  // llvm::errs() << "\nWriteSet=";
  // interleaveComma(writeSet, llvm::errs(),
  //                 [&](Attribute attr) { llvm::errs() << attr; });
  // llvm::errs() << "\n";
  // op->dump();
  // after->print(llvm::errs());
  // llvm::errs() << "\n\n";
}

void SynchronizationAnalysis::setToEntryState(SetLattice *lattice) {
  lattice->clear();
}

LogicalResult synchronizeBuffers(RewriterBase &rewriter, Operation *root,
                                 std::function<void(OpBuilder builder)> sync) {

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<SynchronizationAnalysis>();
  if (solver.initializeAndRun(root).failed()) {
    return failure();
  }

  // Collect all operations with a synchronization attribute.
  SmallVector<Operation *, 4> opRequiresSync;
  root->walk([&](Operation *op) {
    if (op->hasAttr(kRaWBarrier)) {
      opRequiresSync.push_back(op);
      op->removeAttr(kRaWBarrier);
    }

    if (op->hasAttr(kWaRBarrier)) {
      opRequiresSync.push_back(op);
      op->removeAttr(kWaRBarrier);
    }
  });

  // Insert synchronization before ops requiring it.
  for (Operation *op : opRequiresSync) {
    rewriter.setInsertionPoint(op);
    sync(rewriter);
  }

  // Replace #requires_sync<#memory_space, ...> with #memory_space on memref
  // types.
  auto replaceRequiresSync =
      [](Attribute memorySpace) -> std::optional<Attribute> {
    if (auto requiresSync =
            dyn_cast<IREE::LinalgExt::RequiresSyncAttr>(memorySpace)) {
      return requiresSync.getMemorySpace();
    }
    return std::nullopt;
  };

  return replaceMemRefMemorySpace(root, replaceRequiresSync);
}

} // namespace mlir::iree_compiler
