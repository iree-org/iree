// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "iree-codegen-vector-tile-size-analysis"

// The purpose of this analysis is to propagate information about the
// undistributed vector tile size across the operation graph. The vector tile
// size is important information for the vectorization of operations.
// For example, the vector tile size can be used by GenericVectorization to
// introduce the necessary masking in the presence of padding/masking.
//
// The analysis is a bi-directional dataflow analysis building on top of the
// upstream MLIR dataflow analysis framework. To implement the bi-directional
// propagation, it combines a sparse forward analysis and a sparse backward
// analysis in the same solver.
//
// The lattice for the dataflow analysis is shared by both analyses (forward and
// backward). For each N-dimensional ShapedType SSA value, we have a lattice
// element comprising N sets, where each set contains the candidate tile sizes
// for that dimension. The bottom (uninitialized) state of the lattice is simply
// empty. The join/merge operation for two lattice elements is the per-dimension
// set-union of candidates. For example, in the 2D case:
// ({2, 4}, {16}) U ({8}, {32}) = ({2, 4, 8}, {16, 32})
//
// As the sets can only grow, the join/meet operator is by definition monotonic.
// As the set union can not result in a conflict, no lattice state for top
// (overdefined) is required in this lattice.
//
// The lattice is initialized from `to_layout` operations.
//
// Forward propagation and backward propagation work similarly:
// - For elementwise operations, candidates from the different operands
//   (forward) or results (backwards) are merged. The merged lattice state is
//   then propagated to all results (forward) or operands (backward).
// - For linalg.generic operations, all available information from operands
//   (forward) or results & operands (backward) is mapped to the iteration space
//   based on indexing maps and merged into a single lattice state. That lattice
//   state in the iteration space is then mapped to each result (forward) or
//   operand (backward) based on indexing maps and the mapped state is
//   propagated.
//
// The only exception to this process are duplicatable operations such as
// `tensor.empty`. CSE connects otherwise unrelated compute ops by deduplicating
// their DPS init operands to a single tensor.empty (or similar). To avoid
// cross-polluting the vector tile size of unrelated operations, propagation
// from duplicatable operations is stopped if they contain multiple candidates
// tile sizes in at least one dimension.
//
// For some other operations, no propagation rules are defined on purpose. For
// example, `extract_slice` and `insert_slice` operations are natural boundaries
// of tiling/padding, therefore no information is propagated across them.
//
// After the dataflow solver reaches a fixpoint, the
// MaterializeVectorTileSizesPass materializes the result as a discardable
// attribute. At this point, the result is a set of candidate vector tile sizes
// per iteration dimension. It is up to the users of the analysis how to select
// a tile size from the set of candidates.

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

using TileSizeSet = llvm::SmallSet<int64_t, 2>;

/// Per-dimension tile size candidates. Each dimension has an independent set
/// of candidate tile sizes. Satisfies the requirements for use as a
/// `Lattice<ValueT>` value type.
class TileSizeCandidates {
public:
  TileSizeCandidates() = default;
  explicit TileSizeCandidates(unsigned rank) : dims(rank) {}

  /// Construct from a single concrete tile size (one value per dimension).
  static TileSizeCandidates fromSizes(ArrayRef<int64_t> sizes) {
    TileSizeCandidates result(sizes.size());
    for (unsigned i = 0; i < sizes.size(); ++i) {
      result.dims[i].insert(sizes[i]);
    }
    return result;
  }

  unsigned rank() const { return dims.size(); }
  bool empty() const { return dims.empty(); }

  const TileSizeSet &operator[](unsigned i) const { return dims[i]; }
  TileSizeSet &operator[](unsigned i) { return dims[i]; }

  /// Merge candidates from `other` into this. Uninitialized is identity.
  void merge(const TileSizeCandidates &other) {
    if (empty()) {
      *this = other;
      return;
    }
    if (other.empty()) {
      return;
    }
    assert(rank() == other.rank() && "rank mismatch");
    for (unsigned i = 0; i < rank(); ++i) {
      dims[i].insert_range(other.dims[i]);
    }
  }

  /// Returns true if any dimension has more than one candidate.
  bool hasAlternatives() const {
    return llvm::any_of(dims,
                        [](const TileSizeSet &s) { return s.size() > 1; });
  }

  /// Map from operand space to iteration space via an indexing map.
  TileSizeCandidates mapToIterationSpace(AffineMap indexingMap,
                                         unsigned numLoops) const {
    TileSizeCandidates result(numLoops);
    for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        continue;
      }
      unsigned iterDim = dimExpr.getPosition();
      for (int64_t v : dims[i]) {
        result.dims[iterDim].insert(v);
      }
    }
    return result;
  }

  /// Map from iteration space to operand space via an indexing map.
  /// Returns empty TileSizeCandidates if any operand dim can't be determined.
  TileSizeCandidates mapFromIterationSpace(AffineMap indexingMap) const {
    unsigned numResults = indexingMap.getNumResults();
    TileSizeCandidates result(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        return {};
      }
      unsigned iterDim = dimExpr.getPosition();
      if (iterDim >= rank() || dims[iterDim].empty()) {
        return {};
      }
      result.dims[i] = dims[iterDim];
    }
    return result;
  }

  /// Lattice join: per-dimension set union. Uninitialized is identity.
  static TileSizeCandidates join(const TileSizeCandidates &lhs,
                                 const TileSizeCandidates &rhs) {
    TileSizeCandidates result = lhs;
    result.merge(rhs);
    return result;
  }

  /// Lattice meet: same as join (both directions accumulate via set union).
  static TileSizeCandidates meet(const TileSizeCandidates &lhs,
                                 const TileSizeCandidates &rhs) {
    return join(lhs, rhs);
  }

  bool operator==(const TileSizeCandidates &rhs) const {
    return dims == rhs.dims;
  }

  void print(raw_ostream &os) const {
    os << "[";
    llvm::interleaveComma(dims, os, [&](const TileSizeSet &s) {
      os << "{";
      llvm::interleaveComma(s, os);
      os << "}";
    });
    os << "]";
  }

private:
  SmallVector<TileSizeSet> dims;
};

/// Returns true if the operation is trivially duplicatable and should not
/// propagate merged tile sizes across independent consumers.
static bool isDuplicatable(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (isa<tensor::EmptyOp>(defOp)) {
    return true;
  }
  if (defOp->hasTrait<OpTrait::ConstantLike>()) {
    return true;
  }
  // Catches linalg.fill that has been lowered/fused into linalg.generic form
  // (scalar input broadcast into tensor.empty output).
  if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
    if (genericOp.getNumDpsInputs() == 1 && genericOp.getNumDpsInits() == 1 &&
        !isa<ShapedType>(genericOp.getDpsInputs()[0].getType())) {
      Value init = genericOp.getDpsInits()[0];
      if (init.getDefiningOp<tensor::EmptyOp>()) {
        return true;
      }
    }
  }
  if (auto fillOp = dyn_cast<linalg::FillOp>(defOp)) {
    if (fillOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Lattice and analysis definitions
//===----------------------------------------------------------------------===//

class TileSizeLattice : public dataflow::Lattice<TileSizeCandidates> {
public:
  using Lattice::Lattice;
};

/// Read the TileSizeCandidates from a lattice, returning empty candidates
/// if the lattice value is duplicatable with alternatives.
static const TileSizeCandidates &
getCandidatesFor(Value val, const TileSizeLattice *lattice) {
  static const TileSizeCandidates empty;
  if (!lattice) {
    return empty;
  }
  auto &candidates = lattice->getValue();
  if (candidates.empty()) {
    return empty;
  }
  if (isDuplicatable(val) && candidates.hasAlternatives()) {
    return empty;
  }
  return candidates;
}

/// Forward analysis: propagates tile size candidates from operands to results.
/// Control flow through scf.for/scf.if is handled automatically by the
/// framework via RegionBranchOpInterface.
class TileSizeForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TileSizeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    // Seed to_layout anchors before the regular initialization. This ensures
    // seeds are set even for to_layout ops in regions that DeadCodeAnalysis
    // hasn't yet marked as live during init.
    top->walk([&](ToLayoutOp toLayout) {
      LDBG() << "Anchor: " << toLayout;
      auto candidates = TileSizeCandidates::fromSizes(
          toLayout.getLayout().getUndistributedShape());
      auto *lattice = getLatticeElement(toLayout.getResult());
      propagateIfChanged(lattice, lattice->join(candidates));
    });
    return SparseForwardDataFlowAnalysis::initialize(top);
  }

  void setToEntryState(TileSizeLattice *lattice) override {
    // Entry state is uninitialized (identity for join).
    propagateIfChanged(lattice, lattice->join(TileSizeCandidates()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const TileSizeLattice *> operands,
                               ArrayRef<TileSizeLattice *> results) override {
    // to_layout: don't propagate operand forward (anchor boundary).
    // Seeding is done in initialize().
    if (isa<ToLayoutOp>(op)) {
      return success();
    }

    // linalg.generic: propagate through indexing maps.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      unsigned numLoops = genericOp.getNumLoops();
      // Combine the information from all operands into a single candidate in
      // iteration space.
      TileSizeCandidates iterCandidates(numLoops);
      for (OpOperand &operand : genericOp->getOpOperands()) {
        auto &candidates = getCandidatesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        if (candidates.empty()) {
          continue;
        }
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      // For each result, map the combined candidate in iteration space back to
      // the result (DPS init operand) space.
      for (unsigned i = 0; i < genericOp.getNumDpsInits(); ++i) {
        OpOperand *init = genericOp.getDpsInitOperand(i);
        AffineMap map = genericOp.getMatchingIndexingMap(init);
        auto resultCandidates = iterCandidates.mapFromIterationSpace(map);
        if (!resultCandidates.empty()) {
          propagateIfChanged(results[i], results[i]->join(resultCandidates));
        }
      }
      return success();
    }

    // Elementwise ops: propagate to all results.
    if (OpTrait::hasElementwiseMappableTraits(op)) {
      TileSizeCandidates combined;
      for (auto [operandLattice, operandVal] :
           llvm::zip(operands, op->getOperands())) {
        combined.merge(getCandidatesFor(operandVal, operandLattice));
      }
      for (TileSizeLattice *result : results) {
        propagateIfChanged(result, result->join(combined));
      }
      return success();
    }

    return success();
  }
};

/// Backward analysis: propagates tile size candidates from results to operands.
/// Control flow through scf.for/scf.if is handled automatically by the
/// framework via RegionBranchOpInterface.
class TileSizeBackwardAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<TileSizeLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(TileSizeLattice *lattice) override {
    // Exit state is uninitialized (identity for meet).
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TileSizeLattice *> operands,
                 ArrayRef<const TileSizeLattice *> results) override {
    // to_layout: propagate result tile sizes backward to input.
    if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
      auto &candidates = getCandidatesFor(toLayout.getResult(), results[0]);
      if (!candidates.empty()) {
        TileSizeLattice *inputLattice = operands[0];
        propagateIfChanged(inputLattice, inputLattice->meet(candidates));
      }
      return success();
    }

    // linalg.generic: propagate through indexing maps.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      unsigned numLoops = genericOp.getNumLoops();
      TileSizeCandidates iterCandidates(numLoops);
      // Gather result candidates into iteration space via DPS init maps.
      for (auto [result, resultLattice] :
           llvm::zip(genericOp.getResults(), results)) {
        auto &candidates = getCandidatesFor(result, resultLattice);
        if (candidates.empty()) {
          continue;
        }
        unsigned resultIdx = cast<OpResult>(result).getResultNumber();
        OpOperand *init = genericOp.getDpsInitOperand(resultIdx);
        AffineMap map = genericOp.getMatchingIndexingMap(init);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      // Gather operand candidates into iteration space.
      for (OpOperand &operand : genericOp->getOpOperands()) {
        auto &candidates = getCandidatesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        if (candidates.empty()) {
          continue;
        }
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
      }
      // Map iteration space candidates back to each operand.
      for (OpOperand &operand : genericOp->getOpOperands()) {
        AffineMap map = genericOp.getMatchingIndexingMap(&operand);
        auto operandCandidates = iterCandidates.mapFromIterationSpace(map);
        if (operandCandidates.empty()) {
          continue;
        }
        TileSizeLattice *operandLattice = operands[operand.getOperandNumber()];
        propagateIfChanged(operandLattice,
                           operandLattice->meet(operandCandidates));
      }
      return success();
    }

    // Elementwise ops: propagate to all operands.
    if (OpTrait::hasElementwiseMappableTraits(op)) {
      TileSizeCandidates combined;
      for (auto [resultVal, resultLattice] :
           llvm::zip(op->getResults(), results)) {
        combined.merge(getCandidatesFor(resultVal, resultLattice));
      }
      for (auto [operandLattice, operandVal] :
           llvm::zip(operands, op->getOperands())) {
        if (!isa<ShapedType>(operandVal.getType())) {
          continue;
        }
        propagateIfChanged(operandLattice, operandLattice->meet(combined));
      }
      return success();
    }

    return success();
  }

  // Required by the base class. Non-forwarded branch operands (e.g., loop
  // bounds, conditions) are scalars irrelevant to tile size propagation.
  // Forwarded values (iter_args, yields) are handled by the framework via
  // RegionBranchOpInterface.
  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}
  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
};

//===----------------------------------------------------------------------===//
// Result querying
//===----------------------------------------------------------------------===//

/// Gather tile size candidates into the iteration space of a linalg op by
/// looking up each operand's lattice state in the solver.
static TileSizeCandidates
getIterationSpaceTileSizes(linalg::LinalgOp linalgOp,
                           const DataFlowSolver &solver) {
  unsigned numLoops = linalgOp.getNumLoops();
  TileSizeCandidates iterCandidates(numLoops);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    Value val = operand.get();
    auto *lattice = solver.lookupState<TileSizeLattice>(val);
    auto &candidates = getCandidatesFor(val, lattice);
    if (candidates.empty()) {
      continue;
    }
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    iterCandidates.merge(candidates.mapToIterationSpace(map, numLoops));
  }
  return iterCandidates;
}

/// Given a linalg op and the solver, compute per-dimension sets of
/// candidate tile sizes. Returns a vector of size numLoops, where each entry
/// is the deduplicated set of tile sizes for that iteration dimension.
/// Returns an empty vector if any dimension has no candidates.
static SmallVector<SmallVector<int64_t>>
getPerDimTileSizes(linalg::LinalgOp linalgOp, const DataFlowSolver &solver) {
  auto perDimSizes = getIterationSpaceTileSizes(linalgOp, solver);

  // Return empty if any dimension has no candidates.
  unsigned numLoops = linalgOp.getNumLoops();
  SmallVector<SmallVector<int64_t>> results;
  for (unsigned i = 0; i < numLoops; ++i) {
    if (perDimSizes[i].empty()) {
      return {};
    }
    results.push_back(
        SmallVector<int64_t>(perDimSizes[i].begin(), perDimSizes[i].end()));
  }
  return results;
}

//===----------------------------------------------------------------------===//
// MaterializeVectorTileSizesPass
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_MATERIALIZEVECTORTILESIZESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class MaterializeVectorTileSizesPass final
    : public impl::MaterializeVectorTileSizesPassBase<
          MaterializeVectorTileSizesPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<TileSizeForwardAnalysis>();
    SymbolTableCollection symbolTable;
    solver.load<TileSizeBackwardAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      auto perDimSizes = getPerDimTileSizes(linalgOp, solver);
      if (perDimSizes.empty()) {
        return;
      }

      LDBG() << "Materializing tile size on " << *linalgOp;

      SmallVector<Attribute> dimAttrs;
      for (const auto &dimSizes : perDimSizes) {
        dimAttrs.push_back(
            DenseI64ArrayAttr::get(linalgOp->getContext(), dimSizes));
      }
      linalgOp->setAttr(kVectorTileSizesAttrName,
                        ArrayAttr::get(linalgOp->getContext(), dimAttrs));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
