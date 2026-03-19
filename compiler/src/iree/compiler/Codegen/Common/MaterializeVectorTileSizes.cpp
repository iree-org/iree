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

#define DEBUG_TYPE "iree-codegen-vector-tile-size-analysis"

// The purpose of this analysis is to propagate information about the
// vector tile size across the operation graph. The vector tile size is
// important information for the vectorization of operations. For example, the
// vector tile size can be used by GenericVectorization to introduce the
// necessary masking in the presence of padding/masking.
//
// The analysis is a bi-directional dataflow analysis building on top of the
// upstream MLIR dataflow analysis framework. To implement the bi-directional
// propagation, it combines a sparse forward analysis and a sparse backward
// analysis in the same solver.
//
// The lattice for the dataflow analysis is shared by both analyses (forward and
// backward). For each N-dimensional ShapedType SSA value, we have a lattice
// element comprising N dimensions, where each dimension is in one of three
// states: uninitialized (bottom), a single tile size value, or overdefined
// (top). The bottom (uninitialized) state is the identity for the merge
// operation. Merging two equal tile sizes for the same dimension is identity.
// Merging two different tile sizes for the same dimension results in the
// overdefined state. Overdefined is absorbing — once a dimension reaches
// overdefined, it stays overdefined.
//
// The lattice is initialized from anchor operations that provide information
// about vector tile size (e.g., `to_layout`).
//
// Forward propagation and backward propagation work similarly:
// - For linalg operations, all available information from operands (forward) or
//   results & operands (backward) is mapped to the iteration space based on
//   indexing maps and merged into a single lattice state. That lattice state in
//   the iteration space is then mapped to each result (forward) or operand
//   (backward) based on indexing maps and the mapped state is propagated.
//
// Duplicatable operations such as `tensor.empty`, constants, and generator
// linalg ops (e.g. linalg.fill) are excluded from propagation entirely. CSE
// connects otherwise unrelated compute ops by deduplicating their DPS init
// operands to a single tensor.empty (or similar). To avoid cross-polluting the
// vector tile size of unrelated operations, tile sizes from duplicatable
// operations are never propagated.
//
// For some other operations, no propagation rules are defined on purpose. For
// example, `extract_slice` and `insert_slice` operations are natural boundaries
// of tiling/padding, therefore no information is propagated across them.
//
// After the dataflow solver reaches a fixpoint, the
// MaterializeVectorTileSizesPass materializes the result as a discardable
// attribute. Only dimensions with a single defined (non-overdefined) tile size
// are materialized. Operations where any dimension is uninitialized or
// overdefined do not receive the attribute.

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

/// Per-dimension tile sizes. Each dimension holds a single tile size value, or
/// one of the sentinel values kUninitialized/kOverdefined. Satisfies the
/// requirements for use as a `Lattice<ValueT>` value type.
class TileSizes {
public:
  TileSizes() = default;
  explicit TileSizes(unsigned rank) : dims(rank, kUninitialized) {}

  /// Construct from concrete tile sizes (one value per dimension).
  TileSizes(ArrayRef<int64_t> sizes) : dims(sizes) {}

  unsigned rank() const { return dims.size(); }
  bool empty() const { return dims.empty(); }
  const llvm::SmallVector<int64_t> &getDims() const { return dims; }

  int64_t operator[](unsigned i) const { return dims[i]; }

  /// Returns true if all dimensions have a defined (positive) tile size.
  bool isDefined() const {
    return !empty() && llvm::all_of(dims, [](int64_t v) {
      return v != kUninitialized && v != kOverdefined;
    });
  }

  /// Returns true if any dimension is overdefined.
  bool isOverdefined() const {
    return llvm::any_of(dims, [](int64_t v) { return v == kOverdefined; });
  }

  /// Merge tile sizes from `other` into this. Uninitialized is identity.
  void merge(const TileSizes &other) {
    if (empty()) {
      *this = other;
      return;
    }
    if (other.empty()) {
      return;
    }
    assert(rank() == other.rank() && "rank mismatch");
    for (unsigned i = 0; i < rank(); ++i) {
      dims[i] = mergeDim(dims[i], other.dims[i]);
    }
  }

  /// Map from operand space to iteration space via an indexing map.
  TileSizes mapToIterationSpace(AffineMap indexingMap) const {
    TileSizes result(indexingMap.getNumDims());
    for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        continue;
      }
      unsigned iterDim = dimExpr.getPosition();
      result.dims[iterDim] = mergeDim(result.dims[iterDim], dims[i]);
    }
    return result;
  }

  /// Map from iteration space to operand space via an indexing map.
  /// Returns empty TileSizes if any operand dim can't be determined.
  TileSizes mapFromIterationSpace(AffineMap indexingMap) const {
    unsigned numResults = indexingMap.getNumResults();
    TileSizes result(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
      if (!dimExpr) {
        return {};
      }
      unsigned iterDim = dimExpr.getPosition();
      if (iterDim >= rank() || dims[iterDim] == kUninitialized) {
        return {};
      }
      result.dims[i] = dims[iterDim];
    }
    return result;
  }

  /// Lattice join: per-dimension merge. Uninitialized is identity.
  static TileSizes join(const TileSizes &lhs, const TileSizes &rhs) {
    TileSizes result = lhs;
    result.merge(rhs);
    return result;
  }

  /// Lattice meet: same as join (both directions merge the same way).
  static TileSizes meet(const TileSizes &lhs, const TileSizes &rhs) {
    return join(lhs, rhs);
  }

  bool operator==(const TileSizes &rhs) const { return dims == rhs.dims; }

  void print(raw_ostream &os) const {
    os << "[";
    llvm::interleaveComma(dims, os, [&](int64_t v) {
      if (v == kUninitialized) {
        os << "?";
      } else if (v == kOverdefined) {
        os << "<overdefined>";
      } else {
        os << v;
      }
    });
    os << "]";
  }

private:
  SmallVector<int64_t> dims;

  /// Sentinel values for per-dimension tile size state. Valid tile sizes are
  /// always positive.
  static constexpr int64_t kUninitialized = 0;
  static constexpr int64_t kOverdefined = -1;

  /// Merge a single dimension's tile size value. Returns the merged result.
  static int64_t mergeDim(int64_t a, int64_t b) {
    if (a == kOverdefined || b == kOverdefined) {
      return kOverdefined;
    }
    if (a == kUninitialized) {
      return b;
    }
    if (b == kUninitialized) {
      return a;
    }
    return a == b ? a : kOverdefined;
  }
};

/// Returns true if the operation is trivially duplicatable and should not
/// propagate tile sizes across independent consumers.
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
  // A linalg op that doesn't read any tensor data (e.g., linalg.fill or a
  // fill-like linalg.generic broadcasting a scalar) is a generator and
  // duplicatable.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(defOp)) {
    if (llvm::none_of(linalgOp->getOpOperands(), [&](OpOperand &operand) {
          return isa<ShapedType>(operand.get().getType()) &&
                 linalgOp.payloadUsesValueFromOperand(&operand);
        })) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Lattice and analysis definitions
//===----------------------------------------------------------------------===//

class TileSizeLattice : public dataflow::Lattice<TileSizes> {
public:
  using Lattice::Lattice;
};

/// Read the TileSizes from a lattice, returning empty tile sizes if the lattice
/// value is from a duplicatable operation.
static const TileSizes getTileSizesFor(Value val,
                                       const TileSizeLattice *lattice) {
  if (!lattice) {
    return {};
  }
  auto &tileSizes = lattice->getValue();
  if (tileSizes.empty()) {
    return {};
  }
  if (isDuplicatable(val)) {
    return {};
  }
  return tileSizes;
}

/// Forward analysis: propagates tile sizes from operands to results.
/// Control flow through scf.for/scf.if is handled automatically by the
/// framework via RegionBranchOpInterface.
class TileSizeForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TileSizeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(TileSizeLattice *lattice) override {
    // Entry state is uninitialized (identity for join).
    propagateIfChanged(lattice, lattice->join(TileSizes()));
  }

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const TileSizeLattice *> operands,
                               ArrayRef<TileSizeLattice *> results) override {
    // to_layout: seed from layout, don't propagate operand forward.
    if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
      LDBG() << "Anchor: " << toLayout;
      TileSizes tileSizes(toLayout.getLayout().getUndistributedShape());
      propagateIfChanged(results[0], results[0]->join(tileSizes));
      return success();
    }

    // Linalg ops: propagate through indexing maps.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      unsigned numLoops = linalgOp.getNumLoops();
      TileSizes iterTileSizes(numLoops);
      for (OpOperand &operand : linalgOp->getOpOperands()) {
        auto &ts = getTileSizesFor(operand.get(),
                                   operands[operand.getOperandNumber()]);
        if (ts.empty()) {
          continue;
        }
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(ts.mapToIterationSpace(map));
      }
      for (unsigned i = 0; i < linalgOp.getNumDpsInits(); ++i) {
        OpOperand *init = linalgOp.getDpsInitOperand(i);
        AffineMap map = linalgOp.getMatchingIndexingMap(init);
        auto resultTileSizes = iterTileSizes.mapFromIterationSpace(map);
        if (!resultTileSizes.empty()) {
          propagateIfChanged(results[i], results[i]->join(resultTileSizes));
        }
      }
      return success();
    }

    return success();
  }
};

/// Backward analysis: propagates tile sizes from results to operands.
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
    // to_layout is always an anchor op; Propagate tile sizes backward to the
    // input.
    if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
      auto &ts = getTileSizesFor(toLayout.getResult(), results[0]);
      if (!ts.empty()) {
        TileSizeLattice *inputLattice = operands[0];
        propagateIfChanged(inputLattice, inputLattice->meet(ts));
      }
      return success();
    }

    // Linalg ops: propagate through indexing maps.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      unsigned numLoops = linalgOp.getNumLoops();
      TileSizes iterTileSizes(numLoops);
      // Gather result tile sizes into iteration space via DPS init maps.
      for (auto [result, resultLattice] :
           llvm::zip(linalgOp.getOperation()->getResults(), results)) {
        auto &ts = getTileSizesFor(result, resultLattice);
        if (ts.empty()) {
          continue;
        }
        unsigned resultIdx = cast<OpResult>(result).getResultNumber();
        OpOperand *init = linalgOp.getDpsInitOperand(resultIdx);
        AffineMap map = linalgOp.getMatchingIndexingMap(init);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(ts.mapToIterationSpace(map));
      }
      // Gather operand tile sizes into iteration space.
      for (OpOperand &operand : linalgOp->getOpOperands()) {
        auto &ts = getTileSizesFor(operand.get(),
                                   operands[operand.getOperandNumber()]);
        if (ts.empty()) {
          continue;
        }
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(ts.mapToIterationSpace(map));
      }
      // Map iteration space tile sizes back to each operand.
      for (OpOperand &operand : linalgOp->getOpOperands()) {
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        auto operandTileSizes = iterTileSizes.mapFromIterationSpace(map);
        if (operandTileSizes.empty()) {
          continue;
        }
        TileSizeLattice *operandLattice = operands[operand.getOperandNumber()];
        propagateIfChanged(operandLattice,
                           operandLattice->meet(operandTileSizes));
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

/// Gather tile sizes into the iteration space of a linalg op by looking up each
/// operand's lattice state in the solver.
static TileSizes getIterationSpaceTileSizes(linalg::LinalgOp linalgOp,
                                            const DataFlowSolver &solver) {
  unsigned numLoops = linalgOp.getNumLoops();
  TileSizes iterTileSizes(numLoops);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    Value val = operand.get();
    auto *lattice = solver.lookupState<TileSizeLattice>(val);
    auto &ts = getTileSizesFor(val, lattice);
    if (ts.empty()) {
      continue;
    }
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    assert(map.getNumDims() == numLoops);
    iterTileSizes.merge(ts.mapToIterationSpace(map));
  }
  return iterTileSizes;
}

/// Given a linalg op and the solver, compute per-dimension tile sizes.
/// Returns a vector of one tile size per iteration dimension, or nullopt if
/// any dimension is uninitialized or overdefined.
static std::optional<SmallVector<int64_t>>
getPerDimTileSizes(linalg::LinalgOp linalgOp, const DataFlowSolver &solver) {
  TileSizes tileSizes = getIterationSpaceTileSizes(linalgOp, solver);
  if (!tileSizes.isDefined()) {
    return std::nullopt;
  }
  assert(tileSizes.rank() == linalgOp.getNumLoops());
  return tileSizes.getDims();
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
      std::optional<SmallVector<int64_t>> perDimSizes =
          getPerDimTileSizes(linalgOp, solver);
      if (!perDimSizes) {
        LDBG() << "Analysis did not determine tile size for" << *linalgOp;
        return;
      }

      LDBG() << "Materializing tile size on " << *linalgOp;
      linalgOp->setAttr(
          kVectorTileSizesAttrName,
          DenseI64ArrayAttr::get(linalgOp->getContext(), *perDimSizes));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
