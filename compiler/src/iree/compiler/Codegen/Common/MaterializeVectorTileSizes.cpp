// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/Im2colUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler {
#define GEN_PASS_DEF_MATERIALIZEVECTORTILESIZESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace mlir::iree_compiler

#define DEBUG_TYPE "iree-codegen-materialize-vector-tile-sizes"

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
  ArrayRef<int64_t> getDims() const { return dims; }

  int64_t operator[](unsigned i) const { return dims[i]; }

  /// Returns true if the tile sizes are non-empty and every dimension has a
  /// concrete tile size (not uninitialized or overdefined).
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
    if (empty()) {
      // Early return in case this candidate is empty.
      return result;
    }
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

  /// Map tile sizes from pack source space (rank N) to pack dest space
  /// (rank N+K). Divides packed dims by inner tile sizes, applies
  /// outer_dims_perm, and appends inner tile sizes as new dimensions.
  /// The static inner tiles are assumed to not be dynamic values.
  TileSizes mapPackSourceToDest(ArrayRef<int64_t> innerDimsPos,
                                ArrayRef<int64_t> staticInnerTiles,
                                ArrayRef<int64_t> outerDimsPerm) const {
    if (empty()) {
      return {};
    }
    TileSizes result = *this;
    for (auto [dimPos, tileSize] :
         llvm::zip_equal(innerDimsPos, staticInnerTiles)) {
      assert(ShapedType::isStatic(tileSize) &&
             "expecting static inner tile size");
      if (result.dims[dimPos] == kUninitialized ||
          result.dims[dimPos] == kOverdefined) {
        continue;
      }
      result.dims[dimPos] =
          llvm::divideCeilSigned(result.dims[dimPos], tileSize);
    }
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector(result.dims, outerDimsPerm);
    }
    llvm::append_range(result.dims, staticInnerTiles);
    return result;
  }

  /// Map tile sizes from pack dest space (rank N+K) back to source space
  /// (rank N). Truncates to outer dims, applies inverse permutation, and
  /// multiplies packed dims by inner tile sizes.
  TileSizes mapPackDestToSource(unsigned sourceRank,
                                ArrayRef<int64_t> innerDimsPos,
                                ArrayRef<int64_t> staticInnerTiles,
                                ArrayRef<int64_t> outerDimsPerm) const {
    if (empty()) {
      return {};
    }
    assert(sourceRank <= rank() && "sourceRank exceeds dest tile size rank");
    TileSizes result(sourceRank);
    for (unsigned i = 0; i < sourceRank; ++i) {
      result.dims[i] = dims[i];
    }
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector(result.dims,
                               invertPermutationVector(outerDimsPerm));
    }
    for (auto [dimPos, tileSize] :
         llvm::zip_equal(innerDimsPos, staticInnerTiles)) {
      if (result.dims[dimPos] == kUninitialized ||
          result.dims[dimPos] == kOverdefined) {
        // Uninitialized and overdefined are preserved.
        continue;
      }
      result.dims[dimPos] *= tileSize;
    }
    return result;
  }

  /// Append dimensions from `suffix` to produce a higher-rank TileSizes.
  TileSizes append(ArrayRef<int64_t> suffix) const {
    SmallVector<int64_t> fullDims(dims);
    fullDims.append(suffix.begin(), suffix.end());
    return TileSizes(fullDims);
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
static TileSizes getTileSizesFor(Value val, const TileSizeLattice *lattice) {
  if (!lattice) {
    return {};
  }
  const TileSizes &tileSizes = lattice->getValue();
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
        TileSizes tileSizes = getTileSizesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(tileSizes.mapToIterationSpace(map));
      }
      for (unsigned i = 0; i < linalgOp.getNumDpsInits(); ++i) {
        OpOperand *init = linalgOp.getDpsInitOperand(i);
        AffineMap map = linalgOp.getMatchingIndexingMap(init);
        TileSizes resultTileSizes = iterTileSizes.mapFromIterationSpace(map);
        propagateIfChanged(results[i], results[i]->join(resultTileSizes));
      }
      return success();
    }

    // InnerTiledOp: Propagate the outer dimensions through the indexing maps
    // and then append the static inner dimensions.
    if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
      SmallVector<AffineMap> indexingMaps = innerTiledOp.getIndexingMapsArray();
      unsigned numLoops = indexingMaps[0].getNumDims();
      TileSizes iterTileSizes(numLoops);

      // Merge tile sizes from all operands into iteration space.
      // mapToIterationSpace reads only the first getNumResults() elements
      // from the operand TileSizes, naturally skipping inner dims.
      for (OpOperand &operand : op->getOpOperands()) {
        TileSizes opTileSizes = getTileSizesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        AffineMap map = indexingMaps[operand.getOperandNumber()];
        iterTileSizes.merge(opTileSizes.mapToIterationSpace(map));
      }

      // Propagate to results. Results correspond to outputs.
      unsigned numInputs = innerTiledOp.getNumInputs();
      for (unsigned i = 0; i < innerTiledOp.getNumOutputs(); ++i) {
        AffineMap map = indexingMaps[numInputs + i];
        TileSizes outerTileSizes = iterTileSizes.mapFromIterationSpace(map);
        if (outerTileSizes.empty()) {
          continue;
        }
        // Extend with static inner dims to match full operand rank.
        ArrayRef<int64_t> innerShape =
            innerTiledOp.getOperandInnerShape(numInputs + i);
        TileSizes fullTileSizes = outerTileSizes.append(innerShape);
        propagateIfChanged(results[i], results[i]->join(fullTileSizes));
      }
      return success();
    }

    // Im2col ops: compute tile sizes locally and propagate to result.
    // Im2col's output dimensions are the iteration domain (identity map),
    // so tile sizes go directly on the result lattice.
    if (auto im2colOp = dyn_cast<IREE::LinalgExt::Im2colOp>(op)) {
      OpBuilder builder(op);
      std::optional<SmallVector<int64_t>> maybeTileSizes =
          IREE::LinalgExt::computeIm2colVectorTileSizes(builder, im2colOp);
      if (maybeTileSizes) {
        TileSizes tileSizes(*maybeTileSizes);
        propagateIfChanged(results[0], results[0]->join(tileSizes));
      }
      return success();
    }

    // Pack ops: map source tile sizes to dest space.
    if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
      if (ShapedType::isDynamicShape(packOp.getStaticInnerTiles())) {
        return success();
      }
      TileSizes srcTileSizes = getTileSizesFor(packOp.getSource(), operands[0]);
      TileSizes destTileSizes = srcTileSizes.mapPackSourceToDest(
          packOp.getInnerDimsPos(), packOp.getStaticInnerTiles(),
          packOp.getOuterDimsPerm());
      propagateIfChanged(results[0], results[0]->join(destTileSizes));
      return success();
    }

    // Unpack ops: map source (packed) tile sizes to dest (unpacked) space.
    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
      if (ShapedType::isDynamicShape(unpackOp.getStaticInnerTiles())) {
        return success();
      }
      TileSizes srcTileSizes =
          getTileSizesFor(unpackOp.getSource(), operands[0]);
      TileSizes destTileSizes = srcTileSizes.mapPackDestToSource(
          unpackOp.getDestRank(), unpackOp.getInnerDimsPos(),
          unpackOp.getStaticInnerTiles(), unpackOp.getOuterDimsPerm());
      propagateIfChanged(results[0], results[0]->join(destTileSizes));
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
      TileSizes tileSizes = getTileSizesFor(toLayout.getResult(), results[0]);
      TileSizeLattice *inputLattice = operands[0];
      propagateIfChanged(inputLattice, inputLattice->meet(tileSizes));
      return success();
    }

    // Linalg ops: propagate through indexing maps.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      unsigned numLoops = linalgOp.getNumLoops();
      TileSizes iterTileSizes(numLoops);
      // Gather result tile sizes into iteration space via init maps.
      for (auto [result, resultLattice] :
           llvm::zip_equal(linalgOp->getResults(), results)) {
        TileSizes tileSizes = getTileSizesFor(result, resultLattice);
        unsigned resultIdx = cast<OpResult>(result).getResultNumber();
        OpOperand *init = linalgOp.getDpsInitOperand(resultIdx);
        AffineMap map = linalgOp.getMatchingIndexingMap(init);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(tileSizes.mapToIterationSpace(map));
      }
      // Gather operand tile sizes into iteration space.
      for (OpOperand &operand : linalgOp->getOpOperands()) {
        TileSizes tileSizes = getTileSizesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        assert(map.getNumDims() == numLoops);
        iterTileSizes.merge(tileSizes.mapToIterationSpace(map));
      }
      // Map iteration space tile sizes back to each operand.
      for (OpOperand &operand : linalgOp->getOpOperands()) {
        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        TileSizes operandTileSizes = iterTileSizes.mapFromIterationSpace(map);
        TileSizeLattice *operandLattice = operands[operand.getOperandNumber()];
        propagateIfChanged(operandLattice,
                           operandLattice->meet(operandTileSizes));
      }
      return success();
    }

    // InnerTiledOp: propagate backward through indexing maps.
    if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
      SmallVector<AffineMap> indexingMaps = innerTiledOp.getIndexingMapsArray();
      unsigned numLoops = indexingMaps[0].getNumDims();
      unsigned numInputs = innerTiledOp.getNumInputs();
      TileSizes iterTileSizes(numLoops);

      // Gather from results (full-rank lattice, mapToIterationSpace skips
      // inner dims naturally).
      for (auto [result, resultLattice] :
           llvm::zip_equal(op->getResults(), results)) {
        unsigned resultIdx = cast<OpResult>(result).getResultNumber();
        AffineMap map = indexingMaps[numInputs + resultIdx];
        TileSizes tileSizes = getTileSizesFor(result, resultLattice);
        iterTileSizes.merge(tileSizes.mapToIterationSpace(map));
      }

      // Gather from operands.
      for (OpOperand &operand : op->getOpOperands()) {
        TileSizes opTileSizes = getTileSizesFor(
            operand.get(), operands[operand.getOperandNumber()]);
        AffineMap map = indexingMaps[operand.getOperandNumber()];
        iterTileSizes.merge(opTileSizes.mapToIterationSpace(map));
      }

      // Propagate back to each operand. Extend outer-rank result with inner
      // dims to match full operand rank.
      for (OpOperand &operand : op->getOpOperands()) {
        unsigned idx = operand.getOperandNumber();
        AffineMap map = indexingMaps[idx];
        TileSizes outerTileSizes = iterTileSizes.mapFromIterationSpace(map);
        if (outerTileSizes.empty()) {
          continue;
        }
        ArrayRef<int64_t> innerShape = innerTiledOp.getOperandInnerShape(idx);
        TileSizes fullTileSizes = outerTileSizes.append(innerShape);
        TileSizeLattice *opLattice = operands[idx];
        propagateIfChanged(opLattice, opLattice->meet(fullTileSizes));
      }
      return success();
    }

    // Im2col ops: propagate result tile sizes back to the output operand.
    // Im2col's output dimensions are the iteration domain (identity map),
    // so tile sizes propagate directly.
    if (auto im2colOp = dyn_cast<IREE::LinalgExt::Im2colOp>(op)) {
      TileSizes tileSizes = getTileSizesFor(im2colOp.getResult(0), results[0]);
      // Propagate to the output (DPS init) operand.
      unsigned outputIdx = im2colOp.getDpsInitsMutable()[0].getOperandNumber();
      TileSizeLattice *outputLattice = operands[outputIdx];
      propagateIfChanged(outputLattice, outputLattice->meet(tileSizes));
      return success();
    }

    // Pack ops: result tile sizes → source tile sizes (backward).
    if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
      if (ShapedType::isDynamicShape(packOp.getStaticInnerTiles())) {
        return success();
      }
      if (packOp.getPaddingValue()) {
        // We do not backward propagate for pack with padding, as it would
        // potentially propagate too large tile sizes.
        return success();
      }
      TileSizes resultTileSizes =
          getTileSizesFor(packOp->getResult(0), results[0]);
      TileSizes srcTileSizes = resultTileSizes.mapPackDestToSource(
          packOp.getSourceRank(), packOp.getInnerDimsPos(),
          packOp.getStaticInnerTiles(), packOp.getOuterDimsPerm());
      TileSizeLattice *srcLattice = operands[0];
      propagateIfChanged(srcLattice, srcLattice->meet(srcTileSizes));
      return success();
    }

    // Unpack ops: result tile sizes → source tile sizes (backward).
    if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
      if (ShapedType::isDynamicShape(unpackOp.getStaticInnerTiles())) {
        return success();
      }
      TileSizes resultTileSizes =
          getTileSizesFor(unpackOp->getResult(0), results[0]);
      TileSizes srcTileSizes = resultTileSizes.mapPackSourceToDest(
          unpackOp.getInnerDimsPos(), unpackOp.getStaticInnerTiles(),
          unpackOp.getOuterDimsPerm());
      TileSizeLattice *srcLattice = operands[0];
      propagateIfChanged(srcLattice, srcLattice->meet(srcTileSizes));
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

/// Gather tile sizes into the iteration space of an op by looking up each
/// operand's lattice state in the solver.
static TileSizes getIterationSpaceTileSizes(Operation *op, unsigned numLoops,
                                            ArrayRef<AffineMap> indexingMaps,
                                            const DataFlowSolver &solver) {
  TileSizes iterTileSizes(numLoops);
  for (OpOperand &operand : op->getOpOperands()) {
    Value val = operand.get();
    const TileSizeLattice *lattice = solver.lookupState<TileSizeLattice>(val);
    TileSizes tileSize = getTileSizesFor(val, lattice);
    if (tileSize.empty()) {
      continue;
    }
    AffineMap map = indexingMaps[operand.getOperandNumber()];
    assert(map.getNumDims() == numLoops);
    iterTileSizes.merge(tileSize.mapToIterationSpace(map));
  }
  return iterTileSizes;
}

/// Gather tile sizes into the iteration space of a linalg op by looking up
/// both operand and result lattice states in the solver.
static TileSizes
getLinalgIterationSpaceTileSizes(linalg::LinalgOp linalgOp,
                                 const DataFlowSolver &solver) {
  TileSizes iterTileSizes =
      getIterationSpaceTileSizes(linalgOp, linalgOp.getNumLoops(),
                                 linalgOp.getIndexingMapsArray(), solver);

  for (auto [idx, result] : llvm::enumerate(linalgOp->getResults())) {
    const TileSizeLattice *lattice =
        solver.lookupState<TileSizeLattice>(result);
    TileSizes tileSizes = getTileSizesFor(result, lattice);
    OpOperand *init = linalgOp.getDpsInitOperand(idx);
    AffineMap map = linalgOp.getMatchingIndexingMap(init);
    iterTileSizes.merge(tileSizes.mapToIterationSpace(map));
  }

  return iterTileSizes;
}

/// Get tile sizes for an im2col op from its result lattice. Im2col's output
/// dimensions are the iteration domain, so the result lattice directly holds
/// the iteration-space tile sizes.
static TileSizes getIm2colTileSizes(IREE::LinalgExt::Im2colOp im2colOp,
                                    const DataFlowSolver &solver) {
  Value result = im2colOp.getResult(0);
  const TileSizeLattice *lattice = solver.lookupState<TileSizeLattice>(result);
  return getTileSizesFor(result, lattice);
}

static std::optional<TileSizes>
getUseOperandTileSizes(OpOperand &use, const DataFlowSolver &solver) {
  Operation *user = use.getOwner();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(user)) {
    TileSizes iterTileSizes =
        getLinalgIterationSpaceTileSizes(linalgOp, solver);
    AffineMap map = linalgOp.getMatchingIndexingMap(&use);
    TileSizes operandTileSizes = iterTileSizes.mapFromIterationSpace(map);
    if (!operandTileSizes.isDefined()) {
      return std::nullopt;
    }
    return operandTileSizes;
  }

  if (auto toLayoutOp = dyn_cast<ToLayoutOp>(user)) {
    Value result = toLayoutOp.getResult();
    const TileSizeLattice *lattice =
        solver.lookupState<TileSizeLattice>(result);
    TileSizes tileSizes = getTileSizesFor(result, lattice);
    if (!tileSizes.isDefined()) {
      return std::nullopt;
    }
    return tileSizes;
  }

  return std::nullopt;
}

static LogicalResult
materializeDuplicatableLinalgOp(linalg::LinalgOp linalgOp,
                                const DataFlowSolver &solver) {
  if (linalgOp->getNumResults() != 1) {
    return failure();
  }
  if (linalgOp->hasAttr(kVectorTileSizesAttrName)) {
    return success();
  }
  Value result = linalgOp->getResult(0);
  if (!isDuplicatable(result)) {
    return failure();
  }

  SmallVector<std::pair<TileSizes, SmallVector<OpOperand *>>> useGroups;
  for (OpOperand &use : result.getUses()) {
    std::optional<TileSizes> maybeTileSizes =
        getUseOperandTileSizes(use, solver);
    if (!maybeTileSizes || !maybeTileSizes->isDefined()) {
      continue;
    }
    auto it = llvm::find_if(
        useGroups, [&](auto &entry) { return entry.first == *maybeTileSizes; });
    if (it == useGroups.end()) {
      useGroups.emplace_back(*maybeTileSizes, SmallVector<OpOperand *>{&use});
    } else {
      it->second.push_back(&use);
    }
  }

  if (useGroups.empty()) {
    return success();
  }

  auto setTileSizeAttr = [](Operation *op, TileSizes tileSizes) {
    op->setAttr(kVectorTileSizesAttrName,
                DenseI64ArrayAttr::get(op->getContext(), tileSizes.getDims()));
  };

  setTileSizeAttr(linalgOp, useGroups.front().first);

  OpBuilder builder(linalgOp);
  builder.setInsertionPointAfter(linalgOp);
  for (auto &useGroup : llvm::drop_begin(useGroups)) {
    Operation *cloned = builder.clone(*linalgOp.getOperation());
    setTileSizeAttr(cloned, useGroup.first);
    Value clonedResult = cloned->getResult(0);
    for (OpOperand *use : useGroup.second) {
      use->set(clonedResult);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MaterializeVectorTileSizesPass
//===----------------------------------------------------------------------===//

namespace {

class MaterializeVectorTileSizesPass final
    : public impl::MaterializeVectorTileSizesPassBase<
          MaterializeVectorTileSizesPass> {
public:
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<TileSizeForwardAnalysis>();
    SymbolTableCollection symbolTable;
    solver.load<TileSizeBackwardAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    SmallVector<linalg::LinalgOp> duplicatableLinalgOps;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (linalgOp->getNumResults() == 1 &&
          isDuplicatable(linalgOp->getResult(0))) {
        duplicatableLinalgOps.push_back(linalgOp);
      }
    });
    for (linalg::LinalgOp linalgOp : duplicatableLinalgOps) {
      if (failed(materializeDuplicatableLinalgOp(linalgOp, solver))) {
        return signalPassFailure();
      }
    }

    auto materialize = [](Operation *op, TileSizes tileSizes) -> LogicalResult {
      if (tileSizes.isOverdefined()) {
        if (op->hasAttr(kVectorTileSizesAttrName)) {
          return success();
        }
        op->emitOpError()
            << "tile size analysis did not determine a valid tile size";
        return failure();
      }
      if (!tileSizes.isDefined()) {
        LDBG() << "Analysis did not determine tile size for " << *op;
        return success();
      }
      op->setAttr(
          kVectorTileSizesAttrName,
          DenseI64ArrayAttr::get(op->getContext(), tileSizes.getDims()));
      return success();
    };

    auto result = funcOp->walk([&](Operation *op) -> WalkResult {
      if (isa<linalg::PackOp>(op) || isa<linalg::UnPackOp>(op)) {
        // linalg.pack and linalg.unpack have an unpacked (rank N) and a packed
        // (rank N + K) domain. linalg.pack converts from the unpacked domain to
        // the packed domain, linalg.unpack works the other way round.
        // Vectorization of the operations expects vector sizes in the packed
        // domain. After analysis, these are available on operand of linalg.pack
        // and the result of linalg.unpack, respectively.
        Value packedVal;
        if (auto packOp = dyn_cast<linalg::PackOp>(op)) {
          packedVal = packOp.getResult();
        } else if (auto unpackOp = dyn_cast<linalg::UnPackOp>(op)) {
          packedVal = unpackOp.getSource();
        }
        const TileSizeLattice *lattice =
            solver.lookupState<TileSizeLattice>(packedVal);
        TileSizes tileSizes = getTileSizesFor(packedVal, lattice);
        return WalkResult(materialize(op, tileSizes));
      }
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        TileSizes tileSizes =
            getLinalgIterationSpaceTileSizes(linalgOp, solver);
        assert(!tileSizes.isDefined() ||
               tileSizes.rank() == linalgOp.getNumLoops());
        return WalkResult(materialize(op, tileSizes));
      }

      if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
        SmallVector<AffineMap> indexingMaps =
            innerTiledOp.getIndexingMapsArray();
        unsigned numLoops = indexingMaps[0].getNumDims();
        TileSizes tileSizes =
            getIterationSpaceTileSizes(op, numLoops, indexingMaps, solver);
        return WalkResult(materialize(op, tileSizes));
      }
      if (auto im2colOp = dyn_cast<IREE::LinalgExt::Im2colOp>(op)) {
        TileSizes tileSizes = getIm2colTileSizes(im2colOp, solver);
        return WalkResult(materialize(op, tileSizes));
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
