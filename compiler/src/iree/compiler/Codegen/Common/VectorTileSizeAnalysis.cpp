// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "iree-codegen-vector-tile-size-analysis"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

using TileSizeSet = llvm::SmallSet<int64_t, 2>;

/// Per-dimension tile size candidates. Each dimension has an independent set
/// of candidate tile sizes.
class TileSizeCandidates {
public:
  TileSizeCandidates() = default;
  explicit TileSizeCandidates(unsigned rank) : dims(rank) {}

  unsigned rank() const { return dims.size(); }
  bool empty() const { return dims.empty(); }

  const TileSizeSet &operator[](unsigned i) const { return dims[i]; }
  TileSizeSet &operator[](unsigned i) { return dims[i]; }

  /// Merge candidates from `other` into this. Returns true if anything changed.
  bool merge(const TileSizeCandidates &other) {
    assert(rank() == other.rank() && "rank mismatch");
    bool changed = false;
    for (unsigned i = 0; i < rank(); ++i) {
      for (int64_t v : other.dims[i]) {
        changed |= dims[i].insert(v).second;
      }
    }
    return changed;
  }

  /// Merge a single concrete tile size (one value per dimension).
  /// Values of -1 (unknown) are skipped. If this object is uninitialized
  /// (rank 0), it is initialized from the size of `concreteSizes`.
  bool merge(ArrayRef<int64_t> concreteSizes) {
    if (empty()) {
      dims.resize(concreteSizes.size());
    }
    assert(rank() == concreteSizes.size() && "rank mismatch");
    bool changed = false;
    for (unsigned i = 0; i < rank(); ++i) {
      if (concreteSizes[i] != -1) {
        changed |= dims[i].insert(concreteSizes[i]).second;
      }
    }
    return changed;
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

struct TileSizeState {
  void propagateForward(Value val);
  void propagateBackward(Value val);

  /// Merge candidates into a value and enqueue if anything changed.
  void mergeAndEnqueue(Value val, const TileSizeCandidates &candidates) {
    if (!isa<ShapedType>(val.getType())) {
      return;
    }
    if (candidates.empty()) {
      return;
    }
    // If val is not yet in the map, inserting it may rehash the DenseMap
    // and invalidate `candidates` if it aliases an existing entry. Copy
    // directly into the new entry to avoid the dangling reference.
    if (!tileSizes.count(val)) {
      tileSizes[val] = candidates;
    } else {
      if (!tileSizes[val].merge(candidates)) {
        return;
      }
    }
    // We don't forward multiple alternatives from operations that are easy to
    // duplicate. CSE will deduplicate DPS init operands, creating edges between
    // unrelated compute operations. Propagating different vector tile sizes via
    // shared DPS inits doesn't provide any value in that case.
    if (isDuplicatable(val) && tileSizes[val].hasAlternatives()) {
      return;
    }
    // Propagate the update.
    forward.push(val);
    backward.push(val);
  }

  /// Convenience: merge a single concrete tile size and enqueue if changed.
  void mergeAndEnqueue(Value val, ArrayRef<int64_t> concreteSizes) {
    TileSizeCandidates candidates(concreteSizes.size());
    candidates.merge(concreteSizes);
    mergeAndEnqueue(val, candidates);
  }

  bool hasTileSize(Value val) const { return tileSizes.count(val); }

  const TileSizeCandidates &getCandidates(Value val) const {
    static const TileSizeCandidates empty;
    auto it = tileSizes.find(val);
    if (it == tileSizes.end()) {
      return empty;
    }
    return it->second;
  }

  /// Propagate through a linalg.generic: given known tile sizes on some
  /// operands, infer tile sizes for other operands via indexing maps.
  void propagateGenericOp(linalg::GenericOp genericOp);

  DenseMap<Value, TileSizeCandidates> tileSizes;
  std::queue<Value> forward;
  std::queue<Value> backward;
};

/// Collect per-dimension tile size candidate sets from a linalg op's operands.
/// Returns a TileSizeCandidates of size numLoops, where each dimension is the
/// union of all candidate tile sizes for that iteration dimension across all
/// operands.
static TileSizeCandidates
getIterationSpaceTileSizes(linalg::LinalgOp linalgOp,
                           const TileSizeState &state) {
  unsigned numLoops = linalgOp.getNumLoops();
  TileSizeCandidates result(numLoops);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto &candidates = state.getCandidates(operand.get());
    if (candidates.empty()) {
      continue;
    }
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    auto mapped = candidates.mapToIterationSpace(map, numLoops);
    result.merge(mapped);
  }
  return result;
}

void TileSizeState::propagateGenericOp(linalg::GenericOp genericOp) {
  auto perDimSizes = getIterationSpaceTileSizes(genericOp, *this);

  // Map per-dimension iteration-space candidates to each operand's dimensions
  // via its indexing map.
  for (OpOperand &operand : genericOp->getOpOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(&operand);
    auto operandCandidates = perDimSizes.mapFromIterationSpace(map);
    if (operandCandidates.empty()) {
      continue;
    }
    mergeAndEnqueue(operand.get(), operandCandidates);
  }

  // Propagate to results via their corresponding init operands.
  for (auto [init, result] :
       llvm::zip_equal(genericOp.getDpsInits(), genericOp.getResults())) {
    mergeAndEnqueue(result, getCandidates(init));
  }
}

void TileSizeState::propagateForward(Value val) {
  auto &candidates = getCandidates(val);
  if (candidates.empty()) {
    return;
  }
  LDBG() << "Propagating tile size forward for: " << val;

  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    unsigned operandIdx = use.getOperandNumber();

    // scf.for: propagate to tied loop body arg and result.
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      mergeAndEnqueue(arg, candidates);
      mergeAndEnqueue(result, candidates);
      continue;
    }

    // scf.yield: propagate to parent op's results/args.
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      Operation *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        Value arg = forOp.getRegionIterArg(operandIdx);
        Value result = forOp->getResult(operandIdx);
        mergeAndEnqueue(arg, candidates);
        mergeAndEnqueue(result, candidates);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        Value result = ifOp->getResult(operandIdx);
        mergeAndEnqueue(result, candidates);
        continue;
      }
    }

    // Elementwise ops: propagate to all results.
    if (OpTrait::hasElementwiseMappableTraits(user)) {
      for (OpResult result : user->getOpResults()) {
        mergeAndEnqueue(result, candidates);
      }
      continue;
    }

    // linalg.generic: propagate through indexing maps.
    if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
      propagateGenericOp(genericOp);
      continue;
    }
  }
}

void TileSizeState::propagateBackward(Value val) {
  LDBG() << "Propagating tile size backward for: " << val;
  auto &candidates = getCandidates(val);
  if (candidates.empty()) {
    return;
  }

  // Block arguments (e.g., scf.for iter_args).
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Operation *parent = val.getParentBlock()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      OpOperand *yielded = forOp.getTiedLoopYieldedValue(blockArg);
      OpOperand *init = forOp.getTiedLoopInit(blockArg);
      if (yielded) {
        mergeAndEnqueue(yielded->get(), candidates);
      }
      if (init) {
        mergeAndEnqueue(init->get(), candidates);
      }
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return;
  }

  // Elementwise ops: propagate to all operands.
  if (OpTrait::hasElementwiseMappableTraits(defOp)) {
    for (OpOperand &operand : defOp->getOpOperands()) {
      if (isa<ShapedType>(operand.get().getType())) {
        mergeAndEnqueue(operand.get(), candidates);
      }
    }
    return;
  }

  // linalg.generic: propagate through indexing maps.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(val).getResultNumber();
    mergeAndEnqueue(genericOp.getDpsInitOperand(resultIdx)->get(), candidates);
    propagateGenericOp(genericOp);
    return;
  }

  // to_layout: propagate to input.
  // We only propagate backward for to_layout, not forward, as to_layout is an
  // anchor for initialization itself.
  if (auto toLayout = dyn_cast<ToLayoutOp>(defOp)) {
    mergeAndEnqueue(toLayout.getInput(), candidates);
    return;
  }

  // scf.for results: propagate to yield and init.
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(val).getResultNumber();
    Value init = forOp.getInits()[resultIdx];
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    mergeAndEnqueue(init, candidates);
    mergeAndEnqueue(yieldOp.getOperand(resultIdx), candidates);
    return;
  }

  // scf.if results: propagate to yields in both regions.
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(val).getResultNumber();
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    mergeAndEnqueue(thenYield.getOperand(resultIdx), candidates);
    assert(ifOp.elseBlock() && "scf.if with results must have an else block");
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    mergeAndEnqueue(elseYield.getOperand(resultIdx), candidates);
    return;
  }
}

/// Run the VectorTileSizeAnalysis on the given root operation.
static void runAnalysis(Operation *root, TileSizeState &state) {
  // Initialize from to_layout anchors.
  root->walk([&](ToLayoutOp toLayout) {
    SmallVector<int64_t> undistShape =
        toLayout.getLayout().getUndistributedShape();
    LDBG() << "Anchor: " << toLayout;
    state.mergeAndEnqueue(toLayout.getResult(), undistShape);
  });

  // Fixpoint iteration: forward first, then backward.
  while (!state.forward.empty() || !state.backward.empty()) {
    if (!state.forward.empty()) {
      Value val = state.forward.front();
      state.forward.pop();
      state.propagateForward(val);
    } else {
      Value val = state.backward.front();
      state.backward.pop();
      state.propagateBackward(val);
    }
  }
}

/// Given a linalg op and the analysis state, compute per-dimension sets of
/// candidate tile sizes. Returns a vector of size numLoops, where each entry
/// is the deduplicated set of tile sizes for that iteration dimension.
/// Returns an empty vector if any dimension has no candidates.
static SmallVector<SmallVector<int64_t>>
getPerDimTileSizes(linalg::LinalgOp linalgOp, const TileSizeState &state) {
  auto perDimSizes = getIterationSpaceTileSizes(linalgOp, state);

  // Return empty if any dimension has no candidates.
  SmallVector<SmallVector<int64_t>> results;
  for (unsigned i = 0; i < perDimSizes.rank(); ++i) {
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

    TileSizeState state;
    runAnalysis(funcOp, state);

    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      auto perDimSizes = getPerDimTileSizes(linalgOp, state);
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
