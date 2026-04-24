// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"

#include <cassert>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "iree-codegen-vector-layout-analysis"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

/// Maximum number of candidate layouts tracked per value. Kept small to bound
/// analysis cost; most values see 1-2 candidates in practice.
static constexpr int kMaxCandidatesPerValue = 4;

/// Maximum length of chains of cheap-to-compute operations that get duplicated
/// for layout conflict resolution.
static constexpr size_t kMaxChainLength = 8;

//===----------------------------------------------------------------------===//
// Layout Analysis
//
// Phase 1: Forward propagation with multi-candidate tracking. Seeds from
//   ToLayoutOp anchors, propagates forward through uses. Each value accumulates
//   up to kMaxCandidatesPerValue candidate layouts. No IR mutation.
//
// Resolve: Pick first candidate for each value (first-wins). The multi-
//   candidate data structure is ready for a cost model later.
//
// Phase 2: Backward fixup. Walks operations in reverse program order. For each
//   op, determines operand layouts from resolved result/operand layouts.
//   Assigns missing layouts, clones cheap ops, or inserts to_layout
//   conversions.
//
// The forward analysis is the main driver of the analysis. The reason for this
// is that for a program to be well-formed for vector distribution, there must
// be some way for the final store/return to get a layout. Otherwise, there
// is not enough information in the program to determine how distribution should
// be done. The forward analysis ensures that the final return/store gets a
// layout in a well-formed program. The rest of the program can get their
// layouts from backward propagation, everything in the program must eventually
// reach the store/return.
//===----------------------------------------------------------------------===//

namespace {

struct LayoutAnalysis {
  /// Multiple candidate layouts per value (Phase 1).
  llvm::MapVector<Value, llvm::SmallSetVector<VectorLayoutInterface, 4>>
      candidates;
  /// Resolved layouts: single layout per value (after resolve, used by fixup).
  llvm::MapVector<Value, VectorLayoutInterface> resolved;
  /// Forward worklist (Phase 1 only).
  std::queue<Value> forward;

  //===--- Phase 1: Forward propagation ---===//

  bool addCandidate(Value val, VectorLayoutInterface layout);
  VectorLayoutInterface getFirstCandidate(Value val) const;
  void seed(Operation *root);
  void propagateForward(Value val);
  void propagateOneForward(Value val, VectorLayoutInterface layout);
  void runForward();

  //===--- Resolve ---===//

  void resolve();

  //===--- Phase 2: Backward fixup ---===//

  VectorLayoutInterface getResolvedLayout(Value val) const {
    return resolved.lookup(val);
  }
  bool hasResolvedLayout(Value val) const { return resolved.contains(val); }

  void fixupRegion(Region &region);
  void fixupOp(Operation *op);
  void setLayoutOrClone(OpOperand *val, VectorLayoutInterface layout);
};

} // namespace

//===----------------------------------------------------------------------===//
// Phase 1: Forward Propagation (no IR mutation)
//===----------------------------------------------------------------------===//

/// Add a candidate layout for a value. Returns true if a new candidate was
/// added. Schedules the value for forward propagation.
bool LayoutAnalysis::addCandidate(Value val, VectorLayoutInterface layout) {
  if (!layout) {
    return false;
  }
  if (!isa<ShapedType>(val.getType())) {
    return false;
  }
  llvm::SmallSetVector<VectorLayoutInterface, 4> &set = candidates[val];
  if (set.size() >= kMaxCandidatesPerValue) {
    return false;
  }
  if (!set.insert(layout)) {
    return false;
  }
  forward.push(val);
  return true;
}

/// Return the first candidate layout for a value, or null.
VectorLayoutInterface LayoutAnalysis::getFirstCandidate(Value val) const {
  auto it = candidates.find(val);
  if (it == candidates.end() || it->second.empty()) {
    return {};
  }
  return it->second.front();
}

/// Seed anchors from ToLayoutOps.
void LayoutAnalysis::seed(Operation *root) {
  root->walk([&](ToLayoutOp toLayout) {
    LDBG() << "Seeding layout from to_layout op: " << toLayout << "\n";
    addCandidate(toLayout.getResult(), toLayout.getLayout());
  });
}

/// Propagate all candidates for a value forward through its users.
void LayoutAnalysis::propagateForward(Value val) {
  LDBG() << "Propagating forward for value: " << val << "\n";
  auto it = candidates.find(val);
  if (it == candidates.end()) {
    return;
  }
  for (VectorLayoutInterface layout : it->second) {
    propagateOneForward(val, layout);
  }
}

/// Run Phase 1: drain forward queue. Convergence is guaranteed because each
/// value can contribute at most kMaxCandidatesPerValue new candidates, and
/// addCandidate only enqueues when a genuinely new candidate is inserted.
void LayoutAnalysis::runForward() {
  while (!forward.empty()) {
    Value val = forward.front();
    forward.pop();
    propagateForward(val);
  }
}

/// Propagate a single layout forward through all users of a value.
void LayoutAnalysis::propagateOneForward(Value val,
                                         VectorLayoutInterface layout) {
  for (OpOperand &use : val.getUses()) {
    unsigned operandIdx = use.getOperandNumber();
    Operation *user = use.getOwner();

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      addCandidate(arg, layout);
      addCandidate(result, layout);
      continue;
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      Operation *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        Value arg = forOp.getRegionIterArg(operandIdx);
        Value result = forOp->getResult(operandIdx);
        addCandidate(arg, layout);
        addCandidate(result, layout);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        Value result = ifOp->getResult(operandIdx);
        addCandidate(result, layout);
        continue;
      }
    }

    if (auto yieldOp = dyn_cast<vector::YieldOp>(user)) {
      Operation *parentOp = cast<vector::MaskOp>(yieldOp->getParentOp());
      Value result = parentOp->getResult(operandIdx);
      addCandidate(result, layout);
      continue;
    }

    if (OpTrait::hasElementwiseMappableTraits(user)) {
      for (OpResult result : user->getOpResults()) {
        addCandidate(result, layout);
      }
      continue;
    }

    if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(user)) {
      if (multiReduce.getSource() == val) {
        if (auto maskOp =
                dyn_cast<vector::MaskOp>(multiReduce->getParentOp())) {
          addCandidate(maskOp.getMask(), layout);
        }
        SmallVector<bool> reductionMask = multiReduce.getReductionMask();
        VectorLayoutInterface reduceLayout = layout.project(reductionMask);
        addCandidate(multiReduce.getResult(), reduceLayout);
        continue;
      }
      if (multiReduce.getAcc() == val) {
        addCandidate(multiReduce.getResult(), layout);
        continue;
      }
    }

    if (auto transpose = dyn_cast<vector::TransposeOp>(user)) {
      if (transpose.getVector() == val) {
        addCandidate(transpose.getResult(),
                     layout.permute(transpose.getPermutation()));
        continue;
      }
    }

    if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
      if (contract.getAcc() == val) {
        addCandidate(contract.getResult(), layout);
        continue;
      }
      if (contract.getLhs() == val || contract.getRhs() == val) {
        if (auto maskOp = dyn_cast<vector::MaskOp>(contract->getParentOp())) {
          AffineMap map = contract.getMatchingIndexingMap(&use);
          if (map.isPermutation()) {
            addCandidate(maskOp.getMask(),
                         layout.apply(inversePermutation(map)));
          }
        }
        // Uses first candidate for each operand; first-wins avoids
        // combinatorial explosion over candidate pairings.
        // TODO: Consider all candidate combinations with a cost model.
        VectorLayoutInterface lhsLayout = getFirstCandidate(contract.getLhs());
        VectorLayoutInterface rhsLayout = getFirstCandidate(contract.getRhs());
        if (lhsLayout && rhsLayout) {
          AffineMap lhsMap = contract.getIndexingMapsArray()[0];
          AffineMap rhsMap = contract.getIndexingMapsArray()[1];
          AffineMap resMap = contract.getIndexingMapsArray()[2];
          VectorLayoutInterface resLayout = lhsLayout.getRecombinedLayout(
              {lhsLayout, rhsLayout}, {lhsMap, rhsMap}, resMap);
          addCandidate(contract.getResult(), resLayout);
        }
        continue;
      }
    }

    if (auto gather = dyn_cast<vector::GatherOp>(user)) {
      addCandidate(gather.getResult(), layout);
      continue;
    }

    // ArgCompare reduces along a single dimension, so input layouts must be
    // projected by removing the reduction dimension to derive result layouts.
    // Init operands flow directly to their corresponding results.
    if (auto argCompare = dyn_cast<ArgCompareOp>(user)) {
      if (argCompare.getInputValue() == val ||
          (argCompare.getInputIndex() && argCompare.getInputIndex() == val)) {
        // Project input layout by removing the reduction dimension.
        // NOTE: ArgCompareOp's verifier guarantees dimension < rank.
        int64_t reductionDim = argCompare.getDimension();
        int64_t rank = cast<VectorType>(val.getType()).getRank();
        SmallVector<bool> reductionMask(rank, false);
        reductionMask[reductionDim] = true;
        VectorLayoutInterface reducedLayout = layout.project(reductionMask);
        addCandidate(argCompare.getResultValue(), reducedLayout);
        addCandidate(argCompare.getResultIndex(), reducedLayout);
        continue;
      }
      if (argCompare.getInitValue() == val) {
        addCandidate(argCompare.getResultValue(), layout);
      }
      if (argCompare.getInitIndex() == val) {
        addCandidate(argCompare.getResultIndex(), layout);
      }
      continue;
    }

    if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(user)) {
      addCandidate(shapeCast.getResult(),
                   layout.reshape(shapeCast.getResultVectorType().getShape()));
      continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// Resolve
//===----------------------------------------------------------------------===//

/// Pick first candidate for each value.
void LayoutAnalysis::resolve() {
  for (auto &[val, candidateSet] : candidates) {
    if (!candidateSet.empty()) {
      resolved[val] = candidateSet.front();
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase 2: Backward Fixup (mutates IR)
//===----------------------------------------------------------------------===//

/// Walk operations in reverse within a region, fixing up operand layouts.
/// Ops are collected upfront so that newly inserted to_layout ops (from
/// setLayoutOrClone) are not visited by the walk.
void LayoutAnalysis::fixupRegion(Region &region) {
  for (Block &block : region.getBlocks()) {
    SmallVector<Operation *> ops;
    for (Operation &op : llvm::reverse(block.getOperations())) {
      ops.push_back(&op);
    }
    for (Operation *op : ops) {
      fixupOp(op);
    }
  }
}

/// Fix up operand layouts for a single operation. Result layouts are fixed
/// (from resolve); this determines what operand layouts should be.
void LayoutAnalysis::fixupOp(Operation *op) {
  // transfer_write: vector operand layout -> derive mask layout.
  if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(write.getVector());
    if (!layout || !write.getMask()) {
      return;
    }
    AffineMap maskMap =
        inversePermutation(compressUnusedDims(write.getPermutationMap()));
    setLayoutOrClone(&write.getMaskMutable()[0], layout.apply(maskMap));
    return;
  }

  // transfer_read: result layout -> derive mask layout.
  if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(read.getResult());
    if (!layout || !read.getMask()) {
      return;
    }
    AffineMap maskMap =
        inversePermutation(compressUnusedDims(read.getPermutationMap()));
    setLayoutOrClone(&read.getMaskMutable()[0], layout.apply(maskMap));
    return;
  }

  // elementwise: result layout -> all operands get same layout.
  if (OpTrait::hasElementwiseMappableTraits(op)) {
    VectorLayoutInterface layout = getResolvedLayout(op->getResult(0));
    for (OpOperand &operand : op->getOpOperands()) {
      setLayoutOrClone(&operand, layout);
    }
    return;
  }

  // to_layout: result layout -> input gets same layout.
  if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
    if (toLayout.getSharedMemoryConversion()) {
      // The layout input is coming through shared memory, skip
      // back-propagation.
      return;
    }
    VectorLayoutInterface layout = getResolvedLayout(toLayout.getResult());
    setLayoutOrClone(&toLayout.getInputMutable(), layout);
    return;
  }

  // multi_dim_reduction: result layout -> acc gets same layout.
  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(multiReduce.getResult());
    setLayoutOrClone(&multiReduce.getAccMutable(), layout);
    return;
  }

  // ArgCompare: result layouts propagate back to init operands because
  // inits serve as identity values and must match result distribution.
  if (auto argCompare = dyn_cast<ArgCompareOp>(op)) {
    VectorLayoutInterface valueLayout =
        getResolvedLayout(argCompare.getResultValue());
    VectorLayoutInterface indexLayout =
        getResolvedLayout(argCompare.getResultIndex());
    setLayoutOrClone(&argCompare.getInitValueMutable(), valueLayout);
    setLayoutOrClone(&argCompare.getInitIndexMutable(), indexLayout);
    return;
  }

  // transpose: result layout -> input gets inverse-permuted layout.
  if (auto transpose = dyn_cast<vector::TransposeOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(transpose.getResult());
    if (!layout) {
      return;
    }
    setLayoutOrClone(
        &transpose.getVectorMutable(),
        layout.permute(invertPermutationVector(transpose.getPermutation())));
    return;
  }

  // broadcast: result layout -> source gets projected layout.
  if (auto broadcast = dyn_cast<vector::BroadcastOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(broadcast.getResult());
    if (!layout || !isa<VectorType>(broadcast.getSourceType())) {
      return;
    }
    assert(broadcast.computeBroadcastedUnitDims().empty() &&
           "Stretching in broadcasting not implemented yet.");
    int64_t numBroadcastedDims =
        broadcast.getResultVectorType().getRank() -
        cast<VectorType>(broadcast.getSourceType()).getRank();
    SmallVector<bool> reductionMask(layout.getRank(), false);
    std::fill(reductionMask.begin(), reductionMask.begin() + numBroadcastedDims,
              true);
    setLayoutOrClone(&broadcast.getSourceMutable(),
                     layout.project(reductionMask));
    return;
  }

  // contract: result layout -> acc gets same layout.
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(contract.getResult());
    setLayoutOrClone(&contract.getAccMutable(), layout);
    return;
  }

  // gather: result layout -> indices, mask, passthru get same layout.
  if (auto gather = dyn_cast<vector::GatherOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(gather.getResult());
    setLayoutOrClone(&gather.getIndicesMutable(), layout);
    setLayoutOrClone(&gather.getMaskMutable(), layout);
    setLayoutOrClone(&gather.getPassThruMutable(), layout);
    return;
  }

  // transfer_gather: result layout -> index vecs + mask get projected layouts.
  if (auto gather = dyn_cast<TransferGatherOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(gather.getResult());
    if (!layout) {
      return;
    }
    SmallVector<AffineMap> maps = gather.getIndexingMapsArray();
    int64_t numIndexVecs = gather.getIndexVecs().size();
    for (auto [i, operand] : llvm::enumerate(gather.getIndexVecsMutable())) {
      AffineMap indexVecMap = maps[1 + i];
      AffineMap projected =
          AffineMap::get(indexVecMap.getNumDims(), 0, indexVecMap.getResults(),
                         indexVecMap.getContext());
      setLayoutOrClone(&operand, layout.apply(projected));
    }
    if (gather.getMask()) {
      OpOperand &mask = gather.getMaskMutable()[0];
      AffineMap maskMap = maps[1 + numIndexVecs];
      AffineMap projected = AffineMap::get(
          maskMap.getNumDims(), 0, maskMap.getResults(), maskMap.getContext());
      setLayoutOrClone(&mask, layout.apply(projected));
    }
    return;
  }

  // shape_cast: result layout -> source gets reshaped layout.
  if (auto shapeCast = dyn_cast<vector::ShapeCastOp>(op)) {
    VectorLayoutInterface layout = getResolvedLayout(shapeCast.getResult());
    if (!layout) {
      return;
    }
    setLayoutOrClone(
        &shapeCast.getSourceMutable(),
        layout.reshape(shapeCast.getSourceVectorType().getShape()));
    return;
  }

  // scf.for: fix init_args/yield from result layouts, then recurse into body.
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [i, result] : llvm::enumerate(forOp.getResults())) {
      VectorLayoutInterface layout = getResolvedLayout(result);
      setLayoutOrClone(&yieldOp->getOpOperand(i), layout);
      setLayoutOrClone(&forOp.getInitArgsMutable()[i], layout);
    }
    fixupRegion(forOp.getBodyRegion());
    return;
  }

  // Default: recurse into nested regions for ops we don't explicitly handle
  // (e.g. scf.forall, scf.if, vector.mask).
  for (Region &region : op->getRegions()) {
    fixupRegion(region);
  }
}

/// Returns true if the operation is a duplicatable leaf: trivially cheap to
/// recompute and has no operands that need cloning.
static bool isDuplicatableLeaf(Operation *op) {
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<vector::StepOp, vector::CreateMaskOp, vector::ConstantMaskOp>(op);
}

/// Returns true if the operation is a cheap single-result op that can be
/// cloned as part of a duplicatable chain. These ops must be pure and have
/// exactly one result.
static bool isCheapToClone(Operation *op) {
  if (isDuplicatableLeaf(op)) {
    return true;
  }
  return isPure(op) &&
         (isa<vector::BroadcastOp, vector::TransposeOp, vector::ShapeCastOp>(
              op) ||
          OpTrait::hasElementwiseMappableTraits(op));
}

/// Collect a chain of ops that can be cloned together. Starting from `op`,
/// walk backward through cheap-to-clone ops until we reach duplicatable
/// leaves, constants, or non-vector operands. Returns true if the entire
/// chain is safe to clone. Shared intermediates (with multiple uses) are
/// allowed because all ops in the chain are cheap to duplicate.
static bool collectDuplicatableChain(Operation *op,
                                     SmallVectorImpl<Operation *> &chain) {
  // The chain is built bottom-up (from consumer toward producers).
  Block *block = op->getBlock();
  std::queue<Operation *> worklist;
  llvm::SmallPtrSet<Operation *, 8> visited;
  worklist.push(op);
  while (!worklist.empty()) {
    Operation *current = worklist.front();
    worklist.pop();
    if (!visited.insert(current).second) {
      // Operation was already visited.
      continue;
    }
    if (!isCheapToClone(current)) {
      return false;
    }
    chain.push_back(current);
    if (chain.size() > kMaxChainLength) {
      return false;
    }
    if (isDuplicatableLeaf(current)) {
      continue;
    }
    for (Value operand : current->getOperands()) {
      // Non-vector operands (scalars, indices) don't need cloning.
      if (!isa<VectorType>(operand.getType())) {
        continue;
      }
      // Single-element vectors don't need cloning.
      auto opVecTy = cast<VectorType>(operand.getType());
      if (opVecTy.hasStaticShape() && opVecTy.getNumElements() == 1) {
        continue;
      }
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        return false;
      }
      if (defOp->hasTrait<OpTrait::ConstantLike>()) {
        continue;
      }
      if (defOp->getBlock() != block) {
        return false;
      }
      worklist.push(defOp);
    }
  }
  return true;
}

/// Assign a layout to an operand, cloning cheap ops or inserting conversions
/// on conflict.
void LayoutAnalysis::setLayoutOrClone(OpOperand *val,
                                      VectorLayoutInterface layout) {
  if (!layout) {
    return;
  }
  if (!isa<ShapedType>(val->get().getType())) {
    return;
  }

  // No layout yet -- assign.
  if (!hasResolvedLayout(val->get())) {
    resolved[val->get()] = layout;
    return;
  }

  // Same layout -- nothing to do.
  if (getResolvedLayout(val->get()) == layout) {
    return;
  }

  // Different layout -- clone cheap ops or insert to_layout conversion.
  OpBuilder b(val->getOwner());
  if (Operation *defOp = val->get().getDefiningOp()) {
    // Try to clone a chain of cheap ops rooted at duplicatable leaves.
    if (isCheapToClone(defOp)) {
      SmallVector<Operation *> chain;
      if (collectDuplicatableChain(defOp, chain)) {
        // Sort so cloning visits producers before consumers.
        computeTopologicalSorting(chain);
        IRMapping mapping;
        b.setInsertionPoint(chain.front());
        for (Operation *op : chain) {
          b.clone(*op, mapping);
        }
        Value cloned = mapping.lookup(val->get());
        val->set(cloned);
        resolved[cloned] = layout;
        // Propagate layouts through the cloned chain. The cloned ops are
        // not visited by the outer fixupRegion walk (which collects ops
        // upfront), so we must fix them up here. Walk in reverse program
        // order so that result layouts propagate to operands. This does
        // not recurse because cloned ops are all cheap (no nested regions).
        for (Operation *op : llvm::reverse(chain)) {
          fixupOp(mapping.lookup(op->getResult(0)).getDefiningOp());
        }
        return;
      }
    }
  }

  // Non-cheap op -- insert to_layout conversion.
  Value v = val->get();
  Value converted = ToLayoutOp::create(b, v.getLoc(), v, layout);
  val->set(converted);
  resolved[converted] = layout;
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void propagateVectorLayoutInfo(
    Operation *root, llvm::MapVector<Value, VectorLayoutInterface> &layouts) {
  LayoutAnalysis analysis;

  // Phase 1: Seed anchors and forward propagation (no IR mutation).
  analysis.seed(root);
  analysis.runForward();

  // Resolve: pick first candidate for each value.
  analysis.resolve();

  // Phase 2: Backward fixup (mutates IR).
  for (Region &region : root->getRegions()) {
    analysis.fixupRegion(region);
  }

  layouts = std::move(analysis.resolved);
}

#define GEN_PASS_DEF_TESTVECTORLAYOUTANALYSISPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

struct TestVectorLayoutAnalysisPass final
    : impl::TestVectorLayoutAnalysisPassBase<TestVectorLayoutAnalysisPass> {
  void runOnOperation() override {
    Operation *root = getOperation();
    llvm::MapVector<Value, VectorLayoutInterface> layouts;
    propagateVectorLayoutInfo(root, layouts);

    root->walk([&](Operation *op) {
      if (isa<ToLayoutOp>(op)) {
        return;
      }

      for (OpResult result : op->getOpResults()) {
        if (layouts.contains(result)) {
          op->emitRemark("layout of result #")
              << result.getResultNumber() << " is " << layouts[result];
        }
      }
    });
  }
};
}; // namespace mlir::iree_compiler
