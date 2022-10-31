// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is a variant of DispatchLinalgOnTensors.cpp. DispatchWorkgroupsOps are
// built from DispatchRegionOps. This file can eventually replace the original
// DispatchLinalgOnTensors.cpp
//
// Note: The heuristic part of the implementation is unchanged and copied from
// DispatchLinalgOnTensors.cpp.

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

#define DEBUG_TYPE "iree-flow-dispatch-linalg-on-tensors-via-region-ops"

static const int kInlineConstantByteLength = 256;
static const bool kEnableMultiResultDispatches = false;
static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

//===----------------------------------------------------------------------===//
// Helpers for fusion group formation
//===----------------------------------------------------------------------===//

namespace {
/// A rewriter that keeps track of all tensor::DimOps.
class TensorDimTrackingRewriter : public IRRewriter {
 public:
  /// Create a new rewriter: Scan the given op for tensor::DimOps.
  TensorDimTrackingRewriter(Operation *op) : IRRewriter(op->getContext()) {
    op->walk([&](tensor::DimOp dimOp) { dimOps.insert(dimOp.getOperation()); });
  }

  /// Return all tracked tensor::DimOps.
  SmallVector<tensor::DimOp> getTensorDimOps() {
    SmallVector<tensor::DimOp> result;
    for (Operation *op : dimOps) result.push_back(cast<tensor::DimOp>(op));
    return result;
  }

 protected:
  void notifyOperationRemoved(Operation *op) override {
    IRRewriter::notifyOperationRemoved(op);
    if (isa<tensor::DimOp>(op)) dimOps.erase(op);
  }

  void notifyOperationInserted(Operation *op) override {
    IRRewriter::notifyOperationInserted(op);
    if (isa<tensor::DimOp>(op)) dimOps.insert(op);
  }

 private:
  SmallPtrSet<Operation *, 16> dimOps;
};
}  // namespace

/// Simplfy the given tensor::DimOps as much as possible.
/// * Static dimensions are replaced by constant.
/// * Dynamic dim ops are pushed as much as possible to the top of the function,
///   i.e., if the dim of a value is known to be equal to the dim of a value on
///   the reverse SSA use-def chain, rewrite the value with a dim op of that
///   value.
static LogicalResult simplifyDimOps(RewriterBase &rewriter,
                                    const SmallVector<tensor::DimOp> &dimOps) {
  for (tensor::DimOp dimOp : dimOps) {
    // Only DimOps with static indices are supported.
    Optional<int64_t> idx = dimOp.getConstantIndex();
    if (!idx.has_value()) continue;
    // Only DimOps with ranked tensors are supported.
    auto tensorType = dimOp.getSource().getType().dyn_cast<RankedTensorType>();
    if (!tensorType) continue;

    if (!tensorType.isDynamicDim(*idx)) {
      // Rewrite static dimension with constant.
      int64_t size = tensorType.getShape()[*idx];
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(dimOp, size);
      continue;
    }

    // Try to simplify dynamic dims.
    SmallVector<Value> dynamicDims;
    if (failed(Flow::reifyDynamicResultDims(rewriter, dimOp.getSource(),
                                            dynamicDims)))
      return failure();
    unsigned ctr = 0;
    for (int64_t i = 0; i < *dimOp.getConstantIndex(); ++i)
      if (tensorType.isDynamicDim(i)) ++ctr;
    rewriter.replaceOp(dimOp, dynamicDims[ctr]);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Root and fusion group attribute handling
//===----------------------------------------------------------------------===//

/// Returns true if an op has a root operation.
static bool hasRootOpAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<IntegerAttr>(kRootOpAttr));
}

/// Removes root attribute. Asserts if root attribute is not present.
static void removeRootOpAttribute(Operation *op) {
  op->removeAttr(kRootOpAttr);
}

/// Sets the root attribute for an operation. The root attribute needs a number
/// to identify the root. Asserts if root attribute is already set on an
/// operation.
static void setRootAttribute(MLIRContext *context, Operation *op,
                             int64_t rootNumber) {
  assert(!op->hasAttr(kRootOpAttr) &&
         "invalid to update root attribute on an op");
  op->setAttr(kRootOpAttr,
              IntegerAttr::get(IntegerType::get(context, 64), rootNumber));
}

/// Returns the number of the root. Asserts if the operation is not already set
/// as a root.
static int64_t getRootNumber(Operation *op) {
  return op->getAttrOfType<IntegerAttr>(kRootOpAttr).getInt();
}

/// Returns true if an op is part of a fusion group.
static bool hasFusionGroupsAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr));
}

/// Returns the fusion groups for the given `op`.
static SmallVector<int64_t, 1> getFusionGroups(Operation *op) {
  SmallVector<int64_t, 1> fusionGroups = {};
  if (auto fusionGroupsAttr = op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr)) {
    fusionGroups = llvm::to_vector<1>(llvm::map_range(
        fusionGroupsAttr,
        [](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
  }
  return fusionGroups;
}

/// Appends the given `op` to the `newGroups` fusion groups.
static void appendToFusionGroup(Operation *op, ArrayRef<int64_t> newGroups) {
  SmallVector<int64_t, 1> fusionGroups = getFusionGroups(op);
  fusionGroups.append(newGroups.begin(), newGroups.end());
  op->setAttr(kFusionGroupsAttr, Builder(op).getI64ArrayAttr(fusionGroups));
}

/// Returns true if the given `op` is in the `targetGroup` fusion group.
static bool isInFusionGroup(Operation *op, unsigned targetGroup) {
  if (ArrayAttr opGroupAttr = op->getAttrOfType<ArrayAttr>(kFusionGroupsAttr)) {
    return llvm::any_of(opGroupAttr, [&targetGroup](Attribute attr) {
      return attr.cast<IntegerAttr>().getInt() == targetGroup;
    });
  }
  return false;
}

/// Removes the fusion groups attribute.
static void removeFusionGroupsAttribute(Operation *op) {
  op->removeAttr(kFusionGroupsAttr);
}

//===----------------------------------------------------------------------===//
// Op property charecterizations
//===----------------------------------------------------------------------===//

/// Operations that are treated as root operations for dispatch region
/// formation.
static bool isRootOp(Operation *op) {
  if (op->getParentOfType<IREE::Flow::DispatchRegionOp>() ||
      op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
    return false;
  }
  // Any Linalg named op or generic op with reduction iterator types is a root
  // op.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (isa<linalg::GenericOp>(op)) {
      return linalgOp.getNumReductionLoops() != 0;
    }
    return !isa<linalg::FillOp>(op);
  }
  return isa<TilingInterface>(op);
}

/// Operations that are cloned into dispatch regions formed with other
/// operations as roots.
bool isClonableIntoDispatchOp(Operation *op) {
  // TODO(#8637): `tensor.collapse_shape` and `tensor.expand_shape` are
  // trivially clonable too, but they cause problems
  // with bufferization. Make them clonable when fixed.
  if (isa<arith::IndexCastOp, tensor::EmptyOp, tensor::CastOp,
          tensor::ExtractOp, tensor::ExtractSliceOp, tensor::PadOp>(op)) {
    return true;
  }
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    auto constantValueAttr = constantOp.getValue();
    auto constantType = constantOp.getType();
    if (constantValueAttr.isa<SplatElementsAttr>()) {
      return true;
    } else if (auto denseAttr =
                   constantValueAttr.dyn_cast<DenseElementsAttr>()) {
      auto shapedType = constantOp.getType().cast<ShapedType>();
      uint64_t estimatedByteLength =
          (shapedType.getNumElements() * shapedType.getElementTypeBitWidth()) /
          8;
      return denseAttr.isSplat() ||
             estimatedByteLength <= kInlineConstantByteLength;
    } else if (constantType.isIntOrIndexOrFloat()) {
      return true;
    }
  }
  if (llvm::all_of(op->getOperands(),
                   [&](Value v) { return v.getType().isIntOrFloat(); }) &&
      llvm::all_of(op->getResults(),
                   [&](Value v) { return v.getType().isIntOrFloat(); })) {
    return true;
  }
  return false;
}

/// Checks if the `Value` has a use within the dispatch that is unfusable.
static bool hasUnfusableUseInDispatch(Value v, Operation *dispatchOp) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    Operation *ownerWorkgroups =
        user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
    Operation *ownerRegion =
        user->getParentOfType<IREE::Flow::DispatchRegionOp>();
    Operation *owner = ownerWorkgroups ? ownerWorkgroups : ownerRegion;

    // Ignore uses outside of dispatch workgroups op.
    if (owner != dispatchOp) continue;

    // Cannot fuse producer of `dest` with `tensor.insert_slice`.
    if (auto insertSliceUser = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (insertSliceUser.getDest() == v) return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Methods for getting the workload information for dispatch region creation.
//===----------------------------------------------------------------------===//

/// Compute the workload to use for the workgroup based on the root op.
static SmallVector<Value> getWorkloadForRootOp(OpBuilder &builder,
                                               Operation *rootOp) {
  // Compute workgroup count to use for the dispatch op. These are the ranges
  // of the outermost parallel loops that can be distributed.
  Location loc = rootOp->getLoc();
  SmallVector<Range> loopRanges = Flow::getLoopRanges(rootOp, loc, builder);
  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineMap workload = AffineMap::get(0, 3, (s1 - s0).ceilDiv(s2));
  return llvm::to_vector(llvm::map_range(loopRanges, [&](Range r) -> Value {
    Value offset = getValueOrCreateConstantIndexOp(builder, loc, r.offset);
    Value size = getValueOrCreateConstantIndexOp(builder, loc, r.size);
    Value stride = getValueOrCreateConstantIndexOp(builder, loc, r.stride);
    return builder.create<AffineApplyOp>(rootOp->getLoc(), workload,
                                         ValueRange{offset, size, stride});
  }));
}

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

/// Collect all ops that should be cloned into the given dispatch region op.
static SmallVector<Operation *> getCloneableOps(
    Flow::DispatchRegionOp regionOp) {
  // Find values that are used inside of the dispatch region but defined outside
  // of the dispatch region.
  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(regionOp.getBody(), valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return {};

  // Traverse the defining ops of these values (and the ops on their reverse
  // SSA use-def chain).
  SmallVector<Operation *> result;
  llvm::SetVector<Value> visited;
  SmallVector<Value, 4> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    // Skip values that were already visited.
    if (visited.count(outsideValue)) continue;
    visited.insert(outsideValue);

    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp || !(isClonableIntoDispatchOp(definingOp)) ||
        hasUnfusableUseInDispatch(outsideValue, regionOp)) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    result.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
  }

  return result;
}

static bool areLinalgOpsFusableUsingTileAndFuse(OpOperand &use) {
  auto producer = use.get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!producer || !consumer) return false;

  // 1. Producer has a single result.
  if (producer->getNumResults() != 1) return false;

  // 2. Consumer is elementwise parallel.
  if (consumer.getNumLoops() != consumer.getNumParallelLoops()) return false;

  // 3. Check if a reduction result is used in the following elementwise
  // operation with broadcast. If so, we can fuse the reduction into the
  // elementwise op. The elementwise op on the reduced dimension will be
  // serialized to match the workgroup counts of the fused operations.
  // Otherwise, check if the result of producer is accessed using identity
  // indexing.
  AffineMap consumerIndexingMap = consumer.getMatchingIndexingMap(&use);
  if (!consumerIndexingMap.isIdentity()) {
    return false;
  }
  return true;
}

/// Checks if the producer and consumer LinalgOps can be fused.
static bool areFusableLinalgOps(OpOperand &use) {
  return areLinalgOpsFusableUsingTileAndFuse(use);
}

/// Returns true if this is a fusable use.
static bool isFusableWithConsumer(OpOperand &use) {
  // Check for linalg producer -> consumer fusion with tile + fuse.
  return areFusableLinalgOps(use);
}

/// For all uses of an operation, finds the use that dominates all other uses.
static Optional<OpOperand *> getFusableUse(Operation *op,
                                           DominanceInfo const &dominanceInfo) {
  if (!kEnableMultiResultDispatches) {
    if (op->hasOneUse()) {
      OpOperand &use = *(op->use_begin());
      return &use;
    }
    return llvm::None;
  }
  for (auto &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (llvm::all_of(op->getUsers(), [&](Operation *c) {
          return dominanceInfo.dominates(user, c);
        })) {
      return &use;
    }
  }
  return llvm::None;
}

/// Fuses roots with its consumers. If a root is fused with its consumer, it is
/// no more tagged as a root to aid with the dispatch region formation.
static void fuseRootsWithConsumers(MLIRContext *context,
                                   ArrayRef<Operation *> roots,
                                   DominanceInfo const &dominanceInfo) {
  SmallVector<Operation *> workList(roots.begin(), roots.end());
  // Fuse with consumers where possible.
  while (!workList.empty()) {
    Operation *currRoot = workList.pop_back_val();
    assert(hasRootOpAttribute(currRoot) &&
           "unexpected non-root op in worklist");

    // Helper function to make the consumer the root instead of the producer
    // when they are to be fused.
    auto updateRootTo = [&context, &currRoot](Operation *newRoot) {
      int64_t rootNumber = getRootNumber(currRoot);
      setRootAttribute(context, newRoot, rootNumber);
      removeRootOpAttribute(currRoot);
      appendToFusionGroup(currRoot, rootNumber);
    };

    Optional<OpOperand *> fusableUse = getFusableUse(currRoot, dominanceInfo);
    if (!fusableUse) continue;

    // Analyse the use to see if it is fusable.
    Operation *consumerOp = fusableUse.value()->getOwner();
    if (hasRootOpAttribute(consumerOp) ||
        hasFusionGroupsAttribute(consumerOp)) {
      continue;
    }

    if (isFusableWithConsumer(*(fusableUse.value()))) {
      updateRootTo(consumerOp);
      workList.push_back(consumerOp);
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool isFusableWithProducer(OpOperand &operand) {
  Operation *producer = operand.get().getDefiningOp();
  Operation *consumer = operand.getOwner();

  if (isa<linalg::LinalgOp>(consumer) && isa<linalg::LinalgOp>(producer)) {
    auto consumerLinalgOp = cast<linalg::LinalgOp>(consumer);
    auto producerLinalgOp = cast<linalg::LinalgOp>(producer);
    if (consumerLinalgOp.isDpsInit(&operand) &&
        producerLinalgOp.getNumLoops() ==
            producerLinalgOp.getNumParallelLoops()) {
      return true;
    }
  }
  return false;
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void fuseRootsWithProducers(MLIRContext *context, Operation *root,
                                   unsigned groupNum,
                                   DominanceInfo const &dominanceInfo) {
  // We probably want a worklist algorithm here, but for now just look at
  // immediate producers.
  for (OpOperand &operand : root->getOpOperands()) {
    Operation *producer = operand.get().getDefiningOp();
    if (!producer) continue;
    if (hasFusionGroupsAttribute(producer) || hasRootOpAttribute(producer)) {
      continue;
    }

    Optional<OpOperand *> fusableUse = getFusableUse(producer, dominanceInfo);
    if (!fusableUse || fusableUse.value()->getOwner() != root) continue;

    if (isFusableWithProducer(operand)) {
      appendToFusionGroup(producer, groupNum);
    }
  }
}

/// Some heuristic is needed to fuse a dispatchable op with root operations
/// using tile + fuse. Using some heuristic, each root operation is tagged with
/// an ID (using an IntegerAttr with name `kRootOpAttr`) and all dispatchable
/// ops to be fused with it is tagged with the same ID (using a list of
/// IntegerAttr with name `kFusionGroupsAttr`). Each dispatchable operation can
/// be marked to fuse with multiple root operations (i.e. replicated). For now a
/// very simple heuristic is used below, but the mechanism should be general
/// enough to capture any heuristic.
static unsigned decideFusableLinalgOps(FunctionOpInterface funcOp,
                                       DominanceInfo const &dominanceInfo) {
  unsigned numRootOps = 0;
  MLIRContext *context = funcOp->getContext();
  OpBuilder builder(context);
  for (Block &block : funcOp.getFunctionBody()) {
    // Dispatch region formation works by first cloning the root into
    // the dispatch region and then pulling operations in.
    // So procedure here is to
    // - First find the roots
    // - To fuse with consumers make the consumer the root.
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // Start with a root operation and fuse its producers.
      if (hasFusionGroupsAttribute(&op) || !isRootOp(&op)) continue;
      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo);
  }

  // Once all root linalg ops have been tagged, put all remaining generic ops
  // into their own dispatches.
  for (Block &block : funcOp.getFunctionBody()) {
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // If it is part of a fusion group or root op, ignore it.
      if (hasFusionGroupsAttribute(&op) || hasRootOpAttribute(&op)) continue;
      // Only look for Linalg ops here. Avoid moving `linalg.fill` that aren't
      // fused with anything else into their own dispatches since it is better
      // to convert them to splats.
      if (!isa<linalg::LinalgOp>(op) || isa<linalg::FillOp>(op)) continue;

      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After annotating linalg op fusion scheme ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  return numRootOps;
}

//===----------------------------------------------------------------------===//
// Dispatch region formation
//===----------------------------------------------------------------------===//

/// Clone producers into the dispatch region.
static LogicalResult cloneProducers(RewriterBase &rewriter,
                                    Flow::DispatchRegionOp regionOp) {
  SmallVector<Operation *> cloneableOps = getCloneableOps(regionOp);
  bool sortResult = mlir::computeTopologicalSorting(cloneableOps);
  (void)sortResult;
  assert(sortResult && "could not compute topological sorting");

  for (Operation *producer : llvm::reverse(cloneableOps))
    if (failed(
            clonePrecedingOpIntoDispatchRegion(rewriter, producer, regionOp)))
      return failure();

  return success();
}

/// Helper function that builds the workload region body.
static void buildWorkloadRegionBody(OpBuilder &builder, Location loc,
                                    ArrayRef<BlockArgument> args) {
  auto numWorkgroupsOp =
      builder.create<Flow::DispatchWorkgroupCountFromDagRootOp>(loc, args);
  builder.create<Flow::ReturnOp>(loc, numWorkgroupsOp.getResults());
}

/// Create Flow::DispatchGroupsOps based on a fusion heuristic.
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> createFusionGroups(
    TensorDimTrackingRewriter &rewriter, FunctionOpInterface funcOp,
    DominanceInfo const &dominanceInfo, bool generateWorkloadRegion) {
  // Decide fusion groups (heuristic).
  unsigned numRoots = decideFusableLinalgOps(funcOp, dominanceInfo);
  SmallVector<Operation *> roots(numRoots, nullptr);
  DenseMap<unsigned, SmallVector<Operation *>> producers;

  // TODO: Incrementally add ops to an empty DispatchGroupOp instead of
  // annotating fusion group IDs via attributes.
  funcOp.walk([&](Operation *op) {
    if (hasRootOpAttribute(op)) roots[getRootNumber(op)] = op;
    if (hasFusionGroupsAttribute(op)) {
      assert(getFusionGroups(op).size() == 1 && "expected exactly one group");
      producers[getFusionGroups(op).front()].push_back(op);
    }
  });

  // Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Flow::DispatchRegionOp> regionOps;
  DenseMap<Flow::DispatchRegionOp, SmallVector<Value>> workloads;
  for (const auto &it : llvm::enumerate(roots)) {
    // Compute workload.
    SmallVector<Value> workload;
    if (generateWorkloadRegion) {
      rewriter.setInsertionPoint(it.value());
      FailureOr<SmallVector<Value>> maybeWorkload =
          getWorkloadForRootOp(rewriter, it.value());
      if (failed(maybeWorkload)) return failure();
      workload = *maybeWorkload;
    }

    // Simplify tensor::DimOps.
    SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
    if (failed(simplifyDimOps(rewriter, dimOps))) return failure();

    // Create fusion group.
    Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, it.value());
    if (failed(maybeRegionOp)) return failure();
    regionOp = *maybeRegionOp;
    workloads[regionOp] = workload;

    // Sort producers topologically. All producers must be in the same block as
    // the root.
    bool sortResult = mlir::computeTopologicalSorting(producers[it.index()]);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    // Move ops into the region.
    for (Operation *producer : llvm::reverse(producers[it.index()])) {
      auto newRegionOp =
          movePrecedingOpIntoDispatchRegion(rewriter, producer, regionOp);
      if (failed(newRegionOp)) return failure();
      regionOp = *newRegionOp;
    }

    regionOps.push_back(regionOp);
  }

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  SmallVector<Flow::DispatchWorkgroupsOp> result;
  for (auto regionOp : regionOps) {
    if (failed(cloneProducers(rewriter, regionOp))) return failure();
    auto maybeWorkgroupOp =
        Flow::rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
            regionOp, rewriter, workloads[regionOp],
            generateWorkloadRegion ? buildWorkloadRegionBody : nullptr);
    if (failed(maybeWorkgroupOp)) return failure();

    result.push_back(*maybeWorkgroupOp);
  }

  return result;
}

/// Wrap a single op in a DispatchWorkgroupsOp.
static FailureOr<Flow::DispatchWorkgroupsOp> wrapInWorkgroupsOp(
    TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Compute workload.
  SmallVector<Value> workload;
  if (generateWorkloadRegion) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    FailureOr<SmallVector<Value>> maybeWorkload =
        getWorkloadForRootOp(rewriter, op);
    if (failed(maybeWorkload)) return failure();
    workload = *maybeWorkload;
  }

  // Simplify tensor::DimOps.
  SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
  if (failed(simplifyDimOps(rewriter, rewriter.getTensorDimOps())))
    return failure();

  // Wrap operation.
  auto regionOp = Flow::wrapOpInDispatchRegion(rewriter, op);
  if (failed(regionOp)) return failure();
  if (failed(cloneProducers(rewriter, *regionOp))) return failure();
  auto workgroupsOp = Flow::rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
      *regionOp, rewriter, workload,
      generateWorkloadRegion ? buildWorkloadRegionBody : nullptr);
  if (failed(workgroupsOp)) return failure();
  return *workgroupsOp;
}

/// Wrap all given ops in a DispatchWorkgroupsOp.
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> wrapInWorkgroupsOp(
    TensorDimTrackingRewriter &rewriter, SmallVector<Operation *> rootOps,
    bool generateWorkloadRegion) {
  SmallVector<Flow::DispatchWorkgroupsOp> result;
  for (Operation *rootOp : rootOps) {
    auto workgroupsOp =
        wrapInWorkgroupsOp(rewriter, rootOp, generateWorkloadRegion);
    if (failed(workgroupsOp)) return failure();
    result.push_back(*workgroupsOp);
  }
  return result;
}

/// Wrap all ops of the given type that are direct children of the given op in
/// a DispatchWorkgroupsOp.
template <typename OpTy>
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> wrapInWorkgroupsOp(
    TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Find ops of type OpTy.
  SmallVector<Operation *> rootOps;
  for (Region &r : op->getRegions())
    for (Block &b : r.getBlocks())
      for (auto op : b.getOps<OpTy>()) rootOps.push_back(op.getOperation());

  // Wrap ops in DispatchWorkgroupsOps.
  return wrapInWorkgroupsOp(rewriter, rootOps, generateWorkloadRegion);
}

namespace {
/// Pass declaration.
struct DispatchLinalgOnTensorsViaRegionOpsPass
    : public Flow::DispatchLinalgOnTensorsViaRegionOpsBase<
          DispatchLinalgOnTensorsViaRegionOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }
  DispatchLinalgOnTensorsViaRegionOpsPass(bool generateWorkloadRegion) {
    this->generateWorkloadRegion = generateWorkloadRegion;
  }
  DispatchLinalgOnTensorsViaRegionOpsPass(
      const DispatchLinalgOnTensorsViaRegionOpsPass &pass) {
    this->generateWorkloadRegion = pass.generateWorkloadRegion;
  }
  void runOnOperation() override;

 private:
  bool generateWorkloadRegion = true;
};
}  // namespace

void DispatchLinalgOnTensorsViaRegionOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();

  DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
  TensorDimTrackingRewriter rewriter(funcOp);

  // Step 1: Create a DispatchWorkgroupsOp for every fusion group.
  auto maybeWorkgroupsOps = createFusionGroups(rewriter, funcOp, dominanceInfo,
                                               generateWorkloadRegion);
  if (failed(maybeWorkgroupsOps)) return signalPassFailure();
  SmallVector<Flow::DispatchWorkgroupsOp> workgroupsOps = *maybeWorkgroupsOps;

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After first step of dispatch region formation ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Step 2a: Rewrite InsertSliceOps to TensorUpdateOps.
  SmallVector<tensor::InsertSliceOp> insertSliceOps;
  SmallVector<Operation *> remainingInsertSliceOps;
  funcOp.walk([&](tensor::InsertSliceOp op) {
    if (!op->getParentOfType<Flow::DispatchRegionOp>())
      insertSliceOps.push_back(op);
  });
  for (tensor::InsertSliceOp insertSliceOp : insertSliceOps)
    if (failed(
            Flow::convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp)))
      remainingInsertSliceOps.push_back(insertSliceOp);

  // Step 2b: Create a DispatchWorkgroupsOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingInsertSliceOps,
                         generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return signalPassFailure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  // Step 3: Create a DispatchWorkgroupsOp for every remaining ExtractSliceOp.
  newWorkgroupsOps = wrapInWorkgroupsOp<tensor::ExtractSliceOp>(
      rewriter, funcOp, generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return signalPassFailure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  // A few extra canonicalizations/lowerings.
  {
    RewritePatternSet convertToFlowPatterns(context);
    Flow::populateTensorToFlowConversionPatterns(context,
                                                 convertToFlowPatterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(
        convertToFlowPatterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(
        convertToFlowPatterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(convertToFlowPatterns))))
      return signalPassFailure();

    // Finally fold `tensor.insert_slice/extract_slice` operations with
    // `flow.dispatch.tensor.load/store`.
    RewritePatternSet foldExtractInsertSliceOps(context);
    Flow::populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
        foldExtractInsertSliceOps, context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(foldExtractInsertSliceOps))))
      return signalPassFailure();
  }

  // Finally walk all the ops and remove the attributes
  funcOp.walk([](Operation *op) {
    removeFusionGroupsAttribute(op);
    removeRootOpAttribute(op);
    op->removeAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
Flow::createDispatchLinalgOnTensorsViaRegionOpsPass(
    bool generateWorkloadRegion) {
  return std::make_unique<DispatchLinalgOnTensorsViaRegionOpsPass>(
      generateWorkloadRegion);
}
