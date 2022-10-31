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
#include "iree/compiler/Dialect/Flow/Transforms/DispatchRegionHeuristic.h"
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
// Op property charecterizations
//===----------------------------------------------------------------------===//

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
    if (!definingOp || !(Flow::isClonableIntoDispatchOp(definingOp)) ||
        hasUnfusableUseInDispatch(outsideValue, regionOp)) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    result.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
  }

  return result;
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
    TensorDimTrackingRewriter &rewriter,
    const Flow::FusionGroupMapping &fusionGroups, FunctionOpInterface funcOp,
    DominanceInfo const &dominanceInfo, bool generateWorkloadRegion) {
  // Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Flow::DispatchRegionOp> regionOps;
  DenseMap<Flow::DispatchRegionOp, SmallVector<Value>> workloads;
  for (const auto &it : fusionGroups) {
    Operation *rootOp = it.first;
    SmallVector<Operation *> producers = it.second;

    // Compute workload.
    SmallVector<Value> workload;
    if (generateWorkloadRegion) {
      rewriter.setInsertionPoint(rootOp);
      FailureOr<SmallVector<Value>> maybeWorkload =
          getWorkloadForRootOp(rewriter, rootOp);
      if (failed(maybeWorkload)) return failure();
      workload = *maybeWorkload;
    }

    // Simplify tensor::DimOps.
    SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
    if (failed(simplifyDimOps(rewriter, dimOps))) return failure();

    // Create fusion group.
    Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, rootOp);
    if (failed(maybeRegionOp)) return failure();
    regionOp = *maybeRegionOp;

    // Sort producers topologically. All producers must be in the same block as
    // the root.
    bool sortResult = mlir::computeTopologicalSorting(producers);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    // Move ops into the region.
    for (Operation *producer : llvm::reverse(producers)) {
      auto newRegionOp =
          movePrecedingOpIntoDispatchRegion(rewriter, producer, regionOp);
      if (failed(newRegionOp)) return failure();
      regionOp = *newRegionOp;
    }

    workloads[regionOp] = workload;
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

  // Step 0: Decide fusion groups (heuristic).
  Flow::FusionGroupMapping fusionGroups = Flow::decideFusableLinalgOps(
      funcOp, dominanceInfo, /*aggressiveFusion=*/false);

  // Step 1: Create a DispatchWorkgroupsOp for every fusion group.
  auto maybeWorkgroupsOps = createFusionGroups(
      rewriter, fusionGroups, funcOp, dominanceInfo, generateWorkloadRegion);
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
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
Flow::createDispatchLinalgOnTensorsViaRegionOpsPass(
    bool generateWorkloadRegion) {
  return std::make_unique<DispatchLinalgOnTensorsViaRegionOpsPass>(
      generateWorkloadRegion);
}
