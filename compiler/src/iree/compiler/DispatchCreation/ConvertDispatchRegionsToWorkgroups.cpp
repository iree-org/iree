// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-convert-dispatch-regions-to-workgroups"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTDISPATCHREGIONSTOWORKGROUPSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertDispatchRegionsToWorkgroupsPass
    : impl::ConvertDispatchRegionsToWorkgroupsPassBase<
          ConvertDispatchRegionsToWorkgroupsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Follow updates that can reuse a Sort result without preserving an earlier
// tensor version. Export the last version so the store does not itself create a
// read-after-write conflict with an otherwise in-place consumer.
static Value getWorkspaceResult(Value value, Block &body) {
  while (true) {
    OpOperand *consumer = nullptr;
    for (OpOperand &use : value.getUses()) {
      if (isa<tensor::DimOp>(use.getOwner())) {
        continue;
      }
      if (consumer) {
        return value;
      }
      consumer = &use;
    }
    if (!consumer || consumer->getOwner()->getBlock() != &body) {
      return value;
    }
    auto dps = dyn_cast<DestinationStyleOpInterface>(consumer->getOwner());
    auto tied = dyn_cast<IREE::Util::TiedOpInterface>(consumer->getOwner());
    if (!dps || !tied || !dps.hasPureTensorSemantics() ||
        !dps.isDpsInit(consumer)) {
      return value;
    }
    Value next = dps.getTiedOpResult(consumer);
    if (next.getType() != value.getType() ||
        tied.getTiedResultOperand(next) != value) {
      return value;
    }
    value = next;
  }
}

// Sort reads and writes its outputs while sorting, even when their final values
// are only used inside the dispatch. Such outputs need a writable binding: an
// unreturned tensor.empty would otherwise become a private kernel allocation
// covering the full sorting axis, which may be unbounded. Sort initializes
// these workspaces from its inputs; dispatch creation only supplies the
// storage.
static LogicalResult
materializeSortWorkspaces(IREE::Flow::DispatchRegionOp &regionOp,
                          RewriterBase &rewriter) {
  if (!llvm::hasSingleElement(regionOp.getBody())) {
    return success();
  }
  Block &body = regionOp.getBody().front();
  if (body.getOps<IREE::LinalgExt::SortOp>().empty()) {
    return success();
  }
  eliminateTriviallyDeadOps(rewriter, regionOp.getBody());
  auto returnOp = cast<IREE::Flow::ReturnOp>(body.getTerminator());
  SmallVector<Value> workspaceResults;
  SmallVector<SmallVector<Value>> workspaceDims;
  OpBuilder::InsertionGuard guard(rewriter);
  // Materialize consumers first: replacing a later Sort's destination can
  // otherwise disconnect an earlier workspace from its exported result.
  auto sortOps = llvm::to_vector(body.getOps<IREE::LinalgExt::SortOp>());
  for (auto sortOp : llvm::reverse(sortOps)) {
    // Only a dynamic sorting axis requires unbounded full-axis storage. Leave
    // fixed sorting axes to ordinary tiling and private storage allocation.
    if (!sortOp.getOperandType(0).isDynamicDim(sortOp.getDimension())) {
      continue;
    }
    auto comparatorArgs = sortOp.getRegion().front().getArguments();
    for (OpResult result : sortOp.getResults()) {
      unsigned index = result.getResultNumber();
      if (result.use_empty() && comparatorArgs[2 * index].use_empty() &&
          comparatorArgs[2 * index + 1].use_empty()) {
        continue;
      }
      Value workspaceResult = getWorkspaceResult(result, body);
      if (llvm::is_contained(returnOp.getOperands(), workspaceResult)) {
        continue;
      }
      rewriter.setInsertionPoint(sortOp);
      SmallVector<Value> dims;
      if (failed(IREE::Flow::getOptimizedDynamicResultDims(rewriter, result,
                                                           dims)) ||
          llvm::any_of(dims, [&](Value dim) {
            return regionOp.getBody().isAncestor(dim.getParentRegion());
          })) {
        continue;
      }
      rewriter.setInsertionPoint(regionOp);
      auto allocation = IREE::Flow::TensorAllocaOp::create(
          rewriter, sortOp.getLoc(), cast<RankedTensorType>(result.getType()),
          dims);
      rewriter.modifyOpInPlace(sortOp, [&]() {
        sortOp.getDpsInitOperand(index)->set(allocation.getResult());
      });
      workspaceResults.push_back(workspaceResult);
      workspaceDims.push_back(std::move(dims));
    }
  }
  if (workspaceResults.empty()) {
    return success();
  }
  auto dialectAttrs = llvm::to_vector(regionOp->getDialectAttrs());
  Region workgroupCount;
  workgroupCount.takeBody(regionOp.getWorkgroupCount());
  auto newRegionOp = IREE::Flow::appendDispatchRegionResults(
      rewriter, regionOp, workspaceResults, workspaceDims);
  if (failed(newRegionOp)) {
    return failure();
  }
  (*newRegionOp)->setDialectAttrs(dialectAttrs);
  newRegionOp->getWorkgroupCount().takeBody(workgroupCount);
  regionOp = *newRegionOp;
  return success();
}

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToWorkgroupsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  TensorDimTrackingRewriter rewriter(funcOp);

  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk(
      [&](IREE::Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  numDispatches += regionOps.size();

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    if (failed(materializeSortWorkspaces(regionOp, rewriter))) {
      return signalPassFailure();
    }
    auto maybeWorkgroupOp =
        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      regionOp.emitError(
          "failed to convert dispatch.region op to dispatch.workgroup op");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
