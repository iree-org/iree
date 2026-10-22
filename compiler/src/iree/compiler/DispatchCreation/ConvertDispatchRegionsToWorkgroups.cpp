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
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

// Move pure initializer computation to its own dispatch so its tensor results
// already have managed storage when Sort needs to update them. Export every
// value still used by the original region rather than recomputing the slice.
static FailureOr<Value>
materializeSortInitializer(IREE::Flow::DispatchRegionOp regionOp, Value source,
                           RewriterBase &rewriter) {
  llvm::SetVector<Operation *> slice;
  BackwardSliceOptions options;
  options.inclusive = true;
  options.omitBlockArguments = true;
  options.omitUsesFromAbove = false;
  options.filter = [&](Operation *op) { return regionOp->isAncestor(op); };
  if (failed(getBackwardSlice(source, &slice, options)) ||
      llvm::any_of(slice, [&](Operation *op) {
        return op->getBlock() != &regionOp.getBody().front() || !isPure(op) ||
               isa<IREE::LinalgExt::SortOp>(op);
      })) {
    return failure();
  }
  SmallVector<Value> exports{source};
  for (Operation *op : slice) {
    for (Value result : op->getResults()) {
      if (result != source &&
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            while (user && user->getParentRegion() != &regionOp.getBody()) {
              user = user->getParentOp();
            }
            return !slice.contains(user);
          })) {
        exports.push_back(result);
      }
    }
  }
  SmallVector<Type> types;
  SmallVector<Value> resultDims;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(source.getDefiningOp());
  for (Value value : exports) {
    if (!isa<RankedTensorType>(value.getType())) {
      return failure();
    }
    SmallVector<Value> dims;
    if (failed(
            IREE::Flow::getOptimizedDynamicResultDims(rewriter, value, dims)) ||
        llvm::any_of(dims, [&](Value dim) {
          return regionOp.getBody().isAncestor(dim.getParentRegion());
        })) {
      return failure();
    }
    types.push_back(value.getType());
    resultDims.append(dims);
  }
  rewriter.setInsertionPoint(regionOp);
  auto initializer = IREE::Flow::DispatchRegionOp::create(
      rewriter, source.getLoc(), types, resultDims, ValueRange{});
  initializer->setDialectAttrs(llvm::to_vector(regionOp->getDialectAttrs()));
  rewriter.createBlock(&initializer.getBody());
  IRMapping mapping;
  for (Operation *op : slice) {
    rewriter.clone(*op, mapping);
  }
  auto returned = llvm::map_to_vector(
      exports, [&](Value value) { return mapping.lookup(value); });
  IREE::Flow::ReturnOp::create(rewriter, source.getLoc(), returned);
  auto initialized =
      rewriteFlowDispatchRegionToFlowDispatchWorkgroups(initializer, rewriter);
  if (failed(initialized)) {
    return failure();
  }
  for (auto [oldValue, newValue] :
       llvm::zip_equal(exports, initialized->getResults())) {
    rewriter.replaceAllUsesWith(oldValue, newValue);
  }
  for (Operation *op : llvm::reverse(slice)) {
    rewriter.eraseOp(op);
  }
  return initialized->getResult(0);
}

// Captured Sort lanes need initialized mutable storage even when their tensor
// results stay inside the dispatch. Initialize independent destinations before
// dispatch tiling instead of requiring writable aliases of the captured values.
static FailureOr<IREE::Flow::DispatchWorkgroupsOp>
convertDispatchRegionWithSortWorkspaces(IREE::Flow::DispatchRegionOp &regionOp,
                                        RewriterBase &rewriter) {
  if (!llvm::hasSingleElement(regionOp.getBody()) ||
      regionOp.getBody().front().getOps<IREE::LinalgExt::SortOp>().empty()) {
    return rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp,
                                                             rewriter);
  }
  eliminateTriviallyDeadOps(rewriter, regionOp.getBody());
  auto returnOp =
      cast<IREE::Flow::ReturnOp>(regionOp.getBody().front().getTerminator());
  // A later tied consumer can already yield the same working role. Do not
  // export an earlier tensor version again after the consumer updates it.
  llvm::DenseSet<Value> yieldedValues;
  for (Value value : returnOp.getOperands()) {
    while (regionOp.getBody().isAncestor(value.getParentRegion())) {
      yieldedValues.insert(value);
      auto tied = value.getDefiningOp<IREE::Util::TiedOpInterface>();
      if (!tied) {
        break;
      }
      value = tied.getTiedResultOperand(value);
      if (!value) {
        break;
      }
    }
  }
  SmallVector<Value> workspaceResults;
  SmallVector<SmallVector<Value>> workspaceDims;
  OpBuilder::InsertionGuard guard(rewriter);

  for (auto sortOp :
       regionOp.getBody().front().getOps<IREE::LinalgExt::SortOp>()) {
    bool hasLiveResult = llvm::any_of(sortOp.getResults(), [&](Value result) {
      return llvm::any_of(result.getUses(), [&](OpOperand &use) {
        if (isa<tensor::DimOp>(use.getOwner())) {
          return false;
        }
        return use.getOwner() != returnOp ||
               !regionOp->getResult(use.getOperandNumber()).use_empty();
      });
    });
    if (!hasLiveResult) {
      continue;
    }
    auto comparatorArgs = sortOp.getRegion().front().getArguments();
    for (auto result : sortOp.getResults()) {
      unsigned index = result.getResultNumber();
      bool hasDataUser = false;
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner() == returnOp) {
          hasDataUser |=
              !regionOp->getResult(use.getOperandNumber()).use_empty();
        } else if (!isa<tensor::DimOp>(use.getOwner())) {
          hasDataUser = true;
        }
      }
      if (!hasDataUser && comparatorArgs[2 * index].use_empty() &&
          comparatorArgs[2 * index + 1].use_empty()) {
        continue;
      }
      OpOperand &init = *sortOp.getDpsInitOperand(index);
      Value source = init.get();
      if (regionOp.getBody().isAncestor(source.getParentRegion())) {
        auto sourceResult = dyn_cast<OpResult>(source);
        auto dps = dyn_cast_or_null<DestinationStyleOpInterface>(
            source.getDefiningOp());
        auto tied = dyn_cast_or_null<IREE::Util::TiedOpInterface>(
            source.getDefiningOp());
        if (sourceResult && dps && tied && dps.hasPureTensorSemantics() &&
            dps->getBlock() == &regionOp.getBody().front() &&
            isPure(dps.getOperation()) &&
            llvm::all_of(source.getUses(), [&](OpOperand &use) {
              return &use == &init || isa<tensor::DimOp>(use.getOwner());
            })) {
          OpOperand *destination = dps.getTiedOpOperand(sourceResult);
          if (isa_and_nonnull<tensor::EmptyOp>(
                  destination->get().getDefiningOp()) &&
              destination->get().getType() == source.getType() &&
              tied.getTiedResultOperand(source) == destination->get()) {
            SmallVector<Value> dims;
            rewriter.setInsertionPoint(sortOp);
            if (succeeded(IREE::Flow::getOptimizedDynamicResultDims(
                    rewriter, source, dims)) &&
                llvm::none_of(dims, [&](Value dim) {
                  return regionOp.getBody().isAncestor(dim.getParentRegion());
                })) {
              // An empty destination has no data to preserve. Keep the
              // initializer in place and give it storage managed outside
              // the kernel before Sort updates its result.
              rewriter.setInsertionPoint(regionOp);
              auto allocation = IREE::Flow::TensorAllocaOp::create(
                  rewriter, source.getLoc(),
                  cast<RankedTensorType>(source.getType()), dims);
              rewriter.modifyOpInPlace(dps.getOperation(), [&]() {
                destination->set(allocation.getResult());
              });
              if (!yieldedValues.contains(result)) {
                workspaceResults.push_back(result);
                workspaceDims.push_back(std::move(dims));
              }
              continue;
            }
          }
        }
        auto initialized =
            materializeSortInitializer(regionOp, source, rewriter);
        if (failed(initialized)) {
          // A uniquely chained update of an earlier Sort can keep using its
          // managed workspace when no earlier tensor version is observed.
          Value version = source;
          OpOperand *consumer = &init;
          bool reusesEarlierSort = false;
          while (auto versionResult = dyn_cast<OpResult>(version)) {
            if (llvm::any_of(version.getUses(), [&](OpOperand &use) {
                  return &use != consumer &&
                         !isa<tensor::DimOp>(use.getOwner());
                })) {
              break;
            }
            if (isa<IREE::LinalgExt::SortOp>(versionResult.getOwner())) {
              reusesEarlierSort = true;
              break;
            }
            auto dps =
                dyn_cast<DestinationStyleOpInterface>(versionResult.getOwner());
            if (!dps || !dps.hasPureTensorSemantics() ||
                !isPure(dps.getOperation())) {
              break;
            }
            consumer = dps.getTiedOpOperand(versionResult);
            if (!consumer) {
              break;
            }
            version = consumer->get();
          }
          if (reusesEarlierSort) {
            continue;
          }
          // Keep a local source in this dispatch when its initializer cannot
          // move. A separate initialized workspace lets bufferization preserve
          // the source's tensor version without an in-kernel allocation.
          auto type = dyn_cast<RankedTensorType>(source.getType());
          if (!type || type.hasStaticShape()) {
            continue;
          }
          SmallVector<Value> dims;
          if (failed(IREE::Flow::getOptimizedDynamicResultDims(rewriter, source,
                                                               dims)) ||
              llvm::any_of(dims, [&](Value dim) {
                return regionOp.getBody().isAncestor(dim.getParentRegion());
              })) {
            continue;
          }
          rewriter.setInsertionPoint(regionOp);
          auto allocation = IREE::Flow::TensorAllocaOp::create(
              rewriter, sortOp.getLoc(), type, dims);
          rewriter.setInsertionPoint(sortOp);
          auto copy = linalg::CopyOp::create(
              rewriter, sortOp.getLoc(), ValueRange{source},
              ValueRange{allocation.getResult()});
          rewriter.modifyOpInPlace(sortOp,
                                   [&]() { init.set(copy.getResult(0)); });
          if (!yieldedValues.contains(result)) {
            workspaceResults.push_back(result);
            workspaceDims.push_back(std::move(dims));
          }
          continue;
        }
        source = *initialized;
      }
      auto producer = source.getDefiningOp<IREE::Flow::DispatchWorkgroupsOp>();
      if (producer &&
          !cast<IREE::Util::TiedOpInterface>(producer.getOperation())
               .getTiedResultOperand(source) &&
          llvm::all_of(source.getUses(), [&](OpOperand &use) {
            return &use == &init || isa<tensor::DimOp>(use.getOwner());
          })) {
        // This untied producer owns its destination. A captured tensor can
        // still have caller-visible storage despite having one data use.
        if (!yieldedValues.contains(result)) {
          rewriter.setInsertionPoint(regionOp);
          workspaceResults.push_back(result);
          workspaceDims.push_back(IREE::Util::buildDynamicDimsForValue(
              sortOp.getLoc(), source, rewriter));
        }
        continue;
      }
      rewriter.replaceUsesWithIf(result, source, [](OpOperand &use) {
        return isa<tensor::DimOp>(use.getOwner());
      });
      rewriter.setInsertionPoint(regionOp);
      auto dims = IREE::Util::buildDynamicDimsForValue(sortOp.getLoc(), source,
                                                       rewriter);
      auto type = cast<RankedTensorType>(source.getType());
      // A clone is pure and equal clones may CSE. The allocation preserves the
      // identity of each working role before its initializing copy and Sort.
      auto allocation = IREE::Flow::TensorAllocaOp::create(
          rewriter, sortOp.getLoc(), type, dims);
      auto initializer = IREE::Flow::DispatchRegionOp::create(
          rewriter, sortOp.getLoc(), TypeRange{type}, dims, ValueRange{});
      initializer->setDialectAttrs(
          llvm::to_vector(regionOp->getDialectAttrs()));
      rewriter.createBlock(&initializer.getBody());
      auto copy =
          linalg::CopyOp::create(rewriter, sortOp.getLoc(), ValueRange{source},
                                 ValueRange{allocation.getResult()});
      IREE::Flow::ReturnOp::create(rewriter, sortOp.getLoc(),
                                   ValueRange{copy.getResult(0)});
      auto initialized = rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
          initializer, rewriter);
      if (failed(initialized)) {
        return failure();
      }
      rewriter.modifyOpInPlace(sortOp,
                               [&]() { init.set(initialized->getResult(0)); });
      if (!yieldedValues.contains(result)) {
        workspaceResults.push_back(result);
        workspaceDims.push_back(std::move(dims));
      }
    }
  }

  if (!workspaceResults.empty()) {
    // A hidden workspace represents the last uniquely updated version, not an
    // earlier value that bufferization would have to preserve with a copy.
    for (Value &result : workspaceResults) {
      while (true) {
        OpOperand *update = nullptr;
        for (OpOperand &use : result.getUses()) {
          if (isa<tensor::DimOp>(use.getOwner())) {
            continue;
          }
          if (update) {
            update = nullptr;
            break;
          }
          update = &use;
        }
        if (!update) {
          break;
        }
        auto dps = dyn_cast<DestinationStyleOpInterface>(update->getOwner());
        auto tied = dyn_cast<IREE::Util::TiedOpInterface>(update->getOwner());
        if (!dps || !tied || !dps.hasPureTensorSemantics() ||
            !dps.isDpsInit(update) ||
            update->getOwner()->getBlock() != &regionOp.getBody().front()) {
          break;
        }
        Value next = dps.getTiedOpResult(update);
        if (next.getType() != result.getType() ||
            tied.getTiedResultOperand(next) != result) {
          break;
        }
        result = next;
      }
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
  }
  return rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);
}

// Creates a DispatchWorkgroupsOp for every DispatchRegionOp.
void ConvertDispatchRegionsToWorkgroupsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  TensorDimTrackingRewriter rewriter(funcOp);

  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  funcOp.walk(
      [&](IREE::Flow::DispatchRegionOp op) { regionOps.push_back(op); });

  int64_t dispatchCountDelta = 0;
  funcOp.walk([&](IREE::Flow::DispatchWorkgroupsOp) { --dispatchCountDelta; });

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  for (auto regionOp : regionOps) {
    auto maybeWorkgroupOp =
        convertDispatchRegionWithSortWorkspaces(regionOp, rewriter);
    if (failed(maybeWorkgroupOp)) {
      regionOp.emitError(
          "failed to convert dispatch.region op to dispatch.workgroup op");
      return signalPassFailure();
    }
  }
  funcOp.walk([&](IREE::Flow::DispatchWorkgroupsOp) { ++dispatchCountDelta; });
  numDispatches += dispatchCountDelta;
}
} // namespace mlir::iree_compiler::DispatchCreation
