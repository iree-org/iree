// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-flow-form-scalar-dispatches"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Pass declaration.
struct FormScalarDispatchesPass
    : public FormScalarDispatchesBase<FormScalarDispatchesPass> {
  using FormScalarDispatchesBase<
      FormScalarDispatchesPass>::FormScalarDispatchesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

/// Return true if type represents a value less than `n` elements.
static bool isScalarOrTensorOfLinearSizeN(int n, Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return true;
  }
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    if (!tensorType.hasStaticShape()) {
      return false;
    }
    return tensorType.getNumElements() <= n;
  }
  return false;
}

/// Return `true` for operations that are to be treated as compute operations.
static bool isComputeOperation(Operation *op) {
  MLIRContext *context = op->getContext();
  if (op->getDialect() == context->getLoadedDialect<linalg::LinalgDialect>()) {
    return true;
  }
  if (op->getDialect() == context->getLoadedDialect<tensor::TensorDialect>()) {
    return !isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::EmptyOp,
                tensor::ExpandShapeOp, tensor::PackOp, tensor::UnPackOp>(op);
  }
  return false;
}

/// Return `true` if the workload of this operation is less than `n`.
static bool isOperationWorkloadLessThanSizeN(int n, Operation *candidateOp) {
  return llvm::all_of(candidateOp->getOperands(),
                      [&](Value v) {
                        return isScalarOrTensorOfLinearSizeN(n, v.getType());
                      }) &&
         llvm::all_of(candidateOp->getResultTypes(), [&](Type t) {
           return isScalarOrTensorOfLinearSizeN(n, t);
         });
}

/// Return `true` is the operation is to be treated as a scalar operation
/// and moved into a scalar dispatch (not necessarily as the root of the
/// dispatch).
static bool isScalarOperation(int workload, Operation *op) {
  // 1. Ignore most operations. Only look for a whitelist set of operations.
  if (!isComputeOperation(op)) {
    return false;
  }

  // 2. Check that the workload of the operation is less then the limit
  if (!isOperationWorkloadLessThanSizeN(workload, op)) {
    return false;
  }

  // 3. Do not move operations that are cloned into the dispatch region.
  // TODO: This might prevent moving all scalar operations into dispatch
  // resulting in artifical splits. Revisit after more examples.
  return !isClonableIntoDispatchOp(op);
}

/// Given a `rootOp` return a DAG of the program that represents
/// operations that can be moved into a scalar dispatch with the `rootOp`
/// as the root of the DAG.
llvm::SetVector<Operation *> computeSliceToMoveIntoDispatch(
    int workload, Operation *rootOp,
    const llvm::DenseMap<Operation *, Operation *> &opToRootMap) {
  BackwardSliceOptions options;
  options.filter = [&](Operation *currentOp) {
    assert(currentOp && "current op is null");
    if (opToRootMap.count(currentOp)) {
      return false;
    }
    // Operations needs to be in the same block as `rootOp`.
    if (currentOp->getBlock() != rootOp->getBlock()) {
      return false;
    }

    if (!isScalarOperation(workload, currentOp)) {
      return false;
    }

    // All its uses must be in the `opToRootMap`, i.e. they are either
    // in the current dispatches, or those already formed.
    return llvm::all_of(currentOp->getUsers(), [&](Operation *user) {
      return opToRootMap.count(user);
    });
  };
  options.omitBlockArguments = true;
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(rootOp, &slice, options);
  return slice;
}

/// Return `true` if the op is to be treated as a root of a scalar dispatch.
static bool isSliceRoot(int workload, Operation *op) {
  return !op->getParentOfType<DispatchRegionOp>() &&
         isScalarOperation(workload, op);
}

// Form dispatch regions from slice of the operation.
static FailureOr<DispatchRegionOp>
formDispatchRegionFromSlice(RewriterBase &rewriter, Operation *rootOp,
                            ArrayRef<Operation *> slice) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(rootOp);
  FailureOr<DispatchRegionOp> dispatchRegionOp =
      wrapOpInDispatchRegion(rewriter, rootOp);
  if (failed(dispatchRegionOp)) {
    return rootOp->emitOpError("failed to form dispatch region with root op");
  }
  FailureOr<DispatchRegionOp> newDispatchOp =
      movePrecedingOpsIntoDispatchRegion(rewriter, slice,
                                         dispatchRegionOp.value());
  if (failed(newDispatchOp)) {
    return dispatchRegionOp.value()->emitOpError(
        "failed to move slice into op");
  }
  return newDispatchOp.value();
}

void FormScalarDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  int scalarWorkloadLimit = 1;
  // Convenient struct to hold all operations that need to be moved into a
  // descriptor.
  struct DispatchRegionDescriptor {
    Operation *rootOp;
    SmallVector<Operation *> fusedOps;
  };

  SmallVector<DispatchRegionDescriptor> dispatches;
  llvm::DenseMap<Operation *, Operation *> opToRootMap;

  // Walk the function in postorder, reverse orded ignore all operations
  // not immediately nested within the `funcOp`.
  funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (op->getParentOp() != funcOp || opToRootMap.count(op)) {
      return;
    }

    if (!isSliceRoot(scalarWorkloadLimit, op)) {
      return;
    }

    llvm::SetVector<Operation *> fusedOpsSet =
        computeSliceToMoveIntoDispatch(scalarWorkloadLimit, op, opToRootMap);
    for (Operation *sliceOp : fusedOpsSet) {
      assert(!opToRootMap.count(sliceOp) &&
             "trying to add same op to two dispatches");
      opToRootMap[sliceOp] = op;
    }

    // Iterate backwards within the block to get ops that dont necessarily
    // have producer -> consumer relationship but can still be fused.
    Block *currBlock = op->getBlock();
    Operation *prevOp = op;
    bool didHorizontalFusion = false;
    while (prevOp != &currBlock->front()) {
      prevOp = prevOp->getPrevNode();

      if (opToRootMap.count(prevOp)) {
        continue;
      }

      if (!isSliceRoot(scalarWorkloadLimit, prevOp)) {
        if (isClonableIntoDispatchOp(prevOp)) {
          continue;
        }
        break;
      }

      didHorizontalFusion = true;
      fusedOpsSet.insert(prevOp);
      opToRootMap[prevOp] = op;
      llvm::SetVector<Operation *> currSlice = computeSliceToMoveIntoDispatch(
          scalarWorkloadLimit, prevOp, opToRootMap);
      for (auto sliceOp : currSlice) {
        assert(!opToRootMap.count(sliceOp) &&
               "trying to add same op to two dispatches");
        opToRootMap[sliceOp] = op;
      }
      fusedOpsSet.insert(currSlice.begin(), currSlice.end());
    }

    DispatchRegionDescriptor &currDispatch =
        dispatches.emplace_back(DispatchRegionDescriptor{});
    currDispatch.rootOp = op;
    currDispatch.fusedOps.assign(fusedOpsSet.begin(), fusedOpsSet.end());
    if (didHorizontalFusion) {
      mlir::computeTopologicalSorting(currDispatch.fusedOps);
    }
  });

  LLVM_DEBUG({
    llvm::dbgs() << "Num scalar dispatches : " << dispatches.size() << "\n";
    for (auto [index, dispatch] : llvm::enumerate(dispatches)) {
      llvm::dbgs() << "//--------------------------//\n";
      llvm::dbgs() << "Dispatch : " << index << ", Root :";
      dispatch.rootOp->print(llvm::dbgs());
      llvm::dbgs() << "\nFusedOps :";
      for (auto fusedOp : dispatch.fusedOps) {
        fusedOp->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
      llvm::dbgs() << "//--------------------------//\n";
    }
  });

  IRRewriter rewriter(context);
  for (auto &currDispatch : dispatches) {
    rewriter.setInsertionPoint(currDispatch.rootOp);
    FailureOr<DispatchRegionOp> dispatchRegionOp = formDispatchRegionFromSlice(
        rewriter, currDispatch.rootOp, currDispatch.fusedOps);
    if (failed(dispatchRegionOp)) {
      currDispatch.rootOp->emitOpError(
          "failed to form scalar dispatch region with operation as root");
      return signalPassFailure();
    }

    // Set the workgroup count to {1, 1, 1} since this is to be executed
    // sequentially (at leats for now)
    Region &countRegion = dispatchRegionOp->getWorkgroupCount();
    Block *countBody = rewriter.createBlock(&countRegion, countRegion.begin());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(countBody);
    auto one = rewriter.create<arith::ConstantIndexOp>(
        dispatchRegionOp.value()->getLoc(), 1);
    rewriter.create<Flow::ReturnOp>(dispatchRegionOp.value()->getLoc(),
                                    ValueRange{one, one, one});
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormScalarDispatchesPass() {
  return std::make_unique<FormScalarDispatchesPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
