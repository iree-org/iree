// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fuse-horizontal-contractions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEHORIZONTALCONTRACTIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct FuseHorizontalContractionsPass final
    : public impl::FuseHorizontalContractionsPassBase<
          FuseHorizontalContractionsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

/// Helper method to check operations equivalence
static bool checkOperationEquivalence(Operation *lhsOp, Operation *rhsOp) {
  // During equivalence check, it would have been easier if `checkEquivalence`
  // would just use `OpOperands *`. Since it takes `Value`s for now, just
  // check that the values are the same as operands. This is potentially
  // making the match too broad, but is an OK work-around for now.
  // TODO(MaheshRavishankar): Fix upstream `checkEquivalence` signater in
  // `OperationEquivalence::isEquivalentTo`.
  llvm::SmallDenseSet<Value, 8> operands;
  operands.insert(lhsOp->operand_begin(), lhsOp->operand_end());
  operands.insert(rhsOp->operand_begin(), rhsOp->operand_end());

  llvm::DenseMap<Value, Value> equivalentValues;
  auto checkEquivalent = [&](Value lhsValue, Value rhsValue) {
    if (operands.contains(lhsValue) && operands.contains(rhsValue)) {
      return success();
    }
    return success(equivalentValues.lookup(lhsValue) == rhsValue ||
                   equivalentValues.lookup(rhsValue) == lhsValue);
  };
  auto markEquivalent = [&](Value v1, Value v2) { equivalentValues[v1] = v2; };
  return OperationEquivalence::isEquivalentTo(
      lhsOp, rhsOp, checkEquivalent, markEquivalent,
      /*flags=*/OperationEquivalence::IgnoreLocations);
}

/// Check that an operation is a `empty -> fill -> contraction`
static bool isEquivalentContractionOp(
    linalg::LinalgOp linalgOp,
    std::optional<linalg::LinalgOp> seedContractionOp = std::nullopt) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }
  if (seedContractionOp) {
    return checkOperationEquivalence(linalgOp, seedContractionOp.value());
  }
  return true;
}

/// Check that a given operation is "horizontal" to the group. The operation
/// is horizontal if the `slice` of the operation does not contain any op
/// from the group.
static bool isHorizontalToGroup(Operation *op,
                                const llvm::SetVector<Operation *> &currGroup,
                                const DominanceInfo &dominanceInfo,
                                Operation *seedOp) {
  BackwardSliceOptions options;
  options.inclusive = true;
  // Limit the slice to the seed to make sure the slice is small.
  options.filter = [&](Operation *op) {
    return !dominanceInfo.properlyDominates(op, seedOp);
  };
  llvm::SetVector<Operation *> slice;
  getBackwardSlice(op, &slice, options);
  return !llvm::any_of(currGroup, [&](Operation *groupedOp) {
    return slice.contains(groupedOp);
  });
}

/// Find all candidates that can be used for horizontal fusion. For example
/// ```
/// %0 = linalg.matmul ins(%arg0, %arg1)
/// %1 = linalg.matmul ins(%arg0, %arg2)
/// %2 = linalg.matmul ins(%arg0, %arg3)
/// ```
///
/// where all matmul share an operand can be combined into
///
/// ```
/// %4 = linalg.matmul ins(%arg0, concat(%arg1, %arg2, %arg3))
/// ```
///
/// This method recognizes such patterns. It also accounts for the quantized
/// case where individual operations might be have lower-precision operands and
/// accumulate in higher precision, followed by a `linalg.generic` that performs
/// the `truncf` on the result.
static std::optional<SmallVector<Operation *>> getHorizontalFusionGroupMembers(
    linalg::LinalgOp seedOp,
    const llvm::SmallDenseSet<Operation *> &groupedOperations,
    const DominanceInfo &dominanceInfo, int fusionLimit) {

  Value lhs = seedOp->getOperand(0);
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  Value rhs = seedOp->getOperand(1);
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  Value out = seedOp->getOperand(2);
  auto outType = cast<RankedTensorType>(out.getType());

  SetVector<Operation *> allOps;
  SmallVector<Operation *> contractionOps = {seedOp};
  allOps.insert(seedOp);

  auto canBeGrouped = [&](linalg::LinalgOp linalgOp) -> bool {
    if (linalgOp->getParentOp() != seedOp->getParentOp()) {
      return false;
    }

    // Constraints of the operation itself.
    if (!isEquivalentContractionOp(linalgOp, seedOp)) {
      return false;
    }
    if (linalgOp->getOperand(0).getType() != lhsType ||
        linalgOp->getOperand(1).getType() != rhsType ||
        linalgOp->getOperand(2).getType() != outType) {
      return false;
    }
    if (groupedOperations.contains(linalgOp)) {
      return false;
    }

    // Structural constraints related to being able to fuse the operations.
    if (!dominanceInfo.properlyDominates(seedOp, linalgOp)) {
      return false;
    }
    if (!isHorizontalToGroup(linalgOp, allOps, dominanceInfo, seedOp)) {
      return false;
    }
    return true;
  };

  // Iterate over users of LHS to find ops that can be grouped with the seed.
  SmallVector<Operation *> lhsUsers;
  for (Operation *lhsUser : lhs.getUsers()) {
    if (lhsUser->getBlock() != seedOp->getBlock() || lhsUser == seedOp) {
      continue;
    }

    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser || !canBeGrouped(linalgUser)) {
      continue;
    }
    lhsUsers.push_back(lhsUser);
  }

  // Sort the users so that the order is deterministic
  llvm::sort(lhsUsers, [&](Operation *lhs, Operation *rhs) {
    return dominanceInfo.properlyDominates(lhs, rhs);
  });

  // Collect all contraction op users of lhs.
  for (Operation *lhsUser : lhsUsers) {
    auto linalgUser = dyn_cast<linalg::LinalgOp>(lhsUser);
    if (!linalgUser) {
      continue;
    }

    contractionOps.push_back(linalgUser);
    allOps.insert(linalgUser);
    if (contractionOps.size() >= fusionLimit) {
      break;
    }
  }

  if (contractionOps.size() == 1) {
    return std::nullopt;
  }

  return contractionOps;
}

template <typename V, typename R>
static void appendRange(V &vector, R &&range) {
  vector.append(range.begin(), range.end());
}

static FailureOr<linalg::GenericOp>
fuseHorizontally(RewriterBase &rewriter, Location loc,
                 MutableArrayRef<Operation *> linalgOps) {
  if (!llvm::all_of(linalgOps, [](Operation *op) {
        return isa_and_nonnull<linalg::LinalgOp>(op);
      })) {
    return failure();
  }

  SmallVector<Value> fusedIns;
  SmallVector<Value> fusedOuts;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedInsIndexingMaps;
  SmallVector<AffineMap> fusedOutsIndexingMaps;

  for (auto op : linalgOps) {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    fusedIns.append(linalgOp.getDpsInputs());
    appendRange(fusedOuts, llvm::map_range(linalgOp.getDpsInitsMutable(),
                                           [](OpOperand &operand) {
                                             return operand.get();
                                           }));
    fusedResultTypes.append(linalgOp->result_type_begin(),
                            linalgOp->result_type_end());
    appendRange(
        fusedInsIndexingMaps,
        llvm::map_range(linalgOp.getIndexingMaps().getValue().take_front(
                            linalgOp.getNumDpsInputs()),
                        [](Attribute attr) {
                          return cast<AffineMapAttr>(attr).getValue();
                        }));
    appendRange(
        fusedOutsIndexingMaps,
        llvm::map_range(linalgOp.getIndexingMaps().getValue().drop_front(
                            linalgOp.getNumDpsInputs()),
                        [](Attribute attr) {
                          return cast<AffineMapAttr>(attr).getValue();
                        }));
  }

  SmallVector<utils::IteratorType> fusedIteratorTypes =
      cast<linalg::LinalgOp>(linalgOps.front()).getIteratorTypesArray();
  SmallVector<AffineMap> fusedIndexingMaps = std::move(fusedInsIndexingMaps);
  fusedIndexingMaps.append(fusedOutsIndexingMaps);
  auto fusedOp = rewriter.create<linalg::GenericOp>(
      loc, fusedResultTypes, fusedIns, fusedOuts, fusedIndexingMaps,
      fusedIteratorTypes, [](OpBuilder &, Location, ValueRange) {});

  Block *fusedBody = fusedOp.getBlock();
  auto insIndex = 0;
  auto outsIndex = fusedOp.getNumDpsInputs();
  SmallVector<Value> yieldVals;
  for (auto op : linalgOps) {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    Block *body = linalgOp.getBlock();
    SmallVector<Value> replacements = llvm::map_to_vector(
        fusedBody->getArguments().slice(insIndex, linalgOp.getNumDpsInputs()),
        [](BlockArgument arg) -> Value { return arg; });
    appendRange(
        replacements,
        llvm::map_range(fusedBody->getArguments().slice(
                            outsIndex, linalgOp.getNumDpsInits()),
                        [](BlockArgument arg) -> Value { return arg; }));

    rewriter.mergeBlocks(body, fusedBody, replacements);
    insIndex += linalgOp.getNumDpsInputs();
    outsIndex += linalgOp.getNumDpsInits();

    auto yieldOp = cast<linalg::YieldOp>(fusedBody->getTerminator());
    yieldVals.append(yieldOp->operand_begin(), yieldOp->operand_end());
    rewriter.eraseOp(yieldOp);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(fusedBody);
  rewriter.create<linalg::YieldOp>(loc, yieldVals);

  auto resultsIndex = 0;
  for (auto linalgOp : linalgOps) {
    rewriter.replaceOp(linalgOp, fusedOp->getResults().slice(
                                     resultsIndex, linalgOp->getNumResults()));
    resultsIndex += linalgOp->getNumResults();
  }

  return fusedOp;
}

static FailureOr<linalg::GenericOp>
fuseGroup(RewriterBase &rewriter, MutableArrayRef<Operation *> fusionGroup,
          DominanceInfo &dominanceInfo) {
  if (!llvm::all_of(fusionGroup, [](Operation *op) {
        return isa_and_nonnull<linalg::LinalgOp>(op);
      })) {
    return failure();
  }
  linalg::LinalgOp baseContractOp = cast<linalg::LinalgOp>(fusionGroup.front());
  Location loc = baseContractOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(baseContractOp);

  if (failed(moveOperandDefs(rewriter, fusionGroup, baseContractOp,
                             dominanceInfo))) {
    return baseContractOp.emitOpError("failed to re-order operand definitions");
  }

  FailureOr<linalg::GenericOp> fusedContractionOp =
      fuseHorizontally(rewriter, loc, fusionGroup);
  if (failed(fusedContractionOp)) {
    return baseContractOp.emitOpError(
        "failed to fuse contraction ops horizontally");
  }
  return fusedContractionOp.value();
}

void FuseHorizontalContractionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  DominanceInfo dominanceInfo(getOperation());

  SmallVector<SmallVector<Operation *>> horizontalFusionGroups;
  llvm::SmallDenseSet<Operation *> groupedOperations;

  getOperation()->walk([&](linalg::LinalgOp linalgOp) {
    if (!isEquivalentContractionOp(linalgOp)) {
      return;
    }
    // Avoid already grouped operations;
    if (groupedOperations.contains(linalgOp)) {
      return;
    }

    std::optional<SmallVector<Operation *>> fusionGroup =
        getHorizontalFusionGroupMembers(linalgOp, groupedOperations,
                                        dominanceInfo, fusionLimit);

    if (!fusionGroup) {
      return;
    }

    // Update statistics.
    numFusionGroups++;
    switch (fusionGroup->size()) {
    case 2:
      numSize2FusionGroups++;
      break;
    case 3:
      numSize3FusionGroups++;
      break;
    default:
      break;
    }

    groupedOperations.insert(fusionGroup->begin(), fusionGroup->end());
    horizontalFusionGroups.emplace_back(std::move(fusionGroup.value()));
  });

  if (horizontalFusionGroups.empty()) {
    return;
  }

  IRRewriter rewriter(context);
  for (auto &fusionGroup : horizontalFusionGroups) {
    FailureOr<linalg::GenericOp> fusedOperation =
        fuseGroup(rewriter, fusionGroup, dominanceInfo);
    if (failed(fusedOperation)) {
      return signalPassFailure();
    }
    auto fusedOp = fusedOperation.value();
    rewriter.setInsertionPoint(fusedOp);
    if (failed(linalg::deduplicateOperandsAndRemoveDeadResults(
            rewriter, fusedOp, /*removeOutputs=*/false))) {
      fusedOp->emitOpError("failed to remove duplicate operands");
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
