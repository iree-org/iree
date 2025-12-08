// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMDISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

/// Returns a bit vector of size number of loops of the `interfaceOp` with
/// the bits corresponding to outer parallel loops set to `true`.
static llvm::SmallBitVector getOuterParallelLoops(Operation *op) {
  if (auto setEncodingOp = dyn_cast<IREE::Encoding::SetEncodingOp>(op)) {
    return llvm::SmallBitVector(setEncodingOp.getResultType().getRank(), true);
  }
  if (auto unsetEncodingOp = dyn_cast<IREE::Encoding::UnsetEncodingOp>(op)) {
    return llvm::SmallBitVector(unsetEncodingOp.getResultType().getRank(),
                                true);
  }

  auto interfaceOp = dyn_cast<TilingInterface>(op);
  if (!interfaceOp) {
    // For ops that dont implement the `TilingInterface` just return empty.
    return llvm::SmallBitVector{};
  }
  SmallVector<utils::IteratorType> loopIteratorTypes =
      interfaceOp.getLoopIteratorTypes();
  llvm::SmallBitVector parallelLoops(loopIteratorTypes.size());
  for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType.value() != utils::IteratorType::parallel)
      break;
    parallelLoops.set(iteratorType.index());
  }
  return parallelLoops;
}

//===----------------------------------------------------------------------===//
// Root and fusion group handling
//===----------------------------------------------------------------------===//

namespace {
// `FusionGroup` is used to track operations that are to be fused with a given
// `rootOp`.
//
// This class contains an AffineMap for each operation to be fused. This map
// represents a mapping from the root op's outer parallel dims to this op's
// iteration space. `0` is used to represent when the iteration dimension has no
// mapping to the root op's outer parallel dimensions.
//
// For example:
//   affine_map<(d0, d1) -> (d0, 0, d1)>
//
// The root op has 2 outer parallel loops (`d0` and `d1`) and the example op
// has 3 dimensions where the first and last map `d0` and `d1` and the middle
// has no mapping to the root's outer parallel dimensions.
class FusionGroup {
public:
  FusionGroup(Operation *op) : rootOp(op) {
    llvm::SmallBitVector loops = getOuterParallelLoops(op);
    auto map = AffineMap::getFilteredIdentityMap(
        op->getContext(), loops.size(), [&](AffineDimExpr dimExpr) {
          return loops.test(dimExpr.getPosition());
        });
    map = inverseAndBroadcastProjectedPermutation(map);
    loopMaps.insert({op, map});
  };

  SmallVector<Operation *> getFusedOperations() const {
    return llvm::map_to_vector(
        loopMaps.getArrayRef(),
        [](std::pair<Operation *, AffineMap> pair) { return pair.first; });
  }

  Operation *getRoot() const { return rootOp; }

  // Get the mapping from `rootOp`'s outer parallel loops to `op`. This assumes
  // that the dependency chain from `rootOp` to `op` has already been inserted
  // into the group.
  //
  // Returns failure when there is no mapping or more than one mapping exists.
  FailureOr<AffineMap> getRootParallelLoopToOpMap(Operation *op) const;

  bool isFusable(Operation *op) const {
    // We only handle fusion across operation's operands. Don't fuse if the
    // operation is using values in the fusion group in it's body.
    bool hasUseFromAbove = false;
    mlir::visitUsedValuesDefinedAbove(
        op->getRegions(), [&](OpOperand *operand) {
          if (loopMaps.contains(operand->get().getDefiningOp())) {
            hasUseFromAbove = true;
          }
        });
    if (hasUseFromAbove) {
      return false;
    }

    FailureOr<AffineMap> maybeMap = getRootParallelLoopToOpMap(op);
    if (failed(maybeMap)) {
      return false;
    }

    // If the candidate is not all parallel, then its loop configuration should
    // be the same as the root.
    auto candidateOuterParallelLoop = getOuterParallelLoops(op);
    if (candidateOuterParallelLoop.size() !=
        candidateOuterParallelLoop.count()) {
      return loopMaps.lookup(rootOp) == maybeMap.value();
    }
    return true;
  }

  bool contains(Operation *op) const { return loopMaps.contains(op); }

  // Insert `op` into the fusion group.
  void insert(Operation *op);

  /// Returns true if `consumerOp` has a transitive dependency on the fusion
  /// group. This means that some transitive dependency of `consumerOp` (not in
  /// the fusion group) itself uses an operation in the fusion group. This is
  /// required for fusion because it must be legal to take a program slice that
  /// contains only the ops in the fusion group.
  bool
  hasTransitiveDependencyOnFusionGroup(Operation *consumerOp,
                                       DominanceInfo const &dominance) const {
    BackwardSliceOptions options;
    options.inclusive = true;
    options.omitUsesFromAbove = false;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *sliceBoundaryOp) {
      return !llvm::all_of(
          loopMaps.getArrayRef(), [&](std::pair<Operation *, AffineMap> pair) {
            return dominance.properlyDominates(sliceBoundaryOp, pair.first);
          });
    };

    llvm::SetVector<Operation *> slice;
    auto populateSlice = [&](OpOperand *operand) {
      // It's okay if the consumer directly uses an operation in the fusion
      // group.
      if (loopMaps.contains(operand->get().getDefiningOp())) {
        return;
      }
      LogicalResult result = getBackwardSlice(operand->get(), &slice, options);
      assert(result.succeeded() && "expected a backward slice");
      (void)result;
    };

    // Search all of the operands op `consumerOp` as well as all the values used
    // in its regions.
    mlir::visitUsedValuesDefinedAbove(consumerOp->getRegions(), populateSlice);
    for (OpOperand &operand : consumerOp->getOpOperands()) {
      populateSlice(&operand);
    }

    return llvm::any_of(loopMaps.getArrayRef(),
                        [&](std::pair<Operation *, AffineMap> pair) {
                          return slice.contains(pair.first);
                        });
  }

  // Check if adding `op` would exceed the operand limit.
  bool wouldExceedOperandLimit(Operation *op) const;

private:
  Operation *rootOp;
  // All operations to be fused with the root op. This does not include
  // `rootOp`.
  llvm::MapVector<Operation *, AffineMap> loopMaps;
};
} // namespace

void FusionGroup::insert(Operation *op) {
  assert(!contains(op) && "op already fused");
  FailureOr<AffineMap> map = getRootParallelLoopToOpMap(op);
  if (succeeded(map)) {
    loopMaps.insert({op, map.value()});
  } else {
    // TODO(IanWood1): some ops can be fused but don't implement
    // `LinalgFusionOpInterface` e.g. `tensor.insert_slice` or `linalg.unpack`.
    // `getRootParallelLoopToOpMap` fails when `op` is trying to fuse with one
    // of these ops. So, give `op` a root map.
    llvm::SmallBitVector loops = getOuterParallelLoops(op);
    auto map = AffineMap::getFilteredIdentityMap(
        op->getContext(), loops.size(), [&](AffineDimExpr dimExpr) {
          return loops.test(dimExpr.getPosition());
        });
    map = inverseAndBroadcastProjectedPermutation(map);
    loopMaps.insert({op, map});
  }
}

bool FusionGroup::wouldExceedOperandLimit(Operation *newOp) const {
  llvm::SmallSetVector<Operation *, kIreeMaxOperandCount> dispatchOperands;
  int64_t numResults = 0;

  auto visitOp = [&](Operation *op) {
    auto visitOperand = [&](OpOperand *operand) {
      if (!isa<RankedTensorType>(operand->get().getType())) {
        return;
      }
      Operation *definingOp = operand->get().getDefiningOp();
      if (llvm::isa_and_nonnull<linalg::FillOp, tensor::EmptyOp>(definingOp)) {
        return;
      }
      if (definingOp && definingOp != newOp && !loopMaps.contains(definingOp)) {
        dispatchOperands.insert(definingOp);
      }
    };
    visitUsedValuesDefinedAbove(op->getRegions(), visitOperand);
    llvm::for_each(llvm::make_pointer_range(op->getOpOperands()), visitOperand);

    for (OpResult result : op->getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return user != newOp && !loopMaps.contains(user);
          })) {
        ++numResults;
      }
    }
  };

  visitOp(newOp);
  for (auto [op, map] : this->loopMaps) {
    visitOp(op);
  }
  return (dispatchOperands.size() + numResults) > kIreeMaxOperandCount;
}

FailureOr<AffineMap>
FusionGroup::getRootParallelLoopToOpMap(Operation *op) const {
  assert(!contains(op) && "op cannot already be in group");
  auto fusionOp = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(op);
  if (!fusionOp) {
    return failure();
  }

  bool isConsumer = llvm::any_of(op->getOperands(), [this](Value v) {
    return contains(v.getDefiningOp());
  });
  assert(isConsumer !=
             llvm::any_of(op->getUsers(),
                          [this](Operation *op) { return contains(op); }) &&
         "op must be not be a producer and consumer");

  /// Computes the mapping from the root ops outer parallel loops to `op`'s
  /// iteration space via a direct producer/consumer of `op` that is already in
  /// the fusion group.
  auto getMapFromOpInFusionGroup =
      [&](AffineMap otherToOperand, AffineMap thisToOperand,
          AffineMap otherMap) -> FailureOr<AffineMap> {
    if (!otherToOperand || !thisToOperand ||
        !otherToOperand.isProjectedPermutation() ||
        !thisToOperand.isProjectedPermutation()) {
      return failure();
    }

    // `thisToOperand` is a mapping from the iteration space of `op` to the
    // operand's data space.
    // `inverseMap` is the  mapping from the operand data space to `op`'s
    // iteration space.
    AffineMap inverseMap =
        inverseAndBroadcastProjectedPermutation(thisToOperand);

    // `otherToOperand` maps "other's" (an op in the fusion group) iteration
    // space to the same operand's data space. Composing the two yields a
    // mapping from other's iteration space to `op`'s iteration space.
    AffineMap composedMap = inverseMap.compose(otherToOperand);

    // `otherMap` is other's mapping from the root's outer parallel loops to
    // other's iteration space. `composedMap.compose(otherMap)` computes the
    // mapping from the root's outer parallel loops to `op`'s iteration space.
    return composedMap.compose(otherMap);
  };

  AffineMap newMap;
  if (isConsumer) {
    for (OpOperand &operand : op->getOpOperands()) {
      Operation *definingOp = operand.get().getDefiningOp();
      if (!contains(definingOp)) {
        continue;
      }
      auto fusionProducer =
          operand.get()
              .getDefiningOp<IREE::LinalgExt::LinalgFusionOpInterface>();
      if (!fusionProducer) {
        return failure();
      }
      auto it = loopMaps.find(fusionProducer);
      assert(it != loopMaps.end());

      AffineMap producerResultMap = fusionProducer.getIndexingMapMatchingResult(
          cast<OpResult>(operand.get()));
      AffineMap consumerOperandMap = fusionOp.getMatchingIndexingMap(&operand);
      FailureOr<AffineMap> composedMap = getMapFromOpInFusionGroup(
          producerResultMap, consumerOperandMap, it->second);
      // Mapping must be the same for all operands.
      if (failed(composedMap) || (newMap && composedMap != newMap)) {
        return failure();
      }
      if (composedMap.value().getNumResults() ==
          composedMap.value().getNumOfZeroResults()) {
        return failure();
      }
      newMap = composedMap.value();
    }
  } else {
    for (OpOperand &operand : op->getUses()) {
      if (!contains(operand.getOwner())) {
        continue;
      }
      auto fusionConsumer = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(
          operand.getOwner());
      if (!fusionConsumer) {
        return failure();
      }
      auto it = loopMaps.find(operand.getOwner());
      assert(it != loopMaps.end());

      AffineMap consumerOperandMap =
          fusionConsumer.getMatchingIndexingMap(&operand);
      AffineMap producerResultMap =
          fusionOp.getIndexingMapMatchingResult(cast<OpResult>(operand.get()));
      FailureOr<AffineMap> composedMap = getMapFromOpInFusionGroup(
          consumerOperandMap, producerResultMap, it->second);
      // Mapping must be the same for all operands.
      if (failed(composedMap) || (newMap && composedMap != newMap)) {
        return failure();
      }
      newMap = composedMap.value();

      // Producers cannot be more parallel than consumers.
      if (compressUnusedDims(newMap).getNumDims() != it->second.getNumDims()) {
        return failure();
      }
    }
  }

  // Fail if there is no mapping or if there are no parallel loops in common.
  if (!newMap) {
    return failure();
  }
  return newMap;
}

namespace {

/// Tracks all the FusionGroups for the program.
class FusionTracker {
public:
  /// Create a new fusion group with `op` as the root.
  FusionGroup &createFusionGroup(MLIRContext *ctx, Operation *op) {
    fusionGroups.push_back(std::make_unique<FusionGroup>(op));
    opToGroup[op] = fusionGroups.back().get();
    return *fusionGroups.back();
  }

  // Get the fusion group that contains `op`.
  const FusionGroup &getFusionGroup(Operation *op) const {
    return *opToGroup.at(op);
  }

  // Get the fusion group that contains `op`.
  FusionGroup &getFusionGroup(Operation *op) { return *opToGroup.at(op); }

  const SmallVector<std::unique_ptr<FusionGroup>> &getFusionGroups() const {
    return fusionGroups;
  }

  void appendToFusionGroup(Operation *op, FusionGroup &fusionGroup) {
    assert(!isFusedOp(op) && "op already in a group");
    fusionGroup.insert(op);
    opToGroup[op] = &fusionGroup;
  }

  // Returns if `op` has been added to a FusionGroup in the tracker.
  bool isFusedOp(Operation *op) const { return opToGroup.contains(op); }

  // Returns if `op` is the root of a FusionGroup.
  bool isRootOp(Operation *op) const {
    return isFusedOp(op) && op == getFusionGroup(op).getRoot();
  }

private:
  SmallVector<std::unique_ptr<FusionGroup>> fusionGroups;
  DenseMap<Operation *, FusionGroup *> opToGroup;
};
} // namespace

//===----------------------------------------------------------------------===//
// Op property charecterizations
//===----------------------------------------------------------------------===//

/// Returns true if the reduced dimensions in the linalgOp of the unpack result
/// are not unpacked by the producer linalg::UnPackOp. This means the reduced
/// dimensions of the unpack result are not part of the inner_dims_pos.
static bool hasNoPackedReductionDimensions(linalg::LinalgOp linalgOp,
                                           Operation *producer) {
  auto unpack = dyn_cast<linalg::UnPackOp>(producer);
  if (!unpack) {
    return false;
  }
  AffineMap map;
  for (auto &use : producer->getResult(0).getUses()) {
    if (use.getOwner() == linalgOp) {
      map = linalgOp.getMatchingIndexingMap(&use);
      break;
    }
  }
  if (!map) {
    return false;
  }
  auto iterators = linalgOp.getIteratorTypesArray();
  auto reduction = utils::IteratorType::reduction;
  for (auto expr : llvm::enumerate(map.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr.value());
    if (!dim) {
      return false;
    }
    unsigned pos = dim.getPosition();
    if (iterators[pos] == reduction &&
        llvm::any_of(unpack.getInnerDimsPos(),
                     [expr](int64_t idp) { return expr.index() == idp; })) {
      return false;
    }
  }
  return true;
}

/// Returns true if the linalgOp is fusable with an unpack producer
static bool hasFusableUnpackProducer(linalg::LinalgOp linalgOp) {
  return llvm::any_of(linalgOp->getOperands(), [&](Value operand) {
    auto producer = operand.getDefiningOp<linalg::UnPackOp>();
    return producer && hasNoPackedReductionDimensions(linalgOp, producer);
  });
}

/// Operations that are treated as root operations for dispatch region
/// formation.
static bool isRootLikeOp(Operation *op) {
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
    return false;
  }
  // Dequantization-like ops get cloned into dispatches later.
  if (IREE::LinalgExt::isBitExtendOp(op)) {
    return false;
  }
  // Any Linalg named op or generic op with reduction iterator types is a root
  // op.
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    if (isa<linalg::GenericOp>(op)) {
      return linalgOp.getNumReductionLoops() != 0 &&
             !hasFusableUnpackProducer(linalgOp);
    }
    return !isa<linalg::FillOp>(op);
  }
  if (isa<TilingInterface>(op)) {
    return !isa<IREE::LinalgExt::GatherOp, tensor::PadOp, tensor::ConcatOp,
                linalg::PackOp>(op);
  }
  return isa<linalg::UnPackOp>(op);
}

/// Returns true if the operation is a `pack` op or a `set_encoding` op that
/// has pack semantics.
// TODO(ravishankarm): This seems like a use case for an interface.
static bool isPackLikeOp(Operation *op) {
  return isa<IREE::Encoding::SetEncodingOp, linalg::PackOp>(op);
}

/// Returns true if the operation is an `unpack` op or an `unset_encoding` op.
static bool isUnpackLikeOp(Operation *op) {
  return isa<IREE::Encoding::UnsetEncodingOp, linalg::UnPackOp>(op);
}

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

/// For all uses of an operation, return the uses that could be fused.
/// The returned vector contains the uses in dominance order.
static SmallVector<OpOperand *>
getFusableUses(MLIRContext *context, Operation *op,
               DominanceInfo const &dominanceInfo, bool aggressiveFusion) {
  if (!aggressiveFusion && llvm::count_if(op->getUses(), [](OpOperand &use) {
                             return !isa<tensor::DimOp>(use.getOwner());
                           }) != 1) {
    return {};
  }

  // Collect all fusable user candidates.
  SetVector<OpOperand *> fusableUses;
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (isa<tensor::DimOp>(user)) {
      continue;
    }
    if (op->getBlock() != user->getBlock()) {
      continue;
    }
    fusableUses.insert(&use);
  }

  SmallVector<OpOperand *> usesVec = fusableUses.takeVector();
  llvm::sort(usesVec, [&](OpOperand *lhsUse, OpOperand *rhsUse) {
    return dominanceInfo.properlyDominates(lhsUse->getOwner(),
                                           rhsUse->getOwner());
  });

  return usesVec;
}

/// For the fusion of root op -> elementwise operation to be bufferized
/// in-place without use of extra memory, the result of the root operation
/// must be able to reuse the buffer for the result of the elementwise
/// operation. Check if that is possible for the input/init operand pair.
static bool canUseInOperandAsInitOperand(OpOperand *inOperand,
                                         OpOperand *initOperand) {
  assert(inOperand->getOwner() == initOperand->getOwner() &&
         "expected in-operand and init-operand to be owned by same operation");

  // Check that the owner is a `generic` op.
  auto genericOp = dyn_cast<linalg::GenericOp>(inOperand->getOwner());
  if (!genericOp)
    return false;

  // All loops to be parallel.
  if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }

  /// The input operand cannot be an init operand already.
  if (genericOp.isDpsInit(inOperand))
    return false;

  // If the init operand value is used it cannot be reused for the input
  // operand.
  if (genericOp.payloadUsesValueFromOperand(initOperand))
    return false;

  // Indexing map used to access the input and init have to match.
  if (genericOp.getMatchingIndexingMap(inOperand) !=
      genericOp.getMatchingIndexingMap(initOperand)) {
    return false;
  }

  // Types have to match for the input operand to reuse the buffer from the init
  // operand
  if (inOperand->get().getType() != initOperand->get().getType())
    return false;

  return true;
}

/// Returns true if this is a fusable use, while fusing a root with its
/// consumer.
static bool
isFusableWithConsumer(OpOperand &fusedOperand, const FusionTracker &tracker,
                      FormDispatchRegionsPassOptions const &options) {
  Operation *producer = fusedOperand.get().getDefiningOp();
  Operation *consumer = fusedOperand.getOwner();

  // If consumer is a dequant operation, dont fuse it. These get cloned
  // into their consumers.
  IREE::Flow::ClonableIntoDispatchOptions clonableOptions;
  clonableOptions.aggressive = options.aggressiveFusion;
  if (IREE::Flow::isClonableIntoDispatchOp(consumer, clonableOptions)) {
    return false;
  }

  // Fuse unset_encoding operations with `tensor.extract_slice` and elementwise
  // generic ops.
  if (isUnpackLikeOp(producer)) {
    // Fuse `unset_encoding/unpack` -> elementwise operations. Fuse unpack with
    // non-overlapping reductions (i.e., the reduction dimension is not packed).
    if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
      if (hasNoPackedReductionDimensions(consumerLinalgOp, producer)) {
        return true;
      }
      return linalg::isElementwise(consumerLinalgOp) &&
             consumerLinalgOp.getNumLoops() ==
                 cast<RankedTensorType>(producer->getResult(0).getType())
                     .getRank();
    }
    return false;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case<tensor::PadOp>([&](auto padOp) { return true; })
        .Case<linalg::LinalgOp>([&](auto linalgOp) {
          AffineMap producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              cast<OpResult>(fusedOperand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return producerIndexingMap.isIdentity();
        })
        .Default([](Operation *) { return false; });
  }

  // By default, padding should be fused with producers. It is hard to square
  // this with fusion of pad with consumer. So for now split the difference.
  // Either fuse pad with producer or with consumer.
  if (auto padOp = dyn_cast<tensor::PadOp>(consumer)) {
    if (options.fusePadWithProducers) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  // Insert slice ops should always be fused with their producers.
  if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(consumer)) {
    // TODO: Enable multi-use slice source fusion.
    Value source = insertSliceOp.getSource();
    if (!source.hasOneUse() || source.getDefiningOp() != producer) {
      return false;
    }
    // Fuse in `insert_slice` consumer operations if destination is a fill.
    // TODO: This can be generalized, but destination cannot be a
    // `arith.constant` or other constant-like objects. `linalg.fill` captures a
    // common case of pad generalization.
    return insertSliceOp.getDest().getDefiningOp<linalg::FillOp>();
  }

  // TODO(#16025): Enable mmt4d fusion. It is disabled because the backends
  // can not set multi lowering_config properly. See the issue for more details.
  if (isa<linalg::Mmt4DOp, linalg::BatchMmt4DOp>(producer)) {
    return false;
  }

  auto producerFusionOp =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(producer);
  auto consumerFusionOp =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(consumer);
  if (!producerFusionOp || !consumerFusionOp)
    return false;

  // Check that the consumer is all parallel.
  if (consumerFusionOp.getNumLoops() !=
      consumerFusionOp.getNumParallelLoops()) {
    return false;
  }

  if (!tracker.getFusionGroup(producer).isFusable(consumer)) {
    return false;
  }

  // Check operand limit before allowing fusion
  if (tracker.getFusionGroup(producer).wouldExceedOperandLimit(consumer)) {
    return false;
  }

  // Check if the iteration spaces of the producer and consumer are same.
  // TODO(#12664): This is unnecessary requirement, but we need a better config
  // to tile the consumer with a larger iteration space.
  if (!options.aggressiveFusion) {
    FailureOr<SmallVector<int64_t>> producerIterationSpace =
        producerFusionOp.getStaticLoopRanges();
    FailureOr<SmallVector<int64_t>> consumerIterationSpace =
        consumerFusionOp.getStaticLoopRanges();
    if (failed(producerIterationSpace) || failed(consumerIterationSpace)) {
      return false;
    }
    if (producerIterationSpace.value().size() <
        consumerIterationSpace.value().size()) {
      return false;
    }
  }

  // Under aggressive fusion assume that the dispatches are vectorized. In which
  // case we dont need to account for the subsequent stack allocation condition.
  if (options.aggressiveFusion) {
    return true;
  }

  // While fusing with consumer, the result of the root might not be the final
  // result of the dispatch. To avoid a stack allocation we have to ensure that
  // all operations can bufferize without needing additional memory.
  auto consumerDstOp =
      dyn_cast<DestinationStyleOpInterface>(consumerFusionOp.getOperation());
  if (!consumerDstOp) {
    return true;
  }

  for (OpOperand *inputOperand : consumerDstOp.getDpsInputOperands()) {
    if (inputOperand->get().getDefiningOp() != producer)
      continue;
    if (isa<linalg::ConvolutionOpInterface>(producer) &&
        !llvm::any_of(
            consumerDstOp.getDpsInitsMutable(), [&](OpOperand &initOperand) {
              return canUseInOperandAsInitOperand(inputOperand, &initOperand);
            })) {
      return false;
    }
  }

  return true;
}

/// Fuses roots with its consumers. If a root is fused with its consumer, it is
/// no more tagged as a root to aid with the dispatch region formation.
static void
fuseRootsWithConsumers(MLIRContext *context, ArrayRef<Operation *> roots,
                       DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker) {
  // Fuse with consumers where possible.
  for (Operation *root : roots) {
    SmallVector<Operation *> workList;
    FusionGroup &fusionGroup = tracker.getFusionGroup(root);
    workList.push_back(root);
    while (!workList.empty()) {
      Operation *currRoot = workList.pop_back_val();

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, currRoot, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty()) {
        continue;
      }

      // Analyse the use to see if it is fusable.
      for (OpOperand *fusableUse : fusableUses) {
        Operation *consumerOp = fusableUse->getOwner();
        if (tracker.isRootOp(consumerOp) || tracker.isFusedOp(consumerOp)) {
          continue;
        }

        // Ensure that fusing the consumer would not cause use-def violations.
        if (tracker.getFusionGroup(currRoot)
                .hasTransitiveDependencyOnFusionGroup(fusableUse->getOwner(),
                                                      dominanceInfo)) {
          continue;
        }

        if (isFusableWithConsumer(*fusableUse, tracker, options)) {
          tracker.appendToFusionGroup(consumerOp, fusionGroup);
          workList.push_back(consumerOp);
        } else {
          break;
        }
      }
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool isFusableWithProducer(OpOperand &operand,
                                  const FusionTracker &tracker,
                                  FormDispatchRegionsPassOptions const &options,
                                  bool fuseWithTruncate) {
  Operation *producer = operand.get().getDefiningOp();
  Operation *consumer = operand.getOwner();

  if (!fuseWithTruncate && IREE::LinalgExt::isBitTruncateOp(producer)) {
    return false;
  }

  if (auto padOp = dyn_cast<tensor::PadOp>(consumer)) {
    if (options.fusePadWithProducers) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  auto linalgConsumer = dyn_cast<linalg::LinalgOp>(consumer);
  if (options.fusePadWithConsumers && isa<tensor::PadOp>(producer) &&
      linalgConsumer && linalg::isaConvolutionOpInterface(linalgConsumer)) {
    return true;
  }

  if (auto attentionOp = dyn_cast<IREE::LinalgExt::AttentionOp>(consumer)) {
    // Disable all other producer fusion. TODO: Enable some producer fusions.
    return false;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case<tensor::PadOp>([&](auto padOp) { return true; })
        .Case<linalg::LinalgOp>([&](auto linalgOp) {
          if (auto packOp = dyn_cast<linalg::PackOp>(consumer)) {
            // TODO(#12746): fusion of pack with dynamic inner tile size
            // causes an error in backend. Disable for now.
            if (!packOp.getInnerTiles().empty()) {
              return false;
            }
          }
          AffineMap producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              cast<OpResult>(operand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return producerIndexingMap.isIdentity();
        })
        .Default([](Operation *) { return false; });
  }

  if (!isa<IREE::LinalgExt::LinalgFusionOpInterface>(consumer) ||
      !isa<IREE::LinalgExt::LinalgFusionOpInterface>(producer)) {
    return false;
  }

  if (!options.aggressiveFusion) {
    auto consumerFusionOp = dyn_cast<DestinationStyleOpInterface>(consumer);
    if (consumerFusionOp && !consumerFusionOp.isDpsInit(&operand)) {
      return false;
    }
  }

  if (!tracker.getFusionGroup(consumer).isFusable(producer)) {
    return false;
  }

  // Check operand limit before allowing fusion
  if (tracker.getFusionGroup(consumer).wouldExceedOperandLimit(producer)) {
    return false;
  }

  return true;
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void
fuseRootsWithProducers(MLIRContext *context, Operation *root,
                       FusionGroup &fusionGroup,
                       DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker, bool fuseWithTruncate) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);
  IREE::Flow::ClonableIntoDispatchOptions clonableOptions;
  clonableOptions.aggressive = options.aggressiveFusion;
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer)
        continue;
      if (IREE::Flow::isClonableIntoDispatchOp(producer, clonableOptions) ||
          tracker.isFusedOp(producer) || tracker.isRootOp(producer)) {
        continue;
      }

      if (!isFusableWithProducer(operand, tracker, options, fuseWithTruncate)) {
        continue;
      }

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, producer, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty() || fusableUses.front()->getOwner() != candidate)
        continue;

      tracker.appendToFusionGroup(producer, fusionGroup);
      worklist.push_back(producer);
    }
  }
}

/// Some heuristic is needed to fuse a dispatchable op with root operations
/// using tile + fuse.
static void
decideFusableLinalgOps(Region &region, DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       FusionTracker &tracker, unsigned numRootOps = 0) {
  MLIRContext *context = region.getContext();
  OpBuilder builder(context);
  IREE::Flow::ClonableIntoDispatchOptions clonableOptions;
  clonableOptions.aggressive = options.aggressiveFusion;
  for (Block &block : region) {
    // Dispatch region formation works by first cloning the root into
    // the dispatch region and then pulling operations in.
    // So procedure here is to
    // - First find the roots
    // - To fuse with consumers make the consumer the root.
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      if (isa<scf::SCFDialect>(op.getDialect())) {
        for (auto &region : op.getRegions()) {
          decideFusableLinalgOps(region, dominanceInfo, options, tracker,
                                 numRootOps);
        }
        continue;
      }

      // Start with a root operation and fuse its producers.
      if (tracker.isFusedOp(&op) || !isRootLikeOp(&op))
        continue;
      FusionGroup &newGroup = tracker.createFusionGroup(context, &op);
      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options, tracker);
    for (Operation *root : roots) {
      FusionGroup &fusionGroup = tracker.getFusionGroup(root);
      fuseRootsWithProducers(context, root, fusionGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/true);
    }
  }

  // Once all root linalg ops have been tagged, put all remaining generic ops
  // into their own dispatches.
  for (Block &block : region) {
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // If it is part of a fusion group or root op, ignore it.
      if (tracker.isFusedOp(&op) || tracker.isRootOp(&op))
        continue;
      // Only look for Linalg ops here. Avoid moving `linalg.fill` that aren't
      // fused with anything else into their own dispatches since it is better
      // to convert them to splats. Also avoid moving dequantization-like ops
      // into their own dispatch since it is better to clone these ops and avoid
      // materializing large tensors between dispatches.
      if (!isa<linalg::LinalgOp, tensor::PadOp, linalg::PackOp>(op) ||
          IREE::Flow::isClonableIntoDispatchOp(&op, clonableOptions)) {
        continue;
      }

      // For now check if this is a rope computation that is to be fused with
      // attention.
      // TODO: Ideally this is just regular gather fusion which will be covered
      // by the `isClonableIntoDispatchOp` call above, but for now this is done
      // as a point fix.
      if (IREE::LinalgExt::isGatherlikeOp(&op) &&
          llvm::all_of(op.getUsers(), [](Operation *op) {
            return isa<IREE::LinalgExt::AttentionOp>(op);
          })) {
        continue;
      }

      FusionGroup &newGroup = tracker.createFusionGroup(context, &op);
      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options, tracker);
    for (Operation *root : roots) {
      FusionGroup &fusionGroup = tracker.getFusionGroup(root);
      fuseRootsWithProducers(context, root, fusionGroup, dominanceInfo, options,
                             tracker,
                             /*fuseWithTruncate=*/true);
    }
  }
}

//===----------------------------------------------------------------------===//
// Dispatch region formation
//===----------------------------------------------------------------------===//

/// Create IREE::Flow::DispatchGroupsOps based on a fusion heuristic.
static LogicalResult
createFusionGroups(TensorDimTrackingRewriter &rewriter,
                   mlir::FunctionOpInterface funcOp,
                   DominanceInfo &dominanceInfo,
                   FormDispatchRegionsPassOptions const &options) {
  // Step 1: Decide fusion groups (heuristic).
  FusionTracker tracker;
  decideFusableLinalgOps(funcOp.getFunctionBody(), dominanceInfo, options,
                         tracker);

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After deciding fusion groups ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Step 2. Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  for (const auto &fusionGroup : tracker.getFusionGroups()) {
    Operation *root = fusionGroup->getRoot();
    // Sort producers and consumers topologically. All fused ops must be in the
    // same block as the root.
    SmallVector<Operation *> currFusedOperations =
        fusionGroup->getFusedOperations();
    bool sortResult = mlir::computeTopologicalSorting(currFusedOperations);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    int rootPos = 0;
    for (auto [index, fusedOperation] : llvm::enumerate(currFusedOperations)) {
      if (fusedOperation == root) {
        rootPos = index;
        break;
      }
    }
    SmallVector<Operation *> producers, consumers;
    if (rootPos > 0) {
      producers = llvm::to_vector(
          ArrayRef<Operation *>(currFusedOperations).take_front(rootPos));
    }
    if (rootPos < currFusedOperations.size() - 1) {
      consumers = llvm::to_vector(
          ArrayRef<Operation *>(currFusedOperations).drop_front(rootPos + 1));
    }

    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
        return failure();
      }
    }

    // Create fusion group.
    IREE::Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = IREE::Flow::wrapOpInDispatchRegion(rewriter, root);
    if (failed(maybeRegionOp)) {
      return root->emitOpError("failed to move root into dispatch");
    }
    regionOp = *maybeRegionOp;

    // Move ops into the region.
    for (Operation *producer : llvm::reverse(producers)) {
      // Simplify tensor::DimOps.
      {
        SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
        if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
          return failure();
        }
      }

      auto newRegionOp =
          movePrecedingOpsIntoDispatchRegion(rewriter, producer, regionOp);
      if (failed(newRegionOp)) {
        producer->emitWarning("failed to move producer into region");
        continue;
      }
      regionOp = *newRegionOp;
    }

    for (Operation *consumer : consumers) {
      // Simplify tensor::DimOps.
      {
        SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
        if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
          return failure();
        }
      }

      auto newRegionOp = IREE::Flow::moveFollowingOpIntoDispatchRegion(
          rewriter, consumer, regionOp);
      if (failed(newRegionOp)) {
        consumer->emitWarning("failed to move consumer into region");
        continue;
      }
      regionOp = *newRegionOp;
    }
    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
        return failure();
      }
    }
    regionOps.push_back(regionOp);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After creating flow.dispatch.region ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  return success();
}

namespace {
/// Pass declaration.
struct FormDispatchRegionsPass final
    : public impl::FormDispatchRegionsPassBase<FormDispatchRegionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void FormDispatchRegionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  DominanceInfo &dominanceInfo = getAnalysis<DominanceInfo>();
  TensorDimTrackingRewriter rewriter(funcOp);
  FormDispatchRegionsPassOptions options{aggressiveFusion, fusePadWithConsumers,
                                         fusePadWithProducers};
  if (failed(createFusionGroups(rewriter, funcOp, dominanceInfo, options))) {
    funcOp->emitOpError("failed to create fusion groups");
    return signalPassFailure();
  }

  // Canonicalize all the dispatch regions to remove unused operands.
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(patterns, context);
  GreedyRewriteConfig config;
  config.setMaxIterations(GreedyRewriteConfig::kNoLimit).enableFolding(true);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
    funcOp.emitOpError("failed in cleanup patterns");
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation
