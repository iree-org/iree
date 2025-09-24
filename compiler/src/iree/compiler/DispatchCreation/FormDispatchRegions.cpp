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

#define DEBUG_TYPE "iree-dispatch-creation-form-dispatch-regions"

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMDISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

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
    fusionGroups = llvm::map_to_vector(fusionGroupsAttr, [](Attribute attr) {
      return llvm::cast<IntegerAttr>(attr).getInt();
    });
  }
  return fusionGroups;
}
/// Appends the given `op` to the `newGroups` fusion groups.
static void appendToFusionGroup(Operation *op, ArrayRef<int64_t> newGroups) {
  SmallVector<int64_t> fusionGroups = getFusionGroups(op);
  fusionGroups.append(newGroups.begin(), newGroups.end());
  op->setAttr(kFusionGroupsAttr, Builder(op).getI64ArrayAttr(fusionGroups));
}
/// Removes the fusion groups attribute.
static void removeFusionGroupsAttribute(Operation *op) {
  op->removeAttr(kFusionGroupsAttr);
}

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
static bool isRootOp(Operation *op) {
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

/// Returns true if `map` is an identity map with zeros, i.e. if you
/// drop the result exprs that are constant zeros, the `map` will become an
/// identity.
static bool isIdentityMapWithZeros(AffineMap map) {
  if (map.getNumSymbols() != 0)
    return false;
  if (map.isEmpty())
    return false;
  unsigned dimsSeen = 0;
  for (AffineExpr result : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(result)) {
      if (dimExpr.getPosition() != dimsSeen) {
        return false;
      }
      dimsSeen++;
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
      if (constExpr.getValue() != 0) {
        return false;
      }
    } else {
      return false;
    }
  }
  return dimsSeen == map.getNumDims();
}

static bool
matchIteratorTypes(const llvm::SmallBitVector &rootOuterParallelLoop,
                   const llvm::SmallBitVector &candidateOuterParallelLoop) {
  // If the candidate is not all parallel, then its loop configuration should be
  // the same as the root.
  if (candidateOuterParallelLoop.size() != candidateOuterParallelLoop.count()) {
    return rootOuterParallelLoop == candidateOuterParallelLoop;
  }

  // If the candidate is all parallel, then it should be at least as parallel as
  // the root.
  for (int pos : llvm::seq<int>(0, std::min(candidateOuterParallelLoop.size(),
                                            rootOuterParallelLoop.size()))) {
    // If we reach the end of the outer loops of the root, break out of the
    // loop.
    if (!rootOuterParallelLoop.test(pos))
      break;
    // If the root loop is parallel, the candidate loop should also be parallel.
    if (!candidateOuterParallelLoop.test(pos))
      return false;
  }
  return true;
}

// Method to check if the op with have compatible indexing map on outer-parallel
// loops. Currently it means the map needs to be identity on the those
// dimensions, ignoring its reduction dimensions.
static bool hasCompatibleOuterParallelLoops(
    TilingInterface tileOp, AffineMap indexingMap,
    const llvm::SmallBitVector &rootOuterParallelLoops) {
  if (!indexingMap.isProjectedPermutation()) {
    return false;
  }

  llvm::SmallBitVector parallelLoops = getOuterParallelLoops(tileOp);
  if (!matchIteratorTypes(rootOuterParallelLoops, parallelLoops)) {
    return false;
  }

  /// Project out the non-parallel dimensions.
  llvm::SmallBitVector projectedDims(rootOuterParallelLoops);
  projectedDims.flip();
  projectedDims.resize(tileOp.getLoopIteratorTypes().size(), true);
  auto projectedMap = getProjectedMap(indexingMap, projectedDims);
  return isIdentityMapWithZeros(projectedMap);
}

// Method to check if two `linalg.generic` op with producer-consumer
// relationship through `operand` have compatible outer-parallel loops.
static bool hasCompatibleOuterParallelLoops(
    OpOperand &operand, const llvm::SmallBitVector &rootOuterParallelLoops) {
  auto producer =
      operand.get().getDefiningOp<IREE::LinalgExt::LinalgFusionOpInterface>();
  auto consumer =
      dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(operand.getOwner());
  if (!producer || !consumer)
    return false;

  auto producerIndexingMap = producer.getIndexingMapMatchingResult(
      llvm::cast<OpResult>(operand.get()));
  auto consumerIndexingMap = consumer.getMatchingIndexingMap(&operand);

  if (!producerIndexingMap || !consumerIndexingMap) {
    return false;
  }

  return hasCompatibleOuterParallelLoops(
             cast<TilingInterface>(producer.getOperation()),
             producerIndexingMap, rootOuterParallelLoops) &&
         hasCompatibleOuterParallelLoops(
             cast<TilingInterface>(consumer.getOperation()),
             consumerIndexingMap, rootOuterParallelLoops);
}

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
    fusableUses.insert(&use);
  }

  SmallVector<OpOperand *> usesVec = fusableUses.takeVector();
  llvm::sort(usesVec, [&](OpOperand *lhsUse, OpOperand *rhsUse) {
    return dominanceInfo.properlyDominates(lhsUse->getOwner(),
                                           rhsUse->getOwner());
  });

  return usesVec;
}

/// Returns true if the operands are fusable.
static bool areOpsFusable(Operation *producer, Operation *consumer,
                          const llvm::SmallBitVector &rootOuterParallelLoops) {
  // Collect all the uses from producer to consumer.
  SmallVector<OpOperand *> allUses;
  for (OpOperand &producerUse : producer->getUses()) {
    if (producerUse.getOwner() != consumer)
      continue;
    allUses.push_back(&producerUse);
  }

  // Check that the consumer and producer have compatible outer parallel loops.
  if (!llvm::all_of(allUses, [&](OpOperand *operand) {
        return hasCompatibleOuterParallelLoops(*operand,
                                               rootOuterParallelLoops);
      })) {
    return false;
  }
  return true;
}

/// The logic to decide fusability (using the `hasCompatibleOuterParallelLoops`)
/// currently works when the indexing map corresponding to result of the
/// producer and indexing map corresponding to operand in the result are not
/// transposed with respect to each other. To find more fusion opportunities for
/// consumer elementwise operation, the indexing maps in the consumer can be
/// made to "align" with the indexing map of the producer to enhance fusion.
static bool makeConsumerFusableViaInterchange(
    OpOperand &fusableOperand,
    const llvm::SmallBitVector &rootOuterParallelLoops) {
  auto producer =
      fusableOperand.get()
          .getDefiningOp<IREE::LinalgExt::LinalgFusionOpInterface>();
  if (!producer) {
    return false;
  }

  auto consumer = dyn_cast<linalg::GenericOp>(fusableOperand.getOwner());
  if (!consumer) {
    return false;
  }

  if (!linalg::isElementwise(consumer) || consumer.getNumResults() != 1) {
    return false;
  }

  // If the indexing map in the consumer is already "compatible" with the
  // indexing map in the producer, do nothing.
  AffineMap producerIndexingMap = producer.getIndexingMapMatchingResult(
      cast<OpResult>(fusableOperand.get()));
  if (!producerIndexingMap) {
    return false;
  }
  producerIndexingMap = getProjectedMap(
      producerIndexingMap, getUnusedDimsBitVector(producerIndexingMap));
  AffineMap consumerIndexingMap =
      consumer.getMatchingIndexingMap(&fusableOperand);

  // Since the iteration space of the consumer is going to be permuted
  // to make it match with the indexing map in the producer, the interchange
  // requires the indexing map in the consumer to be a permutation.
  // If the producer indexing map and consumer indexing map are the same,
  // then the permutation of iteration space becomes a no-op, in which
  // case the permutation wasnt required for fusion. Return false here
  // to indicate that the permutation is not going to "enhance" the
  // fusion opportunities.
  if (!consumerIndexingMap.isPermutation() ||
      producerIndexingMap == consumerIndexingMap) {
    return false;
  }
  OpResult result = cast<OpResult>(consumer.getResult(0));
  if (!consumer.getIndexingMapMatchingResult(result).isPermutation()) {
    return false;
  }

  // For now this is restricting that all indexing maps corresponding to the
  // input are same as the indexing map of the fused operand, or are projected
  // permutations. This avoids ping-ponging between different iteration space
  // permutations without having any way to pick which is better.
  if (!llvm::all_of(
          consumer.getDpsInputOperands(), [&](OpOperand *inputOperand) {
            AffineMap map = consumer.getMatchingIndexingMap(inputOperand);
            return map == consumerIndexingMap ||
                   (map.isProjectedPermutation() && !map.isPermutation());
          })) {
    return false;
  }

  // Make the input map match the producer map by applying a permutation map
  // computed with consumerIndexingMap.compose(inv(producerIndexingMap))
  AffineMap invProducerIndexingMap = inversePermutation(producerIndexingMap);
  AffineMap permutationMap =
      consumerIndexingMap.compose(invProducerIndexingMap);
  auto perm = llvm::map_to_vector(permutationMap.getResults(),
                                  [](AffineExpr e) -> unsigned {
                                    return cast<AffineDimExpr>(e).getPosition();
                                  });
  IRRewriter rewriter(consumer->getContext());
  FailureOr<linalg::GenericOp> interchangedOp =
      linalg::interchangeGenericOp(rewriter, consumer, perm);
  (void)interchangedOp;
  assert(succeeded(interchangedOp) && "expected interchange to succeed");
  assert(interchangedOp.value() == consumer &&
         "expected interchange to happen in place");
  return true;
}

static bool makeProducerFusableViaInterchange(
    OpOperand &fusableOperand,
    const llvm::SmallBitVector &rootOuterParallelLoops) {
  auto producer = fusableOperand.get().getDefiningOp<linalg::GenericOp>();
  if (!producer) {
    return false;
  }

  auto consumer = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(
      fusableOperand.getOwner());
  if (!consumer) {
    return false;
  }

  if (!linalg::isElementwise(producer) || producer.getNumResults() != 1) {
    return false;
  }

  AffineMap producerIndexingMap = producer.getIndexingMapMatchingResult(
      cast<OpResult>(fusableOperand.get()));
  producerIndexingMap = getProjectedMap(
      producerIndexingMap, getUnusedDimsBitVector(producerIndexingMap));
  AffineMap consumerIndexingMap =
      consumer.getMatchingIndexingMap(&fusableOperand);
  if (!consumerIndexingMap || !consumerIndexingMap.isPermutation() ||
      producerIndexingMap == consumerIndexingMap) {
    return false;
  }

  // Make the input map match the consumer map by applying a permutation map
  AffineMap invProducerIndexingMap = inversePermutation(producerIndexingMap);
  AffineMap permutationMap =
      consumerIndexingMap.compose(invProducerIndexingMap);
  auto perm = llvm::map_to_vector(permutationMap.getResults(),
                                  [](AffineExpr e) -> unsigned {
                                    return cast<AffineDimExpr>(e).getPosition();
                                  });
  IRRewriter rewriter(consumer->getContext());
  FailureOr<linalg::GenericOp> interchangedOp =
      linalg::interchangeGenericOp(rewriter, producer, perm);
  (void)interchangedOp;
  assert(succeeded(interchangedOp) && "expected interchange to succeed");
  assert(interchangedOp.value() == producer &&
         "expected interchange to happen in place");
  return true;
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
isFusableWithConsumer(OpOperand &fusedOperand,
                      const llvm::SmallBitVector &rootOuterParallelLoops,
                      FormDispatchRegionsPassOptions const &options) {
  Operation *producer = fusedOperand.get().getDefiningOp();
  Operation *consumer = fusedOperand.getOwner();

  // If consumer is a dequant operation, dont fuse it. These get cloned
  // into their consumers.
  if (IREE::LinalgExt::isBitExtendOp(consumer)) {
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
                 llvm::cast<RankedTensorType>(producer->getResult(0).getType())
                     .getRank();
    }
    return false;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case<tensor::PadOp>([&](auto padOp) { return true; })
        .Case<linalg::LinalgOp>([&](auto linalgOp) {
          auto producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              llvm::cast<OpResult>(fusedOperand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return hasCompatibleOuterParallelLoops(
              cast<TilingInterface>(linalgOp.getOperation()),
              producerIndexingMap, rootOuterParallelLoops);
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

  if (!areOpsFusable(producer, consumer, rootOuterParallelLoops)) {
    // Check if interchange in the consumer makes it fusable.
    // Currently limit it to horizontally fused gemms.
    // TODO(#20019) to remove this restriction.
    if (!IREE::LinalgExt::isaHorizontallyFusedContraction(producer) ||
        !makeConsumerFusableViaInterchange(fusedOperand,
                                           rootOuterParallelLoops)) {
      return false;
    }
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
                       FormDispatchRegionsPassOptions const &options) {
  // Fuse with consumers where possible.
  for (Operation *root : roots) {
    SmallVector<Operation *> workList;
    llvm::SmallBitVector rootOuterParallelLoops = getOuterParallelLoops(root);
    int64_t rootNumber = getRootNumber(root);
    workList.push_back(root);
    while (!workList.empty()) {
      Operation *currRoot = workList.pop_back_val();

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, currRoot, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty()) {
        continue;
      }

      // For now prune the fusable uses due to codegen failures. Ideally we
      // should just be taking the whole set of fusable uses.
      if (IREE::LinalgExt::isBitTruncateOp(fusableUses.front()->getOwner())) {
        fusableUses =
            llvm::filter_to_vector(fusableUses, [](OpOperand *operand) {
              return IREE::LinalgExt::isBitTruncateOp(operand->getOwner());
            });
      } else {
        fusableUses.resize(1);
      }

      // Analyse the use to see if it is fusable.
      for (OpOperand *fusableUse : fusableUses) {
        Operation *consumerOp = fusableUse->getOwner();
        if (hasRootOpAttribute(consumerOp) ||
            hasFusionGroupsAttribute(consumerOp)) {
          continue;
        }

        if (isFusableWithConsumer(*fusableUse, rootOuterParallelLoops,
                                  options)) {
          appendToFusionGroup(consumerOp, rootNumber);
          workList.push_back(consumerOp);
        } else {
          break;
        }
      }
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool isFusableWithProducer(
    OpOperand &operand, const llvm::SmallBitVector &rootOuterParallelLoops,
    FormDispatchRegionsPassOptions const &options, bool fuseWithTruncate) {
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
          auto producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              llvm::cast<OpResult>(operand.get()));
          // Make sure the producer op has an identity result indexing map. As
          // CPU backend currently can't handle transpose between fused ops.
          return hasCompatibleOuterParallelLoops(
              cast<TilingInterface>(linalgOp.getOperation()),
              producerIndexingMap, rootOuterParallelLoops);
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

  if (!areOpsFusable(producer, consumer, rootOuterParallelLoops)) {
    if (!makeProducerFusableViaInterchange(operand, rootOuterParallelLoops)) {
      return false;
    }
  }
  return true;
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void
fuseRootsWithProducers(MLIRContext *context, Operation *root, unsigned groupNum,
                       DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       bool fuseWithTruncate) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);
  llvm::SmallBitVector rootOuterParallelLoops = getOuterParallelLoops(root);
  IREE::Flow::ClonableIntoDispatchOptions clonableOptions;
  clonableOptions.aggressive = options.aggressiveFusion;
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer)
        continue;
      if (IREE::Flow::isClonableIntoDispatchOp(producer, clonableOptions) ||
          hasFusionGroupsAttribute(producer) || hasRootOpAttribute(producer)) {
        continue;
      }

      if (!isFusableWithProducer(operand, rootOuterParallelLoops, options,
                                 fuseWithTruncate)) {
        continue;
      }

      SmallVector<OpOperand *> fusableUses =
          getFusableUses(context, producer, dominanceInfo,
                         /*aggressiveFusion=*/options.aggressiveFusion);
      if (fusableUses.empty() || fusableUses.front()->getOwner() != candidate)
        continue;

      appendToFusionGroup(producer, groupNum);
      worklist.push_back(producer);
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
static unsigned
decideFusableLinalgOps(Region &region, DominanceInfo const &dominanceInfo,
                       FormDispatchRegionsPassOptions const &options,
                       unsigned numRootOps = 0) {
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
          numRootOps = decideFusableLinalgOps(region, dominanceInfo, options,
                                              numRootOps);
        }
        continue;
      }

      // Start with a root operation and fuse its producers.
      if (hasFusionGroupsAttribute(&op) || !isRootOp(&op))
        continue;
      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options);
    for (Operation *root : roots) {
      int64_t rootNumber = getRootNumber(root);
      fuseRootsWithProducers(context, root, rootNumber, dominanceInfo, options,
                             /*fuseWithTruncate=*/true);
    }
  }

  // Once all root linalg ops have been tagged, put all remaining generic ops
  // into their own dispatches.
  for (Block &block : region) {
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // If it is part of a fusion group or root op, ignore it.
      if (hasFusionGroupsAttribute(&op) || hasRootOpAttribute(&op))
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

      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options,
                             /*fuseWithTruncate=*/false);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options);
    for (Operation *root : roots) {
      int64_t rootNumber = getRootNumber(root);
      fuseRootsWithProducers(context, root, rootNumber, dominanceInfo, options,
                             /*fuseWithTruncate=*/true);
    }
  }

  return numRootOps;
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
  // Step 1: Decide fusion groups (heuristic). This marks rootOps with an
  // attribute
  unsigned numRoots =
      decideFusableLinalgOps(funcOp.getFunctionBody(), dominanceInfo, options);
  SmallVector<Operation *> roots(numRoots, nullptr);
  DenseMap<unsigned, SmallVector<Operation *>> fusedOperations;

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After deciding fusion groups ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // TODO: Incrementally add ops to an empty DispatchGroupOp instead of
  // annotating fusion group IDs via attributes.
  funcOp.walk([&](Operation *op) {
    if (hasRootOpAttribute(op)) {
      roots[getRootNumber(op)] = op;
      fusedOperations[getRootNumber(op)].push_back(op);
      removeRootOpAttribute(op);
    }
    if (hasFusionGroupsAttribute(op)) {
      assert(getFusionGroups(op).size() == 1 && "expected exactly one group");
      fusedOperations[getFusionGroups(op).front()].push_back(op);
      removeFusionGroupsAttribute(op);
    }
  });

  // Step 2. Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
  for (auto [rootIndex, root] : llvm::enumerate(roots)) {

    // Sort producers and consumers topologically. All fused ops must be in the
    // same block as the root.
    SmallVector<Operation *> &currFusedOperations = fusedOperations[rootIndex];
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
        return producer->emitOpError("failed to move producer into region");
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
