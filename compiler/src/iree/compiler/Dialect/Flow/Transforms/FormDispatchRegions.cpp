// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-flow-form-dispatch-regions"

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

//===----------------------------------------------------------------------===//
// Definition of TensorDimTrackingRewriter
//===----------------------------------------------------------------------===//

namespace mlir {

TensorDimTrackingRewriter::TensorDimTrackingRewriter(Operation *op)
    : IRRewriter(op->getContext()) {
  setListener(this);
  op->walk([&](tensor::DimOp dimOp) { dimOps.insert(dimOp.getOperation()); });
}
SmallVector<tensor::DimOp> TensorDimTrackingRewriter::getTensorDimOps() {
  SmallVector<tensor::DimOp> result;
  for (Operation *op : dimOps)
    result.push_back(cast<tensor::DimOp>(op));
  return result;
}
void TensorDimTrackingRewriter::notifyOperationRemoved(Operation *op) {
  IRRewriter::Listener::notifyOperationRemoved(op);
  if (isa<tensor::DimOp>(op))
    dimOps.erase(op);
}

void TensorDimTrackingRewriter::notifyOperationInserted(Operation *op,
                                                        InsertPoint previous) {
  IRRewriter::Listener::notifyOperationInserted(op, previous);
  if (isa<tensor::DimOp>(op))
    dimOps.insert(op);
}

} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

LogicalResult simplifyDimOps(RewriterBase &rewriter,
                             const SmallVector<tensor::DimOp> &dimOps) {
  for (tensor::DimOp dimOp : dimOps) {
    // Only DimOps with static indices are supported.
    std::optional<int64_t> idx = dimOp.getConstantIndex();
    if (!idx.has_value())
      continue;
    // Only DimOps with ranked tensors are supported.
    auto tensorType =
        llvm::dyn_cast<RankedTensorType>(dimOp.getSource().getType());
    if (!tensorType)
      continue;

    if (!tensorType.isDynamicDim(*idx)) {
      // Rewrite static dimension with constant.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(dimOp);
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
      if (tensorType.isDynamicDim(i))
        ++ctr;
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
/// are not unpacked by the producer tensor::UnPackOp. This means the reduced
/// dimensions of the unpack result are not part of the inner_dims_pos.
static bool hasNoPackedReductionDimensions(linalg::LinalgOp linalgOp,
                                           Operation *producer) {
  auto unpack = dyn_cast<tensor::UnPackOp>(producer);
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
    auto producer = operand.getDefiningOp<tensor::UnPackOp>();
    return producer && hasNoPackedReductionDimensions(linalgOp, producer);
  });
}

/// Operations that are treated as root operations for dispatch region
/// formation.
static bool isRootOp(Operation *op) {
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
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
    return !isa<tensor::PadOp, tensor::PackOp>(op);
  }
  return isa<LinalgExt::UnsetEncodingOp, tensor::UnPackOp>(op);
}

/// Returns true if the operation is a `pack` op or a `set_encoding` op that
/// has pack semantics.
// TODO(ravishankarm): This seems like a use case for an interface.
static bool isPackLikeOp(Operation *op) {
  return isa<IREE::LinalgExt::SetEncodingOp, tensor::PackOp>(op);
}

/// Returns true if the operation is an `unpack` op or an `unset_encoding` op,
/// or an `extract_slice` op whose source operand matches those criteria,
/// recursively.
/// The idea is that we want to ensure that `extract_slice` ops can't prevent
/// fusion between a `unset_encoding` producer and some linalg consumer. In
///   %0 = unset_encoding ...
///   %1 = extract_slice %0 ...
///   %2 = linalg.generic ins(%1) ...
/// we are not content to be fusing %1 into %0, we also want to be fusing %2,
/// so we want to prevent %1 from acting as a consumer fusion barrier.
static bool isUnpackLikeOpViaExtractSliceOps(Operation *op) {
  if (isa<IREE::LinalgExt::UnsetEncodingOp, tensor::UnPackOp>(op)) {
    return true;
  }
  if (isa<tensor::ExtractSliceOp>(op)) {
    Value source = op->getOperand(0);
    Operation *producer = source.getDefiningOp();
    if (isUnpackLikeOpViaExtractSliceOps(producer)) {
      return true;
    }
  }
  return false;
}

/// Since `iree_linalg_ext.set_encoding` doesnt have padding semantics a
/// `tensor.pad` is introduced to get the shapes of the input and output to
/// match. The `tensor.pad` -> `set_encoding` can be folded later on into a
/// single `tensor.pack` operation. But it means the fusion has to try to keep
/// these in the same dispatch.
// TODO(ravishankarm): Maybe make `set_encoding` have pad semantics that can be
// explicitly broken down if needed.
static bool isPadUsedInSetEncoding(tensor::PadOp padOp) {
  return llvm::any_of(padOp->getUsers(), [](Operation *user) {
    return isa<IREE::LinalgExt::SetEncodingOp>(user);
  });
}

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

/// Returns a bit vector of size number of loops of the `interfaceOp` with
/// the bits corresponding to outer parallel loops set to `true`.
static llvm::SmallBitVector getOuterParallelLoops(Operation *op) {
  if (auto setEncodingOp = dyn_cast<IREE::LinalgExt::SetEncodingOp>(op)) {
    return llvm::SmallBitVector(setEncodingOp.getResultType().getRank(), true);
  }
  if (auto unsetEncodingOp = dyn_cast<IREE::LinalgExt::UnsetEncodingOp>(op)) {
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
  for (int pos : llvm::seq<int>(0, rootOuterParallelLoop.size())) {
    // If we reach the end of the outer loops of the root, break out of the
    // loop.
    if (!rootOuterParallelLoop.test(pos))
      break;
    // If the root loop is parallel, the candidate loop should also be parallel.
    if (pos >= candidateOuterParallelLoop.size() ||
        !candidateOuterParallelLoop.test(pos))
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
  auto producer = operand.get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(operand.getOwner());
  if (!producer || !consumer)
    return false;

  auto producerIndexingMap = producer.getIndexingMapMatchingResult(
      llvm::cast<OpResult>(operand.get()));
  auto consumerIndexingMap = consumer.getMatchingIndexingMap(&operand);

  return hasCompatibleOuterParallelLoops(
             cast<TilingInterface>(producer.getOperation()),
             producerIndexingMap, rootOuterParallelLoops) &&
         hasCompatibleOuterParallelLoops(
             cast<TilingInterface>(consumer.getOperation()),
             consumerIndexingMap, rootOuterParallelLoops);
}

/// For all uses of an operation, finds the use that dominates all other uses.
static std::optional<OpOperand *>
getFusableUse(Operation *op, DominanceInfo const &dominanceInfo,
              bool fuseMultiUse) {
  if (!fuseMultiUse && llvm::count_if(op->getUses(), [](OpOperand &use) {
                         return !isa<tensor::DimOp>(use.getOwner());
                       }) != 1) {
    return std::nullopt;
  }

  // Collect non-dim users.
  SmallVector<Operation *> nonDimUsers;
  for (Operation *user : op->getUsers()) {
    if (isa<tensor::DimOp>(user))
      continue;
    nonDimUsers.push_back(user);
  }

  // Find the use in a non-dim user that dominates all other non-dim users.
  for (auto &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (isa<tensor::DimOp>(user))
      continue;
    if (llvm::all_of(nonDimUsers, [&](Operation *c) {
          return dominanceInfo.dominates(user, c);
        })) {
      return &use;
    }
  }
  return std::nullopt;
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
                      FormDispatchRegionsOptions const &options) {
  Operation *producer = fusedOperand.get().getDefiningOp();
  Operation *consumer = fusedOperand.getOwner();

  // Fuse unset_encoding operations with `tensor.extract_slice` and elementwise
  // generic ops.
  if (isUnpackLikeOpViaExtractSliceOps(producer)) {
    // Fuse `unset_encoding` -> `extract_slice` op since they get folded into
    // `unpack` on materialization.
    if (isa<tensor::ExtractSliceOp>(consumer)) {
      auto sliceOp = cast<tensor::ExtractSliceOp>(consumer);
      return llvm::all_of(
                 sliceOp.getMixedOffsets(),
                 [](OpFoldResult ofr) { return isConstantIntValue(ofr, 0); }) &&
             llvm::all_of(sliceOp.getMixedStrides(), [](OpFoldResult ofr) {
               return isConstantIntValue(ofr, 1);
             });
    }
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
          // Make sure the producer op has an identitiy result indexing map. As
          // CPU backend currently can't handle tranpose between fused ops.
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
    if (options.fusePadWithProducers || isPadUsedInSetEncoding(padOp)) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  // TODO(#16025): Enable mmt4d fusion. It is disabled because the backends
  // can not set multi lowering_config properly. See the issue for more details.
  if (isa<linalg::Mmt4DOp>(producer)) {
    return false;
  }

  auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
  auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer);
  if (!producerLinalgOp || !consumerLinalgOp)
    return false;

  // Check that the consumer is all parallel.
  if (consumerLinalgOp.getNumLoops() !=
      consumerLinalgOp.getNumParallelLoops()) {
    return false;
  }

  if (!areOpsFusable(producer, consumer, rootOuterParallelLoops)) {
    return false;
  }

  // While fusing with consumer, the result of the root might not be the final
  // result of the dispatch. To avoid a stack allocation we have to ensure that
  // all operations can bufferize without needing additional memory.
  for (OpOperand *inputOperand : consumerLinalgOp.getDpsInputOperands()) {
    if (inputOperand->get().getDefiningOp() != producer)
      continue;
    if (isa<linalg::ConvolutionOpInterface>(producer) &&
        !llvm::any_of(
            consumerLinalgOp.getDpsInitsMutable(), [&](OpOperand &initOperand) {
              return canUseInOperandAsInitOperand(inputOperand, &initOperand);
            })) {
      return false;
    }
  }

  // Check if the iteration spaces of the producer and consumer are same.
  // TODO(#12664): This is unnecessary requirement, but we need a better config
  // to tile the consumer with a larger iteration space.
  auto producerIterationSpace = producerLinalgOp.getStaticLoopRanges();
  auto consumerIterationSpace = consumerLinalgOp.getStaticLoopRanges();
  if (producerIterationSpace.size() < consumerIterationSpace.size()) {
    return false;
  }

  return true;
}

/// Fuses roots with its consumers. If a root is fused with its consumer, it is
/// no more tagged as a root to aid with the dispatch region formation.
static void fuseRootsWithConsumers(MLIRContext *context,
                                   ArrayRef<Operation *> roots,
                                   DominanceInfo const &dominanceInfo,
                                   FormDispatchRegionsOptions const &options) {
  // Fuse with consumers where possible.
  for (Operation *root : roots) {
    SmallVector<Operation *> workList;
    llvm::SmallBitVector rootOuterParallelLoops = getOuterParallelLoops(root);
    workList.push_back(root);
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

      std::optional<OpOperand *> fusableUse = getFusableUse(
          currRoot, dominanceInfo, /*fuseMultiUse=*/options.fuseMultiUse);
      if (!fusableUse)
        continue;

      // Analyse the use to see if it is fusable.
      Operation *consumerOp = fusableUse.value()->getOwner();
      if (hasRootOpAttribute(consumerOp) ||
          hasFusionGroupsAttribute(consumerOp)) {
        continue;
      }

      if (isFusableWithConsumer(*(fusableUse.value()), rootOuterParallelLoops,
                                options)) {
        updateRootTo(consumerOp);
        workList.push_back(consumerOp);
      }
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool
isFusableWithProducer(OpOperand &operand,
                      const llvm::SmallBitVector &rootOuterParallelLoops,
                      FormDispatchRegionsOptions const &options) {
  Operation *producer = operand.get().getDefiningOp();
  Operation *consumer = operand.getOwner();

  if (auto padOp = dyn_cast<tensor::PadOp>(consumer)) {
    if (options.fusePadWithProducers || isPadUsedInSetEncoding(padOp)) {
      return isa<linalg::LinalgOp>(producer);
    }
    return false;
  }

  if (options.fusePadWithConsumers && isa<tensor::PadOp>(producer) &&
      isa<linalg::ConvolutionOpInterface>(consumer)) {
    return true;
  }

  if (isPackLikeOp(consumer)) {
    return TypeSwitch<Operation *, bool>(producer)
        .Case<tensor::PadOp>([&](auto padOp) { return true; })
        .Case<linalg::LinalgOp>([&](auto linalgOp) {
          if (auto packOp = dyn_cast<tensor::PackOp>(consumer)) {
            // TODO(#12746): fusion of pack with dynamic inner tile size
            // causes an error in backend. Disable for now.
            if (!packOp.getInnerTiles().empty()) {
              return false;
            }
          }
          auto producerIndexingMap = linalgOp.getIndexingMapMatchingResult(
              llvm::cast<OpResult>(operand.get()));
          // Make sure the producer op has an identitiy result indexing map. As
          // CPU backend currently can't handle tranpose between fused ops.
          return hasCompatibleOuterParallelLoops(
              cast<TilingInterface>(linalgOp.getOperation()),
              producerIndexingMap, rootOuterParallelLoops);
        })
        .Default([](Operation *) { return false; });
  }

  if (!isa<linalg::LinalgOp>(consumer) || !isa<linalg::LinalgOp>(producer)) {
    return false;
  }

  return areOpsFusable(producer, consumer, rootOuterParallelLoops);
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void fuseRootsWithProducers(MLIRContext *context, Operation *root,
                                   unsigned groupNum,
                                   DominanceInfo const &dominanceInfo,
                                   FormDispatchRegionsOptions const &options) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);
  llvm::SmallBitVector rootOuterParallelLoops = getOuterParallelLoops(root);
  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer)
        continue;
      if (isClonableIntoDispatchOp(producer) ||
          hasFusionGroupsAttribute(producer) || hasRootOpAttribute(producer)) {
        continue;
      }

      std::optional<OpOperand *> fusableUse = getFusableUse(
          producer, dominanceInfo, /*fuseMultiUse=*/options.fuseMultiUse);
      if (!fusableUse || fusableUse.value()->getOwner() != candidate)
        continue;

      if (!isFusableWithProducer(operand, rootOuterParallelLoops, options)) {
        continue;
      }

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
                       FormDispatchRegionsOptions const &options,
                       unsigned numRootOps = 0) {
  MLIRContext *context = region.getContext();
  OpBuilder builder(context);
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

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options);
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
      if (!isa<linalg::LinalgOp, tensor::PadOp, tensor::PackOp,
               IREE::LinalgExt::SetEncodingOp>(op) ||
          isa<linalg::FillOp>(op) || isDequantizationLikeOp(&op)) {
        continue;
      }

      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo, options);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, options);
  }

  return numRootOps;
}

//===----------------------------------------------------------------------===//
// Dispatch region formation
//===----------------------------------------------------------------------===//

/// Create Flow::DispatchGroupsOps based on a fusion heuristic.
static LogicalResult
createFusionGroups(TensorDimTrackingRewriter &rewriter,
                   mlir::FunctionOpInterface funcOp,
                   DominanceInfo const &dominanceInfo,
                   FormDispatchRegionsOptions const &options) {
  // Step 1: Decide fusion groups (heuristic). This marks rootOps with an
  // attribute
  unsigned numRoots =
      decideFusableLinalgOps(funcOp.getFunctionBody(), dominanceInfo, options);
  SmallVector<Operation *> roots(numRoots, nullptr);
  DenseMap<unsigned, SmallVector<Operation *>> producers;

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
      removeRootOpAttribute(op);
    }
    if (hasFusionGroupsAttribute(op)) {
      assert(getFusionGroups(op).size() == 1 && "expected exactly one group");
      producers[getFusionGroups(op).front()].push_back(op);
      removeFusionGroupsAttribute(op);
    }
  });

  // Step 2. Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Flow::DispatchRegionOp> regionOps;
  for (const auto &it : llvm::enumerate(roots)) {
    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(iree_compiler::IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
        return failure();
      }
    }

    // Create fusion group.
    Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, it.value());
    if (failed(maybeRegionOp))
      return failure();
    regionOp = *maybeRegionOp;

    // Sort producers topologically. All producers must be in the same block
    // as the root.
    bool sortResult = mlir::computeTopologicalSorting(producers[it.index()]);
    (void)sortResult;
    assert(sortResult && "could not compute topological sorting");

    // Move ops into the region.
    for (Operation *producer : llvm::reverse(producers[it.index()])) {
      // Simplify tensor::DimOps.
      {
        SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
        if (failed(
                iree_compiler::IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
          return failure();
        }
      }

      auto newRegionOp =
          movePrecedingOpsIntoDispatchRegion(rewriter, producer, regionOp);
      if (failed(newRegionOp))
        return failure();
      regionOp = *newRegionOp;
    }
    // Simplify tensor::DimOps.
    {
      SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
      if (failed(iree_compiler::IREE::Flow::simplifyDimOps(rewriter, dimOps))) {
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
struct FormDispatchRegionsPass
    : public FormDispatchRegionsBase<FormDispatchRegionsPass> {
  using FormDispatchRegionsBase<
      FormDispatchRegionsPass>::FormDispatchRegionsBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::Flow::FlowDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }
  /// These constructors are auto-generated in `Passes.h.inc` from the
  /// tablegen file if `GEN_PASS_DEF_FORMDISPATCHREGIONS` is defined
  /// before including that file. Doing that requires changing
  /// all Flow passes to use similar mechanism.
  // TODO(ravishankarm): Modify Flow passes to use the auto-generated
  // options struct.
  FormDispatchRegionsPass() {}
  FormDispatchRegionsPass(const FormDispatchRegionsOptions &options)
      : FormDispatchRegionsPass() {
    fuseMultiUse = options.fuseMultiUse;
    generateWorkloadRegion = options.generateWorkloadRegion;
    fusePadWithConsumers = options.fusePadWithConsumers;
    fusePadWithProducers = options.fusePadWithProducers;
  }
  FormDispatchRegionsPass(const FormDispatchRegionsPass &other)
      : FormDispatchRegionsPass(FormDispatchRegionsOptions{
            other.fuseMultiUse, other.generateWorkloadRegion,
            other.fusePadWithConsumers, other.fusePadWithProducers}) {}

  void runOnOperation() override;
};
} // namespace

/// Create dispatch.region Ops based on a fusion heuristic.
void FormDispatchRegionsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
  TensorDimTrackingRewriter rewriter(funcOp);
  FormDispatchRegionsOptions options{fuseMultiUse, generateWorkloadRegion,
                                     fusePadWithConsumers,
                                     fusePadWithProducers};
  if (failed(createFusionGroups(rewriter, funcOp, dominanceInfo, options))) {
    funcOp->emitOpError("failed to create fusion groups");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchRegionsPass(FormDispatchRegionsOptions options) {
  return std::make_unique<FormDispatchRegionsPass>(options);
}
} // namespace mlir::iree_compiler::IREE::Flow
