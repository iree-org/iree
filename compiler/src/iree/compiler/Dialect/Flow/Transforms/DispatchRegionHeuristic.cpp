// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/DispatchRegionHeuristic.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "iree-flow-dispatch-region-heuristic"

// NOTE: These flags are added for experimental purposes only
// for developer control. These should be treated as internal
// compiler implementation details.
static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of constant that can be inlined into a "
                   "dispatch region"),
    llvm::cl::init(256));

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupAttr[] = "__fused_op__";

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Root and fusion group attribute handling
//===----------------------------------------------------------------------===//

/// Returns root op of the fusion group that `op` is contained in.
Operation *getRootOfContainingFusionGroup(const FusionGroupMapping &mapping,
                                          Operation *op) {
  for (const auto &it : mapping)
    if (llvm::find(it.second, op) != it.second.end()) return it.first;
  return nullptr;
}

bool isFusionGroupRoot(const FusionGroupMapping &mapping, Operation *op) {
  return mapping.find(op) != mapping.end();
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
             estimatedByteLength <= clInlineConstantByteLength;
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

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

/// Returns a bit vector of size number of loops of the `interfaceOp` with
/// the bits corresponding to outer parallel loops set to `true`.
static llvm::SmallBitVector getOuterParallelLoops(TilingInterface interfaceOp) {
  SmallVector<utils::IteratorType> loopIteratorTypes =
      interfaceOp.getLoopIteratorTypes();
  llvm::SmallBitVector parallelLoops(loopIteratorTypes.size());
  for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType.value() != utils::IteratorType::parallel) break;
    parallelLoops.set(iteratorType.index());
  }
  return parallelLoops;
}

/// Returns true if `map` is an identity map with zeros, i.e. if you
/// drop the result exprs that are constant zeros, the `map` will become an
/// identity.
static bool isIdentityMapWithZeros(AffineMap map) {
  if (map.getNumSymbols() != 0) return false;
  unsigned dimsSeen = 0;
  for (auto result : map.getResults()) {
    bool isValidExpr = TypeSwitch<AffineExpr, bool>(result)
                           .Case<AffineDimExpr>([&dimsSeen](auto dimExpr) {
                             if (dimExpr.getPosition() != dimsSeen)
                               return false;
                             dimsSeen++;
                             return true;
                           })
                           .Case<AffineConstantExpr>([](auto constExpr) {
                             return constExpr.getValue() == 0;
                           })
                           .Default([](AffineExpr) { return false; });
    if (!isValidExpr) return false;
  }
  return dimsSeen == map.getNumDims();
}

/// For the fusion of root op -> elementwise operation to be bufferized
/// in-place without use of extra memory, the result of the root operation
/// must be able to reuse the buffer for the result of the elementwise
/// operation. This is possible if input and output are accessed using the same
/// indexing map.
// TODO: This restriction can go away if we can vectorize always, but that has
// a long tail of tasks.
static bool isInsOperandBufferizable(OpOperand *insOperand,
                                     bool aggressiveFusion) {
  // Ignore the check if in-place bufferization is not required.
  if (aggressiveFusion) return true;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(insOperand->getOwner());
  if (!linalgOp) return false;

  AffineMap insOperandIndexingMap = linalgOp.getMatchingIndexingMap(insOperand);

  auto canTieWithOutsOperand = [&](OpOperand *outsOperand) {
    AffineMap outsOperandIndexingMap =
        linalgOp.getMatchingIndexingMap(outsOperand);

    if (outsOperandIndexingMap != insOperandIndexingMap) {
      // if (!aggressiveFusion) return false;
      // If the operand is a projected permutation a small stack might be
      // fine.
      if (!(insOperandIndexingMap.isProjectedPermutation() &&
            !insOperandIndexingMap.isPermutation())) {
        return false;
      }
    }

    // TODO(#8411): Until ops are vectorized (always), we need
    // to check that the elementtype matches for the operands to be tied.
    // For now just doing this check for convolution ops since we expect
    // contraction ops to be vectorized.
    auto producer = insOperand->get().getDefiningOp();
    if (isa<linalg::GenericOp, linalg::ConvolutionOpInterface>(producer) &&
        insOperand->get().getType().cast<ShapedType>().getElementType() !=
            outsOperand->get().getType().cast<ShapedType>().getElementType()) {
      return false;
    }
    return true;
  };
  return llvm::any_of(linalgOp.getDpsInitOperands(), canTieWithOutsOperand);
}

/// Method to check if two `linalg.generic` op with producer-consumer
/// relationship through `operand` have compatible outer-parallel loops.
static bool hasCompatibleOuterParallelLoops(
    OpOperand &operand, bool allowConsumerParallelismPessimization) {
  auto producer = operand.get().getDefiningOp<linalg::LinalgOp>();
  auto consumer = dyn_cast<linalg::LinalgOp>(operand.getOwner());
  if (!producer || !consumer) return false;

  llvm::SmallBitVector producerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(producer.getOperation()));
  llvm::SmallBitVector consumerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(consumer.getOperation()));

  if (allowConsumerParallelismPessimization) {
    if (producerParallelLoops.count() > consumerParallelLoops.count())
      return false;
  } else if (producerParallelLoops.count() != consumerParallelLoops.count()) {
    return false;
  }

  auto producerIndexingMap =
      producer.getIndexingMapMatchingResult(operand.get().cast<OpResult>());
  auto consumerIndexingMap = consumer.getMatchingIndexingMap(&operand);
  if (!producerIndexingMap.isProjectedPermutation() ||
      !consumerIndexingMap.isProjectedPermutation()) {
    return false;
  }

  /// Project out the non-parallel dimensions.
  llvm::SmallBitVector producerProjectedDims(producerParallelLoops);
  producerProjectedDims.flip();
  auto projectedProducerMap =
      getProjectedMap(producerIndexingMap, producerProjectedDims);

  llvm::SmallBitVector consumerProjectedDims(producerParallelLoops);
  consumerProjectedDims.flip();
  consumerProjectedDims.resize(consumer.getNumLoops(), true);
  auto projectedConsumerMap =
      getProjectedMap(consumerIndexingMap, consumerProjectedDims);

  return isIdentityMapWithZeros(projectedProducerMap) &&
         isIdentityMapWithZeros(projectedConsumerMap);
}

/// For all uses of an operation, finds the use that dominates all other uses.
static Optional<OpOperand *> getFusableUse(Operation *op,
                                           DominanceInfo const &dominanceInfo,
                                           bool fuseMultiUse) {
  if (!fuseMultiUse && !op->hasOneUse()) return llvm::None;

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

/// Returns true if the operands are fusable under the aggressive fusion
/// heuristics.
static bool areOpsAggresiveFusable(Operation *producer, Operation *consumer,
                                   bool allowConsumerParallelismPessimization,
                                   bool aggressiveFusion) {
  // Collect all the uses from producer to consumer.
  SmallVector<OpOperand *> allUses;
  for (OpOperand &producerUse : producer->getUses()) {
    if (producerUse.getOwner() != consumer) continue;
    allUses.push_back(&producerUse);
  }

  // Check that the consumer and producer have compatible outer parallel loops.
  if (!llvm::all_of(allUses, [&](OpOperand *operand) {
        return hasCompatibleOuterParallelLoops(
            *operand, allowConsumerParallelismPessimization);
      })) {
    return false;
  }

  // Finally only fuse if the `ins` operand can be properly bufferized.
  // TODO(#10498): Handle the multi-result case.
  return llvm::all_of(allUses, [&](OpOperand *operand) {
    return isInsOperandBufferizable(operand, aggressiveFusion);
  });
}

/// Returns true if this is a fusable use, while fusing a root with its
/// consumer.
static bool isFusableWithConsumer(OpOperand &fusedOperand,
                                  bool aggressiveFusion) {
  // Logics with aggressive fusion heuristics.
  Operation *producer = fusedOperand.get().getDefiningOp();
  Operation *consumer = fusedOperand.getOwner();

  auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
  auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer);
  if (!producerLinalgOp || !consumerLinalgOp) return false;

  // Check that the consumer is all parallel.
  if (consumerLinalgOp.getNumLoops() !=
      consumerLinalgOp.getNumParallelLoops()) {
    return false;
  }

  if (!areOpsAggresiveFusable(producer, consumer,
                              /*allowConsumerParallelismPessimization=*/true,
                              aggressiveFusion)) {
    return false;
  }

  // Check if the iteration spaces of the producer and consumer are same.
  // TODO: This is unnecessary requirement, but needed to pass tests right now
  if (!aggressiveFusion) {
    auto producerIterationSpace = producerLinalgOp.getStaticLoopRanges();
    auto consumerIterationSpace = consumerLinalgOp.getStaticLoopRanges();
    if (producerIterationSpace.size() < consumerIterationSpace.size()) {
      return false;
    }
  }
  return true;
}

/// Fuses roots with its consumers. If a root is fused with its consumer, it is
/// no more tagged as a root to aid with the dispatch region formation.
static void fuseRootsWithConsumers(FusionGroupMapping &mapping,
                                   ArrayRef<Operation *> roots,
                                   DominanceInfo const &dominanceInfo,
                                   bool aggressiveFusion) {
  SmallVector<Operation *> workList(roots.begin(), roots.end());
  // Fuse with consumers where possible.
  while (!workList.empty()) {
    Operation *currRoot = workList.pop_back_val();
    assert(isFusionGroupRoot(mapping, currRoot) &&
           "unexpected non-root op in worklist");

    // Helper function to make the consumer the root instead of the producer
    // when they are to be fused.
    auto updateRootTo = [&mapping, &currRoot](Operation *newRoot) {
      mapping[newRoot] = mapping[currRoot];
      mapping.erase(currRoot);
      mapping[newRoot].push_back(currRoot);
    };

    Optional<OpOperand *> fusableUse = getFusableUse(
        currRoot, dominanceInfo, /*fuseMultiUse=*/aggressiveFusion);
    if (!fusableUse) continue;

    // Analyse the use to see if it is fusable.
    Operation *consumerOp = fusableUse.value()->getOwner();
    if (isFusionGroupRoot(mapping, consumerOp) ||
        getRootOfContainingFusionGroup(mapping, consumerOp))
      continue;

    if (isFusableWithConsumer(*(fusableUse.value()), aggressiveFusion)) {
      updateRootTo(consumerOp);
      workList.push_back(consumerOp);
    }
  }
}

/// Method to check if the consumer of a use can be fused with its producer.
static bool isFusableWithProducer(OpOperand &operand, bool aggressiveFusion) {
  Operation *producer = operand.get().getDefiningOp();
  Operation *consumer = operand.getOwner();

  if (!isa<linalg::LinalgOp>(consumer) || !isa<linalg::LinalgOp>(producer)) {
    return false;
  }

  auto consumerLinalgOp = cast<linalg::LinalgOp>(consumer);
  if (consumerLinalgOp.isDpsInput(&operand)) {
    // Only fuse on inputs if both ops are generic ops.
    if (!aggressiveFusion || !isa<linalg::GenericOp>(consumer) ||
        !isa<linalg::GenericOp>(producer)) {
      return false;
    }
  } else if (!consumerLinalgOp.isDpsInit(&operand)) {
    return false;
  }

  return areOpsAggresiveFusable(producer, consumer,
                                /*allowConsumerParallelismPessimization=*/false,
                                aggressiveFusion);
}

/// Starting from the `root` op, traverse the operand use-def chain
/// in reverse to fuse with producers.
static void fuseRootsWithProducers(FusionGroupMapping &mapping, Operation *root,
                                   DominanceInfo const &dominanceInfo,
                                   bool aggressiveFusion) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer) continue;
      // Continue if producer is already in a fusion group or is the root of a
      // fusion group.
      if (getRootOfContainingFusionGroup(mapping, producer) ||
          isFusionGroupRoot(mapping, producer))
        continue;

      Optional<OpOperand *> fusableUse = getFusableUse(
          producer, dominanceInfo, /*fuseMultiUse=*/aggressiveFusion);
      if (!fusableUse || fusableUse.value()->getOwner() != candidate) continue;

      if (!isFusableWithProducer(operand, aggressiveFusion)) continue;

      mapping[root].push_back(producer);
      worklist.push_back(producer);
    }
  }
}

/// Some heuristic is needed to fuse a dispatchable op with root operations
/// using tile + fuse. Using some heuristic, each root operation is tagged with
/// an ID (using an IntegerAttr with name `kRootOpAttr`) and all dispatchable
/// ops to be fused with it is tagged with the same ID (using a list of
/// IntegerAttr with name `kFusionGroupAttr`). Each dispatchable operation can
/// be marked to fuse with multiple root operations (i.e. replicated). For now a
/// very simple heuristic is used below, but the mechanism should be general
/// enough to capture any heuristic.
FusionGroupMapping decideFusableLinalgOps(FunctionOpInterface funcOp,
                                          DominanceInfo const &dominanceInfo,
                                          bool aggressiveFusion) {
  FusionGroupMapping mapping;
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
      if (getRootOfContainingFusionGroup(mapping, &op) || !isRootOp(&op))
        continue;
      mapping[&op] = SmallVector<Operation *>();
      fuseRootsWithProducers(mapping, &op, dominanceInfo, aggressiveFusion);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(mapping, roots, dominanceInfo, aggressiveFusion);
  }

  // Once all root linalg ops have been tagged, put all remaining generic ops
  // into their own dispatches.
  for (Block &block : funcOp.getFunctionBody()) {
    SmallVector<Operation *> roots;
    for (Operation &op : llvm::reverse(block)) {
      // If it is part of a fusion group or root op, ignore it.
      if (getRootOfContainingFusionGroup(mapping, &op) ||
          isFusionGroupRoot(mapping, &op))
        continue;
      // Only look for Linalg ops here. Avoid moving `linalg.fill` that aren't
      // fused with anything else into their own dispatches since it is better
      // to convert them to splats.
      if (!isa<linalg::LinalgOp>(op) || isa<linalg::FillOp>(op)) continue;

      mapping[&op] = SmallVector<Operation *>();
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(mapping, roots, dominanceInfo, aggressiveFusion);
  }

  return mapping;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
