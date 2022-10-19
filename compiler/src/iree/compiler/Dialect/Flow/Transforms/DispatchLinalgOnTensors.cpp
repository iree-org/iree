// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/DispatchLinalgOnTensors.h"

#include <deque>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Patterns.h"
#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-flow-dispatch-linalg-on-tensors"

// NOTE: These flags are added for experimental purposes only
// for developer control. These should be treated as internal
// compiler implementation details.
static llvm::cl::opt<int> clInlineConstantByteLength(
    "iree-flow-inline-constants-max-byte-length",
    llvm::cl::desc("Maximum byte-length of constant that can be inlined into a "
                   "dispatch region"),
    llvm::cl::init(256));

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

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
      if (tensorType.isDynamicDim(i)) ++ctr;
    rewriter.replaceOp(dimOp, dynamicDims[ctr]);
  }

  return success();
}

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
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
  return isa<TilingInterface>(op) ||
         isa<LinalgExt::SetEncodingOp, LinalgExt::UnsetEncodingOp>(op);
}

/// Operations that are cloned into dispatch regions formed with other
/// operations as roots.
bool isClonableIntoDispatchOp(Operation *op) {
  // TODO(#8637): `tensor.collapse_shape` and `tensor.expand_shape` are
  // trivially clonable too, but they cause problems
  // with bufferization. Make them clonable when fixed.
  if (isa<AffineApplyOp, arith::IndexCastOp, linalg::FillOp, tensor::EmptyOp,
          tensor::CastOp, tensor::ExtractOp, tensor::ExtractSliceOp,
          tensor::PadOp>(op)) {
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
// Methods for getting the workload information for dispatch region creation.
//===----------------------------------------------------------------------===//

/// Compute the workload to use for the workgroup based on the root op.
static SmallVector<Value> getWorkloadForRootOp(OpBuilder &builder,
                                               Operation *rootOp) {
  // Compute workgroup count to use for the dispatch op. These are the ranges
  // of the outermost parallel loops that can be distributed.
  Location loc = rootOp->getLoc();
  SmallVector<Range> loopRanges = getLoopRanges(rootOp, loc, builder);
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

//===---------------------------------------------------------------------===//
// Methods to legalize a dispatch region op, i.e. make it isolated from above.
//===---------------------------------------------------------------------===//

/// Checks if the `Value` has a use within the dispatch that is unfusable.
static bool hasUnfusableUseInDispatch(Value v, Operation *dispatchOp) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    Operation *ownerWorkgroupsOp =
        user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>();
    Operation *ownerRegionOp =
        user->getParentOfType<IREE::Flow::DispatchRegionOp>();
    Operation *owner = ownerWorkgroupsOp ? ownerWorkgroupsOp : ownerRegionOp;

    // Ignore uses outside of dispatch workgroups op.
    if (owner != dispatchOp) continue;

    // Cannot fuse producer of `dest` with `tensor.insert_slice`.
    if (auto insertSliceUser = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (insertSliceUser.getDest() == v) return true;
    }
  }
  return false;
}

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

  // Fuse unset_encoding operations with `tensor.extract_slice`.
  if (isa<LinalgExt::UnsetEncodingOp>(producer) &&
      isa<tensor::ExtractSliceOp>(consumer)) {
    auto sliceOp = cast<tensor::ExtractSliceOp>(consumer);
    return llvm::all_of(
               sliceOp.getMixedOffsets(),
               [](OpFoldResult ofr) { return isConstantIntValue(ofr, 0); }) &&
           llvm::all_of(sliceOp.getMixedStrides(), [](OpFoldResult ofr) {
             return isConstantIntValue(ofr, 1);
           });
  }

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
static void fuseRootsWithConsumers(MLIRContext *context,
                                   ArrayRef<Operation *> roots,
                                   DominanceInfo const &dominanceInfo,
                                   bool aggressiveFusion) {
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

    Optional<OpOperand *> fusableUse = getFusableUse(
        currRoot, dominanceInfo, /*fuseMultiUse=*/true);
    if (!fusableUse) continue;

    // Analyse the use to see if it is fusable.
    Operation *consumerOp = fusableUse.value()->getOwner();
    if (hasRootOpAttribute(consumerOp) ||
        hasFusionGroupsAttribute(consumerOp)) {
      continue;
    }

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
    bool fuseUses = false;
    if (auto linalgRoot = dyn_cast<linalg::GenericOp>(producer)) {
      SmallVector<unsigned> dims;
      linalgRoot.getReductionDims(dims);
      fuseUses = (dims.size() == 1 &&
                  (linalgRoot.getStaticLoopRanges()[dims[0]] % (64 * 4) == 0) &&
                  (linalgRoot.getStaticLoopRanges()[dims[0]] <= 4096));
    }
    // Only fuse on inputs if both ops are generic ops.
    if (!fuseUses || !isa<linalg::GenericOp>(consumer) ||
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
static void fuseRootsWithProducers(MLIRContext *context, Operation *root,
                                   unsigned groupNum,
                                   DominanceInfo const &dominanceInfo,
                                   bool aggressiveFusion) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *candidate = worklist.pop_back_val();
    for (OpOperand &operand : candidate->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer) continue;
      if (hasFusionGroupsAttribute(producer) || hasRootOpAttribute(producer)) {
        continue;
      }

      Optional<OpOperand *> fusableUse = getFusableUse(
          producer, dominanceInfo, /*fuseMultiUse=*/false);
      if (!fusableUse || fusableUse.value()->getOwner() != candidate) continue;

      if (!isFusableWithProducer(operand, aggressiveFusion)) continue;

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
static unsigned decideFusableLinalgOps(FunctionOpInterface funcOp,
                                       DominanceInfo const &dominanceInfo,
                                       bool aggressiveFusion) {
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

      fuseRootsWithProducers(context, &op, newGroup, dominanceInfo,
                             aggressiveFusion);
      roots.push_back(&op);
    }
    roots = llvm::to_vector(llvm::reverse(roots));
    fuseRootsWithConsumers(context, roots, dominanceInfo, aggressiveFusion);
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
    fuseRootsWithConsumers(context, roots, dominanceInfo, aggressiveFusion);
  }

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

static void buildSetEncodingWorkloadRegion(OpBuilder &builder, Location loc,
                                           ArrayRef<BlockArgument> args) {
  auto numWorkgroupsOp =
      builder.create<Flow::DispatchWorkgroupCountFromSetEncodingOp>(loc, args);
  builder.create<Flow::ReturnOp>(loc, numWorkgroupsOp.getResults());
}

static void buildDefaultWorkloadRegion(OpBuilder &builder, Location loc,
                                       ArrayRef<BlockArgument> args) {
  auto numWorkgroupsOp =
      builder.create<Flow::DispatchWorkgroupCountFromDagRootOp>(loc, args);
  builder.create<Flow::ReturnOp>(loc, numWorkgroupsOp.getResults());
}

/// Computes the workload and provides a workload region builder for the given
/// root op.
static FailureOr<Flow::WorkloadBuilder> getWorkloadBuilder(OpBuilder &builder,
                                                           Operation *rootOp) {
  Flow::WorkloadBuilder result;

  // Compute workload (before entering the dispatch region).
  OpBuilder::InsertionGuard g(builder);
  SmallVector<Value> workload;
  builder.setInsertionPoint(rootOp);
  FailureOr<SmallVector<Value>> maybeWorkload =
      getWorkloadForRootOp(builder, rootOp);
  if (failed(maybeWorkload)) return failure();
  result.workload = *maybeWorkload;

  // The workload region of the WorkgroupsOp is populated by the
  // `regionBuilder` during ConvertRegionToWorkgroups .
  if (isa<LinalgExt::SetEncodingOp>(rootOp)) {
    result.regionBuilder = buildSetEncodingWorkloadRegion;
  } else {
    result.regionBuilder = buildDefaultWorkloadRegion;
  }

  return result;
}

/// Create Flow::DispatchGroupsOps based on a fusion heuristic.
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> createFusionGroups(
    TensorDimTrackingRewriter &rewriter, FunctionOpInterface funcOp,
    DominanceInfo const &dominanceInfo, bool generateWorkloadRegion,
    bool aggressiveFusion) {
  // Decide fusion groups (heuristic).
  unsigned numRoots =
      decideFusableLinalgOps(funcOp, dominanceInfo, aggressiveFusion);
  SmallVector<Operation *> roots(numRoots, nullptr);
  DenseMap<unsigned, SmallVector<Operation *>> producers;

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

  // Create a DispatchRegionOp for every fusion group.
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<Flow::DispatchRegionOp> regionOps;
  DenseMap<Flow::DispatchRegionOp, Optional<Flow::WorkloadBuilder>>
      workloadBuilders;
  for (const auto &it : llvm::enumerate(roots)) {
    // Compute workload.
    Optional<Flow::WorkloadBuilder> workloadBuilder = llvm::None;
    if (generateWorkloadRegion) {
      auto maybeBuilder = getWorkloadBuilder(rewriter, /*rootOp=*/it.value());
      if (failed(maybeBuilder)) return failure();
      workloadBuilder = *maybeBuilder;
    }

    // Simplify tensor::DimOps.
    SmallVector<tensor::DimOp> dimOps = rewriter.getTensorDimOps();
    if (failed(simplifyDimOps(rewriter, dimOps))) return failure();

    // Create fusion group.
    Flow::DispatchRegionOp regionOp;
    auto maybeRegionOp = Flow::wrapOpInDispatchRegion(rewriter, it.value());
    if (failed(maybeRegionOp)) return failure();
    regionOp = *maybeRegionOp;

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

    workloadBuilders[regionOp] = workloadBuilder;
    regionOps.push_back(regionOp);
  }

  // Clone additional producers and rewrite to DispatchWorkgroupsOp.
  SmallVector<Flow::DispatchWorkgroupsOp> result;
  for (auto regionOp : regionOps) {
    if (failed(cloneProducers(rewriter, regionOp))) return failure();
    auto maybeWorkgroupOp =
        Flow::rewriteFlowDispatchRegionToFlowDispatchWorkgroups(
            regionOp, rewriter, workloadBuilders[regionOp]);
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
  Optional<Flow::WorkloadBuilder> workloadBuilder = llvm::None;
  if (generateWorkloadRegion) {
    auto maybeBuilder = getWorkloadBuilder(rewriter, op);
    if (failed(maybeBuilder)) return failure();
    workloadBuilder = *maybeBuilder;
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
      *regionOp, rewriter, workloadBuilder);
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

/// Wrap all ops of the given types that are direct children of the given op in
/// DispatchWorkgroupsOps.
template <typename... OpTys>
static FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> wrapInWorkgroupsOp(
    TensorDimTrackingRewriter &rewriter, Operation *op,
    bool generateWorkloadRegion) {
  // Find ops of type OpTys.
  SmallVector<Operation *> rootOps;
  for (Region &r : op->getRegions())
    for (Block &b : r.getBlocks())
      for (Operation &op : b)
        if (isa<OpTys...>(&op)) rootOps.push_back(&op);

  // Wrap ops in DispatchWorkgroupsOps.
  return wrapInWorkgroupsOp(rewriter, rootOps, generateWorkloadRegion);
}

/// Return `true` if the given op is contained in DispatchWorkgroupsOp or in a
/// DispatchRegionOp.
static bool isInDispatchRegion(Operation *op) {
  return op->getParentOfType<Flow::DispatchWorkgroupsOp>() ||
         op->getParentOfType<Flow::DispatchRegionOp>();
}

/// Rewrite top-level InsertSliceOps to FlowUpdateOps or wrap them in a
/// dispatch region.
LogicalResult convertInsertSliceOps(
    TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<Flow::DispatchWorkgroupsOp> &workgroupsOps,
    bool generateWorkloadRegion) {
  // Find eligible InsertSliceOps.
  SmallVector<tensor::InsertSliceOp> insertSliceOps;
  funcOp.walk([&](tensor::InsertSliceOp op) {
    if (!isInDispatchRegion(op)) insertSliceOps.push_back(op);
  });

  // Rewrite InsertSliceOps to FlowUpdateOps.
  SmallVector<Operation *> remainingInsertSliceOps;
  for (tensor::InsertSliceOp insertSliceOp : insertSliceOps)
    if (failed(
            Flow::convertInsertSliceOpToFlowUpdateOp(rewriter, insertSliceOp)))
      remainingInsertSliceOps.push_back(insertSliceOp);

  // Create a DispatchWorkgroupsOp for every remaining InsertSliceOp.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingInsertSliceOps,
                         generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

/// Rewrite top-level ExtractSliceOps to FlowSliceOps or wrap them in a
/// dispatch region.
LogicalResult convertExtractSliceOps(
    TensorDimTrackingRewriter &rewriter, mlir::FunctionOpInterface funcOp,
    SmallVector<Flow::DispatchWorkgroupsOp> &workgroupsOps,
    bool generateWorkloadRegion) {
  // Find eligible ExtractSliceOps.
  SmallVector<tensor::ExtractSliceOp> extractSliceOps;
  funcOp.walk([&](tensor::ExtractSliceOp op) {
    if (!isInDispatchRegion(op)) extractSliceOps.push_back(op);
  });

  // Rewrite ExtractSliceOps to FlowSliceOps.
  SmallVector<Operation *> remainingExtractSliceOps;
  for (tensor::ExtractSliceOp extractSliceOp : extractSliceOps)
    if (failed(
            Flow::convertExtractSliceOpToFlowSliceOp(rewriter, extractSliceOp)))
      remainingExtractSliceOps.push_back(extractSliceOp);

  // Create a DispatchWorkgroupsOp for every remaining ExtractSliceOp.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp(rewriter, remainingExtractSliceOps,
                         generateWorkloadRegion);
  if (failed(newWorkgroupsOps)) return failure();
  workgroupsOps.append(newWorkgroupsOps->begin(), newWorkgroupsOps->end());

  return success();
}

namespace {
/// Pass declaration.
struct DispatchLinalgOnTensorsPass
    : public DispatchLinalgOnTensorsBase<DispatchLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }
  DispatchLinalgOnTensorsPass(bool aggressiveFusion,
                              bool generateWorkloadRegion) {
    this->aggressiveFusion = aggressiveFusion;
    this->generateWorkloadRegion = generateWorkloadRegion;
  }
  DispatchLinalgOnTensorsPass(const DispatchLinalgOnTensorsPass &pass)
      : DispatchLinalgOnTensorsPass(pass.aggressiveFusion,
                                    pass.generateWorkloadRegion) {}
  void runOnOperation() override;

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};
};
}  // namespace

void DispatchLinalgOnTensorsPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();

  DominanceInfo const &dominanceInfo = getAnalysis<DominanceInfo>();
  TensorDimTrackingRewriter rewriter(funcOp);

  // Step 1: Create a DispatchWorkgroupsOp for every fusion group.
  auto maybeWorkgroupsOps =
      createFusionGroups(rewriter, funcOp, dominanceInfo,
                         generateWorkloadRegion, aggressiveFusion);
  if (failed(maybeWorkgroupsOps)) return signalPassFailure();
  SmallVector<Flow::DispatchWorkgroupsOp> workgroupsOps = *maybeWorkgroupsOps;

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After first step of dispatch region formation ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Step 2: Rewrite InsertSliceOps to FlowUpdateOps.
  if (failed(convertInsertSliceOps(rewriter, funcOp, workgroupsOps,
                                   generateWorkloadRegion)))
    return signalPassFailure();

  // Step 3: Rewrite ExtractSliceOps to FlowUpdateOps.
  if (failed(convertExtractSliceOps(rewriter, funcOp, workgroupsOps,
                                    generateWorkloadRegion)))
    return signalPassFailure();

  // Step 4: Create a DispatchWorkgroupsOp for certain other ops.
  FailureOr<SmallVector<Flow::DispatchWorkgroupsOp>> newWorkgroupsOps =
      wrapInWorkgroupsOp<LinalgExt::SetEncodingOp, LinalgExt::UnsetEncodingOp>(
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
createDispatchLinalgOnTensorsPass(bool aggressiveFusion,
                                  bool generateWorkloadRegion) {
  return std::make_unique<DispatchLinalgOnTensorsPass>(aggressiveFusion,
                                                       generateWorkloadRegion);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
