// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/DestructiveUpdateUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-flow-dispatch-linalg-on-tensors"

// TODO(ravishankarm): Prune this list. These flags should go away ASAP!!

static llvm::cl::list<int64_t> clLinalgOnTensorsTileSizes(
    "iree-flow-dispatch-linalg-on-tensors-tile-sizes",
    llvm::cl::desc("Comma-separated list of tile sizes for tiling on tensors"),
    llvm::cl::CommaSeparated);

// TODO(#5040): This works for the most part but the downstream bufferization
// needs to be sorted out before this can be made the default. Remove after
// making this default.
static llvm::cl::opt<bool> clEnableOperandFusion(
    "iree-flow-dispatch-formation-enable-operand-fusion",
    llvm::cl::desc(
        "Enable fusing operand producers during dispatch region formation"),
    llvm::cl::init(false));

static const char kRootOpAttr[] = "__root_op__";
static const char kFusionGroupsAttr[] = "__fused_op__";

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static unsigned kNumMaxParallelDims = 3;

//===----------------------------------------------------------------------===//
// Root and fusion group attribute handling
//===----------------------------------------------------------------------===//

/// Returns true if an op has a root operation.
static bool hasRootOpAttribute(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<IntegerAttr>(kRootOpAttr));
}
/// Removes root attribute. Asserts if root attribute is not present.
static void removeRootOpAttribute(Operation *op) {
  assert(op->hasAttr(kRootOpAttr) &&
         "removing root attribute from op that is not a root attribute");
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
// Utility methods
//===----------------------------------------------------------------------===//

/// Returns the number of consecutive outer loops that are "parallel". This is a
/// copy of the function from
/// iree/compiler/Codegen/CodegenUtils/FunctionUtils.h that is duplicated
/// here to avoid adding an build dependency.
static size_t getNumOuterParallelLoops(linalg::LinalgOp op) {
  return op.iterator_types()
      .getValue()
      .take_while(
          [](Attribute attr) -> bool { return isParallelIterator(attr); })
      .size();
}

/// Given the `shape` of the computation with the first element being the
/// slowest varying and last element being the fastest warying returns the
/// workload value with
/// - fastest varying dimension first, i.e., x, y, z order
/// - the workload padded to `kNumMaxParallelDims` with ones if needed.
/// The `shape` is expected to be of size less than or equal to
/// `kNumMaxParallelDims`.
static SmallVector<Value, 4> convertToWorkload(OpBuilder &b, Location loc,
                                               ArrayRef<Value> shape) {
  assert(shape.size() <= kNumMaxParallelDims &&
         "workload cannot be more than 3D for now");
  SmallVector<Value, 4> workload = llvm::to_vector<4>(llvm::reverse(shape));
  Value one = b.create<ConstantIndexOp>(loc, 1);
  workload.resize(kNumMaxParallelDims, one);
  return workload;
}

//===----------------------------------------------------------------------===//
// Op property charecterizations
//===----------------------------------------------------------------------===//

/// The current fusion algorithm has some embedded heuristics that are meant to
/// be a first simple start, and can be adapted over time. Note hoever that it
/// is better to have a simple default strategy and use some search-based
/// techniques for actual heuristics. Current heuristics classify operations in
/// this heirarchy
/// - Root Op : These are ops that are computationally intensive and most
///   probably dominate model execution time. These are in general named ops
///   like linalg.matmul, linalg.conv, etc. These are tiled and distributed
///   across workgroups.
/// - Dispatchable ops : These are ops that are not root operations, but still
///   perform some "meaningful" computation. Typically, fused element-wise
///   operations, represented as linalg.generic. These could be fused with root
///   operations using tile + fuse, or could be in their own dispatch regions.
/// - Always fused dispatchable ops : These are ops that are chosen to always be
///   fused into dispatch regions that use their values, since when bufferized
///   they can be converted into being no-copy/aliasing operations. Examples of
///   this is linalg.tensor_reshape that can be converted to a linalg.reshape on
///   bufferization. These are different from dispatchable ops in that they are
///   never in their own dispatch region unless there is no consumer to fuse
///   them with. Typically when the result of the operation is the
///   output.
/// - Always cloned into dispatch op : These are operations that are operations
///   that are always cloned into their consuming dispatch regions and never end
///   up in their own dispatch regions. Typical examples are splat constants and
///   linalg.init_tensor operations.

static bool isRootOp(Operation *op) {
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
    return false;
  }
  return (isa<linalg::LinalgOp>(op) &&
          !isa<linalg::GenericOp, linalg::FillOp>(op)) ||
         isa<linalg_ext::LinalgExtOp>(op);
}

static bool isAlwaysClonedIntoDispatchOp(Operation *op) {
  if (isa<IndexCastOp, linalg::InitTensorOp, tensor::ExtractOp>(op)) {
    return true;
  }
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    return constantOp.getResult().getType().isIntOrIndexOrFloat();
  }
  if (llvm::all_of(op->getOperands(),
                   [&](Value v) { return v.getType().isIntOrFloat(); }) &&
      llvm::all_of(op->getResults(),
                   [&](Value v) { return v.getType().isIntOrFloat(); })) {
    return true;
  }
  return false;
}

static bool isDispatchableOp(Operation *op) {
  // Ignore operations already in dispatch regions.
  if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
    return false;
  }
  // Linalg ops are marked dispatchable.
  if ((op->getDialect() !=
       op->getContext()->getLoadedDialect<linalg::LinalgDialect>()) &&
      !isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op)) {
    return false;
  }

  // Mark linalg.fill as non-dispatchable so that for those linalg.fill ops that
  // cannot be fused together with some root op, they are left out of dispatch
  // region formation, and to be picked up by DMA op conversion.
  if (isa<linalg::FillOp>(op)) return false;

  return !isAlwaysClonedIntoDispatchOp(op);
}

static bool isAlwaysFusedIntoDispatchOp(Operation *op) {
  return isDispatchableOp(op) &&
         (isa<linalg::TensorCollapseShapeOp, tensor::ExtractSliceOp>(op) ||
          isa<linalg::TensorExpandShapeOp, tensor::ExtractSliceOp>(op));
}

//===----------------------------------------------------------------------===//
// Methods that help creating the dispatch regions
//===----------------------------------------------------------------------===//

// Creates a flow.dispatch.workgroup op without arguments.
// All the necessary operands are transiently captured and rewritten late as
// operands. This greatly simplifies transformations into the resulting op.
static std::pair<IREE::Flow::DispatchWorkgroupsOp, Operation *>
buildOperandLessFlowDispatchWorkgroupOp(PatternRewriter &rewriter, Location loc,
                                        ArrayRef<Value> count, Operation *op) {
  auto dispatchOp = rewriter.create<IREE::Flow::DispatchWorkgroupsOp>(
      loc, count, op->getResultTypes(), /*result_dims=*/ValueRange{},
      /*operands=*/ValueRange{},
      /*operand_dims=*/ValueRange{},
      /*tied_operands=*/ArrayRef<int64_t>{});
  Region &region = dispatchOp.body();
  Block *block = &region.front();
  Operation *clonedOp;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(block);
    clonedOp = rewriter.clone(*op);
    for (auto it : llvm::zip(clonedOp->getResults(),
                             dispatchOp.body().getArguments().take_back(
                                 clonedOp->getNumResults()))) {
      rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
          loc, std::get<0>(it), std::get<1>(it), llvm::None, llvm::None,
          llvm::None, rewriter.getArrayAttr({}), rewriter.getArrayAttr({}),
          rewriter.getArrayAttr({}));
    }
    rewriter.create<IREE::Flow::ReturnOp>(loc);
  }
  DEBUG_WITH_TYPE(DEBUG_TYPE, llvm::dbgs() << "Created dispatchOp shell "
                                           << *dispatchOp << "\n");
  return {dispatchOp, clonedOp};
}

// Fuses producers marked in the same group recursively.
//
// The impl does not worry about the dispatchOp, operands and arguments are set
// in a post-pattern `legalizeDispatchWorkgroupOperands` function.
// To simplify the implementation of the dispatch region formation, we just
// clone the op that needs to be fused inside the dispatch region and just fuse
// that one. This avoid any concerns related to tensor operands that are only
// used for their DimOp. This is a canonicalization that is more involved than
// necessary across the boundary of regions without captures.
static void pullInProducersInSameGroup(
    PatternRewriter &rewriter, IREE::Flow::DispatchWorkgroupsOp dispatchOp,
    linalg::LinalgOp tiledOp, ValueRange untiledOpOperands,
    ArrayRef<Operation *> tiledLoops, int64_t groupNum) {
  DEBUG_WITH_TYPE(DEBUG_TYPE, llvm::dbgs() << "pull in producers for tiled op: "
                                           << tiledOp << "\n");

  // Scoped within DispatchWorkgroupOp.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&dispatchOp.getRegion().front());
  for (auto en : llvm::enumerate(untiledOpOperands)) {
    if (auto producer = en.value().getDefiningOp<linalg::LinalgOp>()) {
      if (!isInFusionGroup(producer, groupNum)) continue;
      DEBUG_WITH_TYPE(DEBUG_TYPE,
                      llvm::dbgs() << "current producer: " << producer << "\n");

      Operation *clonedOrigProducer = rewriter.clone(*producer);
      rewriter.replaceOpWithinBlock(producer, clonedOrigProducer->getResults(),
                                    &dispatchOp.getRegion().front());

      linalg::LinalgOp fusedProducer;
      if (tiledLoops.empty()) {
        DEBUG_WITH_TYPE(DEBUG_TYPE, llvm::dbgs()
                                        << "no loops; just copy over the op\n");
        // The root op wasn't tiled. We are done then.
        removeFusionGroupsAttribute(clonedOrigProducer);
        fusedProducer = cast<linalg::LinalgOp>(clonedOrigProducer);
      } else {
        // TODO: this is incorrect on general pattern failures, try pattern
        // within pattern.
        OpResult opResult = en.value().cast<OpResult>();
        auto maybeFusionInfo = linalg::fuseProducerOfTensor(
            rewriter, clonedOrigProducer->getResult(opResult.getResultNumber()),
            tiledOp->getOpOperand(en.index()));
        if (!maybeFusionInfo.hasValue()) {
          DEBUG_WITH_TYPE(DEBUG_TYPE, llvm::dbgs()
                                          << "failed to fuse with tensor\n");
          rewriter.replaceOp(clonedOrigProducer, producer->getResults());
        } else {
          DEBUG_WITH_TYPE(DEBUG_TYPE, llvm::dbgs()
                                          << "succeeded to fuse with tensor\n");
          removeFusionGroupsAttribute(maybeFusionInfo->fusedProducer);
          fusedProducer = maybeFusionInfo->fusedProducer;
        }
      }

      // If the producer is successfully fused, go recursive over the current
      // producer's operands and pull them in if they are marked to be fused
      // into the current group.
      if (fusedProducer) {
        SmallVector<Value> origProducerOpOperands =
            cast<linalg::LinalgOp>(clonedOrigProducer)
                .getInputAndOutputOperands();
        pullInProducersInSameGroup(rewriter, dispatchOp, fusedProducer,
                                   origProducerOpOperands, tiledLoops,
                                   groupNum);
      }
    }
  }
}

template <typename OpTy>
static Value buildFlowWorkgroupInfoOp(OpBuilder &b, unsigned dim) {
  return b.template create<OpTy>(b.getInsertionPoint()->getLoc(), dim);
}

/// Reorders the operations in `ops` such that they could be inlined into the
/// dispatch region in that order to satisfy dependencies.
static SmallVector<Operation *> orderOperations(ArrayRef<Operation *> ops) {
  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "Ops to be inlined :\n";
    for (auto op : ops) {
      llvm::dbgs() << "\t";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });

  llvm::SmallMapVector<Operation *, SmallVector<Operation *>, 16>
      insertAfterMap;
  llvm::SetVector<Operation *> opSet(ops.begin(), ops.end());
  llvm::SetVector<Operation *> leafOps(ops.begin(), ops.end());
  // For each operation compute the list of operations in `ops` that use its
  // results. Also compute the operations that form the leafs of the DAG of
  // operations in `ops`.
  for (auto op : ops) {
    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp || !opSet.count(definingOp)) continue;
      insertAfterMap[definingOp].push_back(op);
      if (leafOps.count(op)) leafOps.remove(op);
    }
  }

  // The leaves are at the head of the ordered list.
  SmallVector<Operation *> orderedOps(leafOps.begin(), leafOps.end());
  orderedOps.reserve(ops.size());
  llvm::SmallPtrSet<Operation *, 16> processed;
  processed.insert(leafOps.begin(), leafOps.end());

  // `readyOps` contains the list of operations that have been just added to the
  // `orderedOps` list. With these marked ready, they might make further
  // operations in `ops` ready as well.
  // The complexity of the algorithm is driven by these
  // - Each operations is added to `readyOps` list at most once, and is removed
  //   after being processed
  // - For every operation in `readyOps` every use of its results (within `ops`)
  //   is looked at once.
  // - For every use, the operands of the user are processed.
  // Assuming operands is O(1), i.e. constant order, the complexity is O(sum of
  // number of uses of each operation). Given that the size of `ops` is at max
  // O(10), and not O(100), this is assumed to be reasonable.
  ArrayRef<Operation *> readyOps(orderedOps);
  size_t startPos = 0;
  while (!readyOps.empty()) {
    auto op = readyOps.front();
    startPos++;
    // Check all uses of `op` within `ops`. If all of the operations that define
    // the operands of the user have been added to `orderedOps`, then the user
    // is ready to be scheduled.
    for (auto insertAfterOp : insertAfterMap[op]) {
      if (processed.count(insertAfterOp)) continue;
      if (llvm::all_of(insertAfterOp->getOperands(), [&](Value operand) {
            Operation *operandDefiningOp = operand.getDefiningOp();
            return !operandDefiningOp || !opSet.count(operandDefiningOp) ||
                   processed.count(operandDefiningOp);
          })) {
        // readyOps.push_back(insertAfterOp);
        orderedOps.push_back(insertAfterOp);
        processed.insert(insertAfterOp);
      }
    }
    readyOps = ArrayRef<Operation *>(orderedOps).drop_front(startPos);
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "Ops to be inlined (sorted) : \n";
    for (auto op : orderedOps) {
      llvm::dbgs() << "\t";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
  });
  assert(orderedOps.size() == ops.size() &&
         "ordering of inlined operations failed");
  return orderedOps;
}

/// Computes the values that will be eventually be used within the dispatch
/// workgroup op but defined outside the op after all clonable operations are
/// cloned into the region. Returns (by reference) the clonable operations too,
/// in order in which they can be cloned within the region to satisfy use-def
/// relationships between them.
static void getUsedValuesDefinedAboveAfterCloningOps(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp,
    llvm::SetVector<Value> &valuesDefinedAbove,
    llvm::SmallVector<Operation *> &clonedOps) {
  llvm::SetVector<Value> visited;
  SmallVector<Value, 4> worklist;
  worklist.assign(valuesDefinedAbove.begin(), valuesDefinedAbove.end());
  valuesDefinedAbove.clear();
  while (!worklist.empty()) {
    Value outsideValue = worklist.pop_back_val();
    if (visited.count(outsideValue)) continue;
    visited.insert(outsideValue);
    Operation *definingOp = outsideValue.getDefiningOp();
    if (!definingOp || !(isAlwaysClonedIntoDispatchOp(definingOp) ||
                         isAlwaysFusedIntoDispatchOp(definingOp))) {
      valuesDefinedAbove.insert(outsideValue);
      continue;
    }
    clonedOps.push_back(definingOp);
    worklist.append(definingOp->operand_begin(), definingOp->operand_end());
  }
  // The cloned operations form a DAG. Return the cloned operations so the
  // leaves come first, and can be cloned in-order into the dispatch region.
  clonedOps = orderOperations(clonedOps);
  // Reverse the values. This is not for correctness, but more for readability
  // of the IR.
  llvm::SetVector<Value> reversedValues;
  reversedValues.insert(valuesDefinedAbove.rbegin(), valuesDefinedAbove.rend());
  std::swap(reversedValues, valuesDefinedAbove);
}

/// Modifies `dispatchOp` to attach operand-result tie information when
/// possible.
static void tryToTieOperandsAndResults(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Block *block = dispatchOp.getBody(0);
  unsigned numResults = dispatchOp.getNumResults();
  auto inputs = block->getArguments().drop_back(numResults);
  auto outputs = block->getArguments().take_back(numResults);

  // Returns the tied operand for the given `resultArg`. Returns nullptr
  // if error or not found.
  auto getTiedOperandBlockArgument =
      [](BlockArgument resultArg) -> BlockArgument {
    // Each output block argument should just have one use.
    if (!llvm::hasSingleElement(resultArg.getUses())) return nullptr;

    // And that's a flow.dispatch.output.store op.
    auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(
        (*resultArg.getUses().begin()).getOwner());
    if (!storeOp) return nullptr;

    Operation *tieOp = storeOp.value().getDefiningOp();
    if (!tieOp) return nullptr;

    // TODO(antiagainst): use TiedOpInterface here instead of hardcoding ops
    // when it's available in MLIR core in some form.
    BlockArgument tiedArg =
        TypeSwitch<Operation *, BlockArgument>(tieOp)
            .Case<tensor::InsertSliceOp>(
                [&](tensor::InsertSliceOp insertOp) -> BlockArgument {
                  auto loadOp = insertOp.dest()
                                    .template getDefiningOp<
                                        IREE::Flow::DispatchTensorLoadOp>();
                  if (!loadOp) return nullptr;
                  return loadOp.source().dyn_cast<BlockArgument>();
                })
            .Case<linalg::LinalgOp, linalg_ext::LinalgExtOp>(
                [&](auto linalgLikeOp) -> BlockArgument {
                  unsigned resultIndex =
                      storeOp.value().cast<OpResult>().getResultNumber();
                  auto loadOp =
                      linalgLikeOp.getOutputTensorOperands()[resultIndex]
                          ->get()
                          .template getDefiningOp<
                              IREE::Flow::DispatchTensorLoadOp>();
                  if (!loadOp) return nullptr;
                  return loadOp.source().template dyn_cast<BlockArgument>();
                })
            .Default([&](Operation *) -> BlockArgument { return nullptr; });

    return tiedArg;
  };

  SmallVector<BlockArgument, 4> tiedOperands;
  tiedOperands.reserve(numResults);

  // Collect all result argument's tied operand arguments.
  for (BlockArgument &arg : outputs) {
    tiedOperands.push_back(getTiedOperandBlockArgument(arg));
  }

  // Go over each result to tie operand when possible, by:
  // 1. Update the tied operand argument to take readwrite tensors.
  // 2. Erase the result argument.
  // 3. Attach the tie information to the DispatchWorkgroupsOp.
  for (int i = outputs.size() - 1; i >= 0; --i) {
    BlockArgument inputArg = tiedOperands[i];
    if (!inputArg) continue;

    auto oldType = inputArg.getType().cast<IREE::Flow::DispatchTensorType>();
    inputArg.setType(IREE::Flow::DispatchTensorType::get(
        IREE::Flow::TensorAccess::ReadWrite, oldType.getShape(),
        oldType.getElementType()));

    BlockArgument outputArg = block->getArgument(inputs.size() + i);
    outputArg.replaceAllUsesWith(inputArg);
    block->eraseArgument(inputs.size() + i);

    dispatchOp.setTiedResultOperandIndex(i, inputArg.getArgNumber());
  }
}

static void replaceAllUsesWithinDispatchOp(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp, Value value,
    Value replacement) {
  SmallPtrSet<Operation *, 4> usesOutsideDispatch;
  for (Operation *user : value.getUsers()) {
    if (isa<IREE::Flow::DispatchWorkgroupsOp>(user) ||
        !dispatchOp->isAncestor(user)) {
      usesOutsideDispatch.insert(user);
    }
  }
  value.replaceAllUsesExcept(replacement, usesOutsideDispatch);
}

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
// A later RematerializeDispatchConstants should be called to avoid passing
// unnecessary constant arguments.
static LogicalResult legalizeDispatchWorkgroupOperands(
    IREE::Flow::DispatchWorkgroupsOp dispatchOp) {
  Location loc = dispatchOp.getLoc();
  Region &region = dispatchOp.body();
  Block &block = region.front();
  unsigned numOldBBArgs = block.getNumArguments();
  OpBuilder b = OpBuilder::atBlockBegin(&block);

  llvm::SetVector<Value> valuesDefinedAbove;
  llvm::SmallVector<Operation *> clonedOps;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return success();

  getUsedValuesDefinedAboveAfterCloningOps(dispatchOp, valuesDefinedAbove,
                                           clonedOps);

  BlockAndValueMapping map;
  SmallVector<Value> toReplaceWithinRegion;
  // Replace valuesDefinedAbove by new BB args (including the op's operands).
  for (Value operand : valuesDefinedAbove) {
    if (auto rt = operand.getType().dyn_cast<RankedTensorType>()) {
      block.addArgument(IREE::Flow::DispatchTensorType::get(
          TensorAccess::ReadOnly, rt.getShape(), rt.getElementType()));
    } else {
      block.addArgument(operand.getType());
    }

    Value bbArg = block.getArguments().back();
    Value repl = bbArg;
    if (bbArg.getType().isa<IREE::Flow::DispatchTensorType>()) {
      repl = b.create<IREE::Flow::DispatchTensorLoadOp>(
          loc, operand.getType().cast<RankedTensorType>(), bbArg);
    }
    map.map(operand, repl);
    toReplaceWithinRegion.push_back(operand);
  }

  // The only existing arguments are for the outputs. Just need to add a new
  // argument for the outputs and remap the value to use the new argument.
  for (unsigned argNum : llvm::seq<unsigned>(0, numOldBBArgs)) {
    BlockArgument arg = block.getArgument(argNum);
    assert(arg.getType().isa<IREE::Flow::DispatchTensorType>());
    arg.replaceAllUsesWith(block.addArgument(arg.getType()));
  }
  // Drop old BB args.
  block.eraseArguments(
      llvm::to_vector<4>(llvm::seq<unsigned>(0, numOldBBArgs)));

  // Clone the marked operations.
  for (Operation *op : clonedOps) {
    b.clone(*op, map);
    toReplaceWithinRegion.append(op->result_begin(), op->result_end());
  }

  // Make the region isolated from above.
  for (auto value : toReplaceWithinRegion) {
    replaceAllUsesWithinDispatchOp(dispatchOp, value, map.lookup(value));
  }

  // Gather the dynamic dimensions for all operands.
  SmallVector<Value, 4> operandDynamicDims;
  OpBuilder builder(dispatchOp);
  for (Value operand : valuesDefinedAbove) {
    if (auto rt = operand.getType().dyn_cast<RankedTensorType>()) {
      for (unsigned i = 0; i < rt.getRank(); ++i) {
        if (!rt.isDynamicDim(i)) continue;
        auto dim = builder.createOrFold<tensor::DimOp>(dispatchOp.getLoc(),
                                                       operand, i);
        operandDynamicDims.push_back(dim);
      }
    }
  }

  // Set the values captured from above as the new operands.
  dispatchOp.operandsMutable().assign(llvm::to_vector<4>(valuesDefinedAbove));
  dispatchOp.operand_dimsMutable().assign(operandDynamicDims);

  return success();
}

/// Returns the loops that are partitioned during dispatch region formations, in
/// order, i.e. starting from the outer-most to innermost.
static SmallVector<unsigned> getPartitionedLoops(Operation *op) {
  SmallVector<unsigned> partitionedLoops;
  if (auto mmt4dOp = dyn_cast<linalg::Mmt4DOp>(op)) {
    return {0, 1};
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    size_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
    partitionedLoops =
        llvm::to_vector<4>(llvm::seq<unsigned>(0, numOuterParallelLoops));
    if (partitionedLoops.size() > kNumMaxParallelDims) {
      partitionedLoops.erase(
          partitionedLoops.begin(),
          std::next(partitionedLoops.begin(),
                    numOuterParallelLoops - kNumMaxParallelDims));
    }
    return partitionedLoops;
  }
  if (auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op)) {
    return tilableOp.getPartitionableLoops(kNumMaxParallelDims);
  }
  return {};
}

/// Computes the shape of the output. This is used to get the workload of the
/// dispatch region if a dispatch region contains a single "Dispatchable op"
static Optional<SmallVector<SmallVector<Value, 4>, 1>> computeOutputShape(
    OpBuilder &builder, Operation *op) {
  SmallVector<SmallVector<Value, 4>, 1> outputShapes;
  for (auto outputType : op->getResultTypes()) {
    // Add empty shape for scalar values.
    if (outputType.isIntOrFloat()) {
      outputShapes.push_back({});
      continue;
    }

    // TODO(ravishankarm): For now only handle static shapes. For dynamic
    // shapes, the shape of the output needs to be resolved using tie shapes,
    // etc.
    if (auto shapedType = outputType.dyn_cast<ShapedType>()) {
      if (!shapedType.hasStaticShape()) return llvm::None;
      outputShapes.push_back(llvm::to_vector<4>(
          llvm::map_range(shapedType.getShape(), [&](int64_t dim) -> Value {
            return builder.create<ConstantIndexOp>(op->getLoc(), dim);
          })));
      continue;
    }
    return llvm::None;
  }
  return outputShapes;
}

static bool hasOnlyDimUses(Operation *op) {
  return llvm::all_of(op->getUsers(), [&](Operation *user) {
    return isa<tensor::DimOp>(user);
  });
}

//===----------------------------------------------------------------------===//
// Patterns that create the dispatch region.
//===----------------------------------------------------------------------===//

namespace {
// Rewrite pattern to ensure only ops with tensor semantics are tiled.
struct TileAndDistributeLinalgOpsPattern
    : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileAndDistributeLinalgOpsPattern(MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter marker,
                                    PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
    if (!hasRootOpAttribute(op)) return failure();

    // TODO(ravishankarm): It is getting strange to track when to apply this
    // pattern and when not to. Need to revisit this, with dynamic shape cases
    // in mind.
    if (hasOnlyDimUses(linalgOp)) return failure();

    // Compute workgroup count to use for the dispatch op. These are the ranges
    // of the outermost parallel loops that can be distributed.
    Location loc = op->getLoc();
    SmallVector<Range> loopRanges = linalgOp.createLoopRanges(rewriter, loc);
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    SmallVector<Value> count;
    for (auto dim : partitionedLoops) {
      count.push_back(loopRanges[dim].size);
    }
    auto workload = convertToWorkload(rewriter, loc, count);

    // Capture dynamic result dimensions.
    SmallVector<Value, 4> resultDynamicDims;
    for (auto result : linalgOp.outputs()) {
      resultDynamicDims.append(Shape::buildOrFindDynamicDimsForValue(
          linalgOp.getLoc(), result, rewriter));
    }

    // Note: DispatchTensorStoreOp generated by the
    // `buildOperandLessFlowDispatchWorkgroupOp` is an abstraction jump that
    // consumes the SSA value produced by `clonedOp` but it does not comply with
    // the semantics of DispatchWorkgroupsOp which explicitly states: "behavior
    // is undefined if multiple workgroups store to the same regions of the
    // output tensors".  Similarly to sequentialized SPMD loops, the semantics
    // is valid assuming a sequential ordering of execution.  After destructive
    // update rewrites, the abstraction gap disappears.
    auto en = buildOperandLessFlowDispatchWorkgroupOp(rewriter, loc, workload,
                                                      linalgOp);
    IREE::Flow::DispatchWorkgroupsOp dispatchOp = en.first;
    linalg::LinalgOp clonedLinalgOp = cast<linalg::LinalgOp>(en.second);
    dispatchOp.result_dimsMutable().assign(resultDynamicDims);

    // Scoped within DispatchWorkgroupOp.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(clonedLinalgOp);

    linalg::TiledLinalgOp tiledLinalgOp;
    LogicalResult tilingResult =
        Base::matchAndRewriteBase(clonedLinalgOp, rewriter, tiledLinalgOp);
    if (failed(tilingResult)) {
      // GreedyPatternRewriter is not transactional and does not stop on
      // failure. Must explicitly delete on all failure paths.
      rewriter.eraseOp(dispatchOp);
      return failure();
    }

    SmallVector<Value> clonedOpOperands =
        clonedLinalgOp.getInputAndOutputOperands();
    pullInProducersInSameGroup(rewriter, dispatchOp, tiledLinalgOp.op,
                               clonedOpOperands, tiledLinalgOp.loops,
                               getRootNumber(op));

    // Keep track of the tiledOpOperands for fusion.
    rewriter.replaceOp(clonedLinalgOp, tiledLinalgOp.tensorResults);

    removeRootOpAttribute(tiledLinalgOp.op);

    rewriter.replaceOpWithIf(op, dispatchOp.getResults(),
                             [&](OpOperand &operand) {
                               return !isa<tensor::DimOp>(operand.getOwner());
                             });
    return success();
  }
};

/// Rewrite pattern to tile and distribute `LinalgExt` ops.
struct TiledOpInterfacePattern
    : public linalg_ext::TiledOpInterfaceBaseTilingPattern {
  using Base = linalg_ext::TiledOpInterfaceBaseTilingPattern;
  using Base::TiledOpInterfaceBaseTilingPattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Check if the op implements the LinalgExt interface and the
    // TiledOpInterface.
    auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op);
    auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op);
    if (!linalgExtOp || !tilableOp) return failure();
    if (!hasRootOpAttribute(op)) return failure();
    if (hasOnlyDimUses(op)) return failure();

    SmallVector<StringRef> iteratorTypes = tilableOp.getLoopIteratorTypes();
    SmallVector<Range> loopRanges = tilableOp.getLoopBounds(rewriter);
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    SmallVector<Value> count;
    for (auto dim : partitionedLoops) {
      count.push_back(loopRanges[dim].size);
    }
    Location loc = op->getLoc();
    auto workload = convertToWorkload(rewriter, loc, count);

    // Capture dynamic result dimensions.
    SmallVector<Value, 4> resultDynamicDims;
    for (auto result : linalgExtOp.outputs()) {
      resultDynamicDims.append(
          Shape::buildOrFindDynamicDimsForValue(loc, result, rewriter));
    }

    // Note: DispatchTensorStoreOp generated by the
    // `buildOperandLessFlowDispatchWorkgroupOp` is an abstraction jump that
    // consumes the SSA value produced by `clonedOp` but it does not comply with
    // the semantics of DispatchWorkgroupsOp which explicitly states: "behavior
    // is undefined if multiple workgroups store to the same regions of the
    // output tensors".  Similarly to sequentialized SPMD loops, the semantics
    // is valid assuming a sequential ordering of execution.  After destructive
    // update rewrites, the abstraction gap disappears.
    auto en =
        buildOperandLessFlowDispatchWorkgroupOp(rewriter, loc, workload, op);
    IREE::Flow::DispatchWorkgroupsOp dispatchOp = en.first;
    auto clonedOp = cast<linalg_ext::LinalgExtOp>(en.second);
    dispatchOp.result_dimsMutable().assign(resultDynamicDims);

    // Scoped within DispatchWorkgroupOp.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(clonedOp);

    linalg_ext::TiledOp tiledOp;
    LogicalResult tilingResult = Base::matchAndRewriteBase(
        cast<linalg_ext::TiledOpInterface>(clonedOp.getOperation()), rewriter,
        tiledOp);
    if (failed(tilingResult)) {
      // GreedyPatternRewriter is not transactional and does not stop on
      // failure. Must explicitly delete on all failure paths.
      rewriter.eraseOp(dispatchOp);
      return failure();
    }
    // Keep track of the tiledOpOperands for fusion.
    SmallVector<Value> tiledOperands = clonedOp.getInputAndOutputOperands();
    if (tiledOp.op != clonedOp) {
      rewriter.replaceOp(clonedOp, tiledOp.results);
    }

    // TODO(ravishankarm): To fuse ops with `linalg_ext` operations (tile+fuse),
    // look into calling `pullInProducersInSameGroup`.
    removeRootOpAttribute(tiledOp.op);

    rewriter.replaceOpWithIf(op, dispatchOp.getResults(),
                             [&](OpOperand &operand) {
                               return !isa<tensor::DimOp>(operand.getOwner());
                             });
    return success();
  }
};

/// Given a list of shapes, returns whether it is statically provable that all
/// shapes are the same. For now checks if
/// 1) Each dimension has the same dynamic value, or,
/// 2) The defining op for each dimension is a `constant` op with the same
///    scalar value.
static bool areAllShapesEqual(ArrayRef<SmallVector<Value>> shapes) {
  assert(!shapes.empty());
  if (shapes.size() == 1) return true;
  auto isSameShape = [&](ArrayRef<Value> lhsShape,
                         ArrayRef<Value> rhsShape) -> bool {
    if (lhsShape.size() != rhsShape.size()) return false;
    return llvm::all_of(
        llvm::zip(lhsShape, rhsShape), [&](std::tuple<Value, Value> vals) {
          APInt lhsInt, rhsInt;
          Value lhs = std::get<0>(vals);
          Value rhs = std::get<1>(vals);
          return lhs == rhs || (matchPattern(lhs, m_ConstantInt(&lhsInt)) &&
                                matchPattern(rhs, m_ConstantInt(&rhsInt)) &&
                                lhsInt == rhsInt);
        });
  };
  return llvm::all_of(
      llvm::make_range(std::next(shapes.begin()), shapes.end()),
      [&](ArrayRef<Value> shape) { return isSameShape(shapes[0], shape); });
}

/// The workload is computed based on the problem size. For a given operation,
/// return the shape of all its results.
static Optional<SmallVector<SmallVector<Value>>> getResultShapes(
    PatternRewriter &rewriter, Operation *op) {
  if (op->getNumResults() == 0) return llvm::None;
  ReifiedRankedShapedTypeDims resultShapes;
  // Check if the op implements the shape interface.
  if (auto shapedOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op)) {
    if (failed(shapedOp.reifyResultShapes(rewriter, resultShapes))) {
      return llvm::None;
    }
    return resultShapes;
  }

  // Fallback is to get the shape using `dim` of the outputs. Since the
  // workload depends on the output shape, set the insertion point to after
  // the operation. After dim canonicalization, the original operation should
  // become dead.
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  auto getShapeOfShapedTypeVal = [&](Value v) -> SmallVector<Value> {
    SmallVector<Value> shape;
    for (auto dim :
         llvm::seq<int64_t>(0, v.getType().cast<ShapedType>().getRank())) {
      shape.push_back(rewriter.createOrFold<tensor::DimOp>(loc, v, dim));
    }
    return shape;
  };
  for (OpResult result : op->getResults()) {
    auto resultType = result.getType().dyn_cast<ShapedType>();
    if (!resultType) return llvm::None;
    rewriter.setInsertionPointAfter(op);
    auto resultShape = getShapeOfShapedTypeVal(result);
    resultShapes.emplace_back(std::move(resultShape));
  }
  return resultShapes;
}

/// Puts ops that are not-tilable or arent tiled into a
/// `flow.dispatch.workgroups` operation. For example tile and distribute of
/// element-wise operations is not beneficial. These are handled appropriately
/// by the backends.
struct MakeDispatchWorkgroupsOp : public RewritePattern {
  MakeDispatchWorkgroupsOp(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isDispatchableOp(op) || hasOnlyDimUses(op)) return failure();

    // If this is a dispatchable op that is to be fused into dispatch ops, and
    // all its uses are dispatchable ops, don't do anything.
    if ((hasFusionGroupsAttribute(op) || isAlwaysFusedIntoDispatchOp(op)) &&
        llvm::all_of(op->getUsers(), [](Operation *user) {
          return isDispatchableOp(user) ||
                 user->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() ||
                 isa<IREE::Flow::DispatchWorkgroupsOp, tensor::DimOp>(user);
        })) {
      return failure();
    }

    // The workgroup count is based on the result shape.
    Optional<SmallVector<SmallVector<Value>>> resultShapesOpt =
        getResultShapes(rewriter, op);
    if (!resultShapesOpt) return failure();
    ArrayRef<SmallVector<Value>> resultShapes = *resultShapesOpt;
    if (resultShapes.size() != op->getNumResults() ||
        !areAllShapesEqual(resultShapes))
      return failure();

    // TODO(ravishankarm): For now the Flow -> HAL conversion only handles
    // workload count of 3, though it should be generalized. For now making sure
    // the flow has three elements of workload size (x, y, z) by linearizing the
    // workloads for all higher dimensions greater than or equal to
    // kNumMaxParallelDims.
    Location loc = op->getLoc();
    SmallVector<Value, 4> count(resultShapes[0].begin(), resultShapes[0].end());
    if (count.size() > kNumMaxParallelDims) {
      unsigned numSymbols = 0;
      AffineExpr expr = rewriter.getAffineSymbolExpr(numSymbols++);
      for (int64_t i = 1; i < count.size() - kNumMaxParallelDims + 1; i++) {
        expr = expr * rewriter.getAffineSymbolExpr(numSymbols++);
      }
      count[count.size() - kNumMaxParallelDims] = linalg::applyMapToValues(
          rewriter, loc, AffineMap::get(0, numSymbols, expr),
          ArrayRef<Value>(count).take_front(count.size() - kNumMaxParallelDims +
                                            1))[0];
      count = llvm::to_vector<4>(
          ArrayRef<Value>(count).take_back(kNumMaxParallelDims));
    }
    auto workload = convertToWorkload(rewriter, loc, count);

    // Capture dynamic result dimensions.
    SmallVector<Value, 4> resultDynamicDims;
    for (auto result : llvm::enumerate(op->getResults())) {
      auto resultType = result.value().getType().cast<ShapedType>();
      for (unsigned i = 0; i < resultType.getRank(); ++i) {
        if (resultType.isDynamicDim(i)) {
          resultDynamicDims.push_back(resultShapes[result.index()][i]);
        }
      }
    }

    auto en = buildOperandLessFlowDispatchWorkgroupOp(rewriter, op->getLoc(),
                                                      workload, op);
    IREE::Flow::DispatchWorkgroupsOp dispatchOp = en.first;
    dispatchOp.result_dimsMutable().assign(resultDynamicDims);

    // If this is a root op for fusion, try to pull in the ops to be fused
    // together with it.
    if (hasRootOpAttribute(op)) {
      auto clonedLinalgOp = dyn_cast<linalg::LinalgOp>(en.second);
      if (clonedLinalgOp) {
        SmallVector<Value> opOperandsVal =
            clonedLinalgOp.getInputAndOutputOperands();
        pullInProducersInSameGroup(
            rewriter, dispatchOp, clonedLinalgOp, opOperandsVal,
            /*tiledLoops=*/ArrayRef<Operation *>(), getRootNumber(op));
        removeRootOpAttribute(clonedLinalgOp);
      }
    }

    rewriter.replaceOpWithIf(op, dispatchOp.getOperation()->getResults(),
                             [&](OpOperand &operand) {
                               Operation *user = operand.getOwner();
                               return !isa<tensor::DimOp>(user);
                             });
    return success();
  }
};
};  // namespace

//===----------------------------------------------------------------------===//
// Heuristics for fusing dispatchble ops with root ops using tile + fuse.
//===----------------------------------------------------------------------===//

/// Some heuristic is needed to fuse a dispatchble op with root operations using
/// tile + fuse. Using some heuristic, each root operation is tagged with an ID
/// (using an IntegerAttr with name `kRootOpAttr`) and all dispatchable ops to
/// be fused with it is tagged with the same ID (using a list of IntegerAttr
/// with name `kFusionGroupsAttr`). Each dispatchable operation can be marked to
/// fuse with multiple root operations (i.e. replicated). For now a very simple
/// heuristic is used below, but the mechanism should be general enough to
/// capture any heuristic.

/// Sets elementwise operations as root operations.
// TODO(#5045): After the regression issue on CPU side is addressed, this can be
// folded into the main logic of fusion.
template <typename GenericOpTy>
static unsigned makeElementwiseOpsRootOps(mlir::FuncOp funcOp,
                                          unsigned numRoots) {
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  for (Block &block : funcOp) {
    auto linalgOps = block.getOps<linalg::LinalgOp>();
    for (linalg::LinalgOp linalgOp : llvm::reverse(linalgOps)) {
      Operation *op = linalgOp.getOperation();
      if (hasRootOpAttribute(op) || hasFusionGroupsAttribute(op)) {
        continue;
      }
      if (!isa<GenericOpTy>(op) ||
          !llvm::all_of(
              cast<linalg::LinalgOp>(op).getIndexingMaps(),
              [](AffineMap map) { return map.isProjectedPermutation(); })) {
        continue;
      }
      unsigned newGroup = numRoots++;
      setRootAttribute(context, op, newGroup);

      for (OpOperand *operand : linalgOp.getOutputTensorOperands()) {
        auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
        if (!producer) continue;
        if (producer.getNumLoops() != producer.getNumParallelLoops()) continue;
        appendToFusionGroup(producer, newGroup);
      }
    }
  }
  return numRoots;
}

/// For a given block partition the LinalgOps in the block into fusable
/// groups. All analysis of what to fuse happens here. For now this is just
/// hard-wiring from basic heuristic but this could be adapted to have 1) better
/// heuristics and 2) use a search approach to decide what all should be fused.
static unsigned decideFusableLinalgOps(mlir::FuncOp funcOp) {
  unsigned numRootOps = 0;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  for (Block &block : funcOp) {
    // Tiling and fusion works by tiling the last operation in the fusion group
    // and then pull producer ops into the tiled loops. So go in the reverse
    // order here.
    for (Operation &op : llvm::reverse(block)) {
      // Start with a root operation and fuse its producers.
      if (!isRootOp(&op)) continue;
      unsigned newGroup = numRootOps++;
      setRootAttribute(context, &op, newGroup);

      linalg::OpOperandVector outOperands =
          TypeSwitch<Operation *, linalg::OpOperandVector>(&op)
              .Case<linalg::LinalgOp>([&](auto linalgOp) {
                return linalgOp.getOutputTensorOperands();
              })
              .Default(
                  [&](Operation *) -> linalg::OpOperandVector { return {}; });
      for (OpOperand *operand : outOperands) {
        auto producer = operand->get().getDefiningOp<linalg::LinalgOp>();
        if (!producer) continue;
        if (producer.getNumLoops() != producer.getNumParallelLoops()) continue;
        appendToFusionGroup(producer, newGroup);
      }
    }

    if (clEnableOperandFusion) {
      // To fuse root operations with their consumers, for all root ops chosen.
      // If, 1) The root op has a single use 2) The consumer is an elementwise
      // operation 3) The indexing map in the producer and consumer are identity
      // maps The root operation can be fused with its consumer. To do this,
      // mark the consumer as the root and add the operation to the fusion
      // group.
      for (linalg::LinalgOp linalgOp : block.getOps<linalg::LinalgOp>()) {
        Operation *op = linalgOp.getOperation();
        if (!hasRootOpAttribute(op)) continue;
        if (op->getNumResults() != 1 || !op->hasOneUse()) continue;
        OpOperand &use = *op->use_begin();
        Operation *user = use.getOwner();
        if (hasRootOpAttribute(user) || hasFusionGroupsAttribute(user)) {
          continue;
        }
        linalg::LinalgOp consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (!consumer ||
            consumer.getNumLoops() != consumer.getNumParallelLoops()) {
          continue;
        }
        AffineMap consumerIndexingMap = consumer.getTiedIndexingMap(&use);
        AffineMap producerIndexingMap =
            linalgOp.getTiedIndexingMap(linalgOp.getOutputOperand(0));
        if (!consumerIndexingMap.isIdentity() ||
            producerIndexingMap.getResults() !=
                consumerIndexingMap.getResults()) {
          continue;
        }
        int64_t rootNumber = getRootNumber(op);
        setRootAttribute(context, user, rootNumber);
        removeRootOpAttribute(op);
        appendToFusionGroup(op, rootNumber);
      }
    }
  }
  return numRootOps;
}

namespace {
/// Pass declaration.
struct DispatchLinalgOnTensorsPass
    : public DispatchLinalgOnTensorsBase<DispatchLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, ShapeDialect, tensor::TensorDialect>();
  }
  DispatchLinalgOnTensorsPass() = default;
  DispatchLinalgOnTensorsPass(const DispatchLinalgOnTensorsPass &pass) {}
  void runOnOperation() override;

 private:
  Statistic numDispatches{this, "number of dispatches",
                          "Number of Flow dispatches created"};
};
}  // namespace

void DispatchLinalgOnTensorsPass::runOnOperation() {
  auto funcOp = getOperation();

  MLIRContext *context = funcOp->getContext();
  context->allowUnregisteredDialects(true);

  unsigned numRoots = decideFusableLinalgOps(funcOp);
  numRoots = makeElementwiseOpsRootOps<linalg::GenericOp>(funcOp, numRoots);

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- After annotating linalg op fusion scheme ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Distribution strategy along at most 3 dimensions with WorkgroupIdOp in
  // range [0, WorkgroupSizeOp).
  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        auto numParallelDims = parallelLoopRanges.size();

        SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
        for (size_t dim = 0; dim < numParallelDims; ++dim) {
          procInfo[numParallelDims - dim - 1] = {
              buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupIDOp>(builder,
                                                                    dim),
              buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupCountOp>(builder,
                                                                       dim)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic},
      DenseMap<StringRef,
               std::function<linalg::ProcInfo(OpBuilder &, Location)>>()};

  auto tileSizeFn = [&](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value, 4> {
    SmallVector<unsigned> partitionedLoops = getPartitionedLoops(op);
    if (partitionedLoops.empty()) return {};
    unsigned maxDepth = partitionedLoops.back() + 1;

    if (!clLinalgOnTensorsTileSizes.empty()) {
      SmallVector<int64_t, 2> tileSizes(clLinalgOnTensorsTileSizes.begin(),
                                        clLinalgOnTensorsTileSizes.end());
      return llvm::to_vector<4>(llvm::map_range(
          ArrayRef<int64_t>(tileSizes).take_front(
              std::min<size_t>(tileSizes.size(), maxDepth)),
          [&](int64_t t) -> Value {
            return builder.create<ConstantIndexOp>(op->getLoc(), t);
          }));
    }

    // Set all loops not partitioned to tile size 0. and those partitioned to
    // `flow.workgroup.size`.
    auto zero = builder.create<ConstantIndexOp>(op->getLoc(), 0);
    SmallVector<Value, 4> useTileSizes(maxDepth, zero);
    llvm::DenseSet<unsigned> partitionedLoopsSet;
    partitionedLoopsSet.insert(partitionedLoops.begin(),
                               partitionedLoops.end());
    unsigned currFlowDim = 0;
    for (size_t dim = maxDepth; dim > 0; dim--) {
      if (partitionedLoopsSet.count(dim - 1)) {
        useTileSizes[dim - 1] =
            buildFlowWorkgroupInfoOp<Flow::DispatchWorkgroupSizeOp>(
                builder, currFlowDim++);
      }
    }
    return useTileSizes;
  };

  {
    // Use the workgroup size as a proxy for tile size here. At the flow level
    // this represents the "workload" per processors and is not necessarily tied
    // to the workgroup size specified by the backend.
    OwningRewritePatternList patterns(&getContext());
    auto linalgTilingOptions =
        linalg::LinalgTilingOptions()
            .setDistributionOptions(workgroupDistributionOptions)
            .setLoopType(linalg::LinalgTilingLoopType::Loops)
            .setTileSizeComputationFunction(tileSizeFn);
    assert(linalgTilingOptions.distribution.hasValue());

    patterns.insert<TileAndDistributeLinalgOpsPattern, TiledOpInterfacePattern>(
        context, linalgTilingOptions,
        // TODO(nicolavasilache): use refactored `getWorkgroupMarker()`
        linalg::LinalgTransformationFilter(
            ArrayRef<Identifier>(), Identifier::get("workgroup", context)));
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Run canonicalization patterns.
    OwningRewritePatternList canonicalizationPattterns(&getContext());
    linalg::populateLinalgTilingCanonicalizationPatterns(
        canonicalizationPattterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPattterns));
  }

  // If elementwise operations are not tiled and distributed, the wont be marked
  // as root ops previously. Mark them so here to allow fusion of `fill` etc.
  numRoots = makeElementwiseOpsRootOps<linalg::GenericOp>(funcOp, numRoots);

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs()
        << "\n--- After annotating linalg op fusion scheme for fallback ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures.
  if (funcOp
          .walk([&](IREE::Flow::DispatchWorkgroupsOp op) -> WalkResult {
            return legalizeDispatchWorkgroupOperands(op);
          })
          .wasInterrupted()) {
    return signalPassFailure();
  }

  // Move other operations into their own dispatch regions.
  {
    OwningRewritePatternList patterns(context);
    patterns.insert<MakeDispatchWorkgroupsOp>(context);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- After dispatch region creation ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures.
  if (funcOp
          .walk([&](IREE::Flow::DispatchWorkgroupsOp op) -> WalkResult {
            numDispatches++;
            return legalizeDispatchWorkgroupOperands(op);
          })
          .wasInterrupted()) {
    return signalPassFailure();
  }

  // Run necessary canonicalization patterns before destructive updates.
  {
    OwningRewritePatternList patterns(&getContext());
    // This is needed because tiling and distribution may create
    // subtensor_insert ops whose source operands come from tensor.cast ops.
    // Those tensor.cast ops cast tensors into a more dynamic shape, in order
    // to guarantee type match during transformation. Later in destructive
    // update subtensor_insert ops will be turned into flow dispatch output
    // store ops.
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- After dispatch op legalization ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Rewrite destructive updates and ensure no remaining store remains to the
  // full output.
  if (funcOp
          .walk([&](IREE::Flow::DispatchWorkgroupsOp op) {
            if (failed(rewriteLinalgDestructiveUpdates(op))) {
              funcOp.emitError("Failed to rewrite destructive updates in:\n")
                  << *op.getOperation();
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    signalPassFailure();
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "\n--- After rewriting destructive updates ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Now try to see if we can tie certain results to operands in order to
  // indicate sharing storage. This need to happen here because it needs to
  // access region block arguments for input/output tensors, which aren't
  // available until now.
  funcOp.walk([&](IREE::Flow::DispatchWorkgroupsOp op) {
    tryToTieOperandsAndResults(op);
  });
}

std::unique_ptr<OperationPass<mlir::FuncOp>>
createDispatchLinalgOnTensorsPass() {
  return std::make_unique<DispatchLinalgOnTensorsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
