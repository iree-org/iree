// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-propagate-timepoints"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_PROPAGATETIMEPOINTSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// TODO(benvanik): factor out into a generic util pass base that lets us share
// with other expanded type propagation passes. The walking of
// functions/blocks/globals/etc are the same across all of them and only the
// exact type expansion and consumption/query ops differ.

//===----------------------------------------------------------------------===//
// Global handling
//===----------------------------------------------------------------------===//

struct ExpandedGlobal {
  IREE::Util::GlobalOp resourceOp;
  IREE::Util::GlobalOp timepointOp;
};
using ExpandedGlobalMap = DenseMap<StringRef, ExpandedGlobal>;

// Expands each !stream.resource global in |rootOp| to have a matching
// timepoint. Does not behave optimally if there already exist timepoint globals
// as duplicates will get added and we'll need to rely on global fusion to
// get rid of them. Note that this only expands globals and does not yet update
// use sites - we just need the ops to reference while doing so.
static ExpandedGlobalMap expandResourceGlobals(Operation *rootOp) {
  ExpandedGlobalMap expandedGlobals;

  // Gather all of the resource globals in the root.
  for (auto &region : rootOp->getRegions()) {
    for (auto globalOp : region.getOps<IREE::Util::GlobalOp>()) {
      if (!llvm::isa<IREE::Stream::ResourceType>(globalOp.getType()))
        continue;
      expandedGlobals[globalOp.getName()].resourceOp = globalOp;
    }
  }

  // Expand each global by adding the timepoint right next to it.
  SymbolTable symbolTable(rootOp);
  auto timepointType = IREE::Stream::TimepointType::get(rootOp->getContext());
  auto immediateAttr =
      IREE::Stream::TimepointAttr::get(rootOp->getContext(), timepointType);
  for (auto &it : expandedGlobals) {
    auto resourceOp = it.second.resourceOp;
    OpBuilder builder(resourceOp);
    auto timepointName = (resourceOp.getName() + "__timepoint").str();
    auto timepointOp = builder.create<IREE::Util::GlobalOp>(
        resourceOp.getLoc(), timepointName,
        /*isMutable=*/true, timepointType, immediateAttr);
    timepointOp.setVisibility(resourceOp.getVisibility());
    symbolTable.insert(timepointOp);
    it.second.timepointOp = timepointOp;
  }

  return expandedGlobals;
}

//===----------------------------------------------------------------------===//
// Structural IR rewriting patterns
//===----------------------------------------------------------------------===//

static bool isResourceType(Type type) {
  return llvm::isa<IREE::Stream::ResourceType>(type);
}

// Returns true if an operands or results of |op| use !stream.resources.
static bool usesResources(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isResourceType) ||
         llvm::any_of(op->getResultTypes(), isResourceType);
}

// Expands resources in the given |types| list to (timepoint, resource).
// This could be changed to some iterator magic to avoid the alloc.
static SmallVector<Type> expandTypes(TypeRange types) {
  if (types.empty())
    return {};
  auto timepointType =
      IREE::Stream::TimepointType::get(types.front().getContext());
  SmallVector<Type> newTypes;
  newTypes.reserve(types.size() * 2);
  for (auto type : types) {
    if (isResourceType(type)) {
      newTypes.push_back(timepointType);
    }
    newTypes.push_back(type);
  }
  return newTypes;
}

// Attempts to find and consume the timepoint associated with |value|.
// Returns a (timepoint, resource) pair where the timepoint indicates when the
// resource - which may differ from the provided |value| - is ready. In cases
// where no associated timepoint was found the timepoint will be immediate.
static std::pair<Value, Value> consumeTimepoint(Location loc, Value value,
                                                IRMapping &resourceTimepointMap,
                                                OpBuilder &builder) {
  // TODO(benvanik): follow ties on value to try to consume there; there are a
  // few other ops we could look through as well (such as select, where we could
  // join). For now we just look at immediate defining ops.
  auto timepoint = resourceTimepointMap.lookupOrNull(value);
  if (timepoint) {
    return std::make_pair(timepoint, value);
  }

  if (auto awaitOp = dyn_cast_or_null<IREE::Stream::TimepointAwaitOp>(
          value.getDefiningOp())) {
    return std::make_pair(awaitOp.getAwaitTimepoint(),
                          awaitOp.getTiedResultOperand(value));
  } else if (auto executeOp = dyn_cast_or_null<IREE::Stream::AsyncExecuteOp>(
                 value.getDefiningOp())) {
    return std::make_pair(executeOp.getResultTimepoint(), value);
  } else {
    return std::make_pair(
        builder.create<IREE::Stream::TimepointImmediateOp>(loc).getResult(),
        value);
  }
}

// Expands resources in |operands| into (timepoint, resource) pairs.
static SmallVector<Value> expandOperands(Location loc, ValueRange operands,
                                         IRMapping &resourceTimepointMap,
                                         OpBuilder &builder) {
  SmallVector<Value> result;
  result.reserve(operands.size() * 2);
  for (auto operand : operands) {
    if (isResourceType(operand.getType())) {
      auto timepointOperand =
          consumeTimepoint(loc, operand, resourceTimepointMap, builder);
      result.push_back(timepointOperand.first);
      result.push_back(timepointOperand.second);
    } else {
      result.push_back(operand);
    }
  }
  return result;
}

static void expandTimepoints(Operation *op, ExpandedGlobalMap &globalMap,
                             IRMapping &resourceTimepointMap);

// Finds the size of a block argument resource or materializes a size if needed.
// The returned SSA value will be valid at the insertion point (by way of clones
// or other trickery required to make it so).
static Value makeBlockArgResourceSize(Location loc, Value resourceValue,
                                      OpBuilder &builder) {
  // We can take any implicitly captured SSA values.
  if (auto sizeAwareOp = dyn_cast_or_null<IREE::Util::SizeAwareOpInterface>(
          resourceValue.getDefiningOp())) {
    auto sizeValue = sizeAwareOp.getResultSizeFromValue(resourceValue);
    if (sizeValue)
      return sizeValue;
  }

  // Try first to scan uses in the IR. Since we carry the shape in most ops we
  // are likely to find at least some SSA value we can inspect.
  for (auto &use : resourceValue.getUses()) {
    auto sizeAwareOp =
        dyn_cast<IREE::Util::SizeAwareOpInterface>(use.getOwner());
    if (!sizeAwareOp)
      continue;
    auto sizeValue = sizeAwareOp.getOperandSize(use.getOperandNumber());
    if (!sizeValue)
      continue;
    if (sizeValue.getParentRegion()->isProperAncestor(
            builder.getInsertionBlock()->getParent())) {
      // Size value found and implicitly captured; we can reuse (could be
      // a parent block argument, a constant, computed, etc).
      return sizeValue;
    } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(sizeValue)) {
      if (blockArg.getParentBlock()->isEntryBlock()) {
        // Dynamic dimension passed in to the entry block; safe to use.
        return sizeValue;
      }
    } else if (sizeValue.getDefiningOp() &&
               sizeValue.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      // Constant op - duplicate at the builder location so we don't have to
      // worry about SSA dominance issues. CSE will clean up the dupes later.
      return builder.clone(*sizeValue.getDefiningOp())->getResult(0);
    }
    // Uninspectable value.
  }

  // If we couldn't find anything we could use we'll insert the size query. The
  // hope is that more program analysis could take care of this for us.
  return builder.create<IREE::Stream::ResourceSizeOp>(loc, resourceValue);
}

// Recursively expands resources into (timepoint, resource) pairs within the
// given |region|. All branches, ops, and nested regions will be processed.
static void expandRegion(Region &region, bool canModifyEntryBlock,
                         ExpandedGlobalMap &globalMap,
                         IRMapping resourceTimepointMap) {
  if (region.empty())
    return;

  // Update all block arguments.
  auto timepointType = IREE::Stream::TimepointType::get(region.getContext());
  for (auto &block : region.getBlocks()) {
    if (!llvm::any_of(block.getArgumentTypes(), isResourceType))
      continue;
    if (block.isEntryBlock() && !canModifyEntryBlock)
      continue;

    // Insert and build a list of expanded (timepoint, resource) pairs.
    SmallVector<std::pair<Value, Value>> expansions;
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      auto resourceArg = block.getArgument(i);
      if (!isResourceType(resourceArg.getType()))
        continue;
      auto timepointArg =
          block.insertArgument(i, timepointType, resourceArg.getLoc());
      expansions.push_back(std::make_pair(timepointArg, resourceArg));
      resourceTimepointMap.map(resourceArg, timepointArg);
    }

    // Insert awaits that we've sunk from callers.
    auto builder = OpBuilder::atBlockBegin(&block);
    for (auto expansion : llvm::reverse(expansions)) {
      // If we can look down the chain and see the size then we can use that.
      // If it's a constant we can't use it as it may be defined anywhere in the
      // region. Dynamic dimensions usually come from outside or entry arguments
      // though and those are available.
      auto timepoint = expansion.first;
      auto resource = expansion.second;
      auto resourceSize =
          makeBlockArgResourceSize(region.getLoc(), resource, builder);
      auto awaitOp = builder.create<IREE::Stream::TimepointAwaitOp>(
          region.getLoc(), resource, resourceSize, timepoint);
      SmallPtrSet<Operation *, 2> excludedUsers;
      excludedUsers.insert(awaitOp);
      if (auto *sizeOp = resourceSize.getDefiningOp()) {
        excludedUsers.insert(sizeOp);
      }
      resource.replaceAllUsesExcept(awaitOp.getResults().front(),
                                    excludedUsers);
    }
  }

  // Walk blocks forward in domination order so that we add dominating values to
  // the timepoint map. Note that DominanceInfo is just determined not to be
  // cool about things when there's only one block so we have to special case.
  if (region.hasOneBlock()) {
    for (auto &op :
         llvm::make_early_inc_range(region.front().getOperations())) {
      expandTimepoints(&op, globalMap, resourceTimepointMap);
    }
  } else {
    DominanceInfo domInfo(region.getParentOp());
    for (auto *blockInfo : llvm::breadth_first(domInfo.getRootNode(&region))) {
      auto *block = blockInfo->getBlock();
      for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
        expandTimepoints(&op, globalMap, resourceTimepointMap);
      }
    }
  }
}

// Moves awaits from global stores to loads.
// Requires that the ExpandGlobalStoreOp pattern elides the await.
//
// Example:
//  %0 = util.global.load @foo : !stream.resource
//  ->
//  %t = util.global.load @foo : !stream.timepoint
//  %0 = util.global.load @foo : !stream.resource
//  %1 = stream.timepoint.await %t, %0
static void expandGlobalLoadOp(IREE::Util::GlobalLoadOp op,
                               ExpandedGlobalMap &globalMap,
                               IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  auto &expandedGlobal = globalMap[op.getGlobal()];
  auto timepoint =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), IREE::Stream::TimepointType::get(op.getContext()),
              expandedGlobal.timepointOp.getName())
          .getResult();
  resourceTimepointMap.map(op.getResult(), timepoint);

  // HACK: queryValueSize may insert other ops that we don't want to replace.
  // TODO(benvanik): carry the size so we don't need to guess here.
  SmallPtrSet<Operation *, 2> replacementExceptions;
  builder.setInsertionPointAfter(op);
  auto resultSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
      op.getLoc(), op.getResult(), builder);
  if (resultSize) {
    replacementExceptions.insert(resultSize.getDefiningOp());
  } else {
    auto sizeOp = builder.create<IREE::Stream::ResourceSizeOp>(op.getLoc(),
                                                               op.getResult());
    replacementExceptions.insert(sizeOp);
    resultSize = sizeOp.getResult();
  }
  assert(resultSize && "need to be able to get a size");

  auto awaitOp = builder.create<IREE::Stream::TimepointAwaitOp>(
      op.getLoc(), op.getResult(), resultSize, timepoint);
  replacementExceptions.insert(awaitOp);

  op.getResult().replaceAllUsesExcept(awaitOp.getResults().front(),
                                      replacementExceptions);
}

// Moves awaits from global stores to loads.
// Requires that the ExpandGlobalLoadOp pattern inserts the await.
//
// Example:
//  %1 = stream.timepoint.await %t, %0
//  util.global.store %1, @foo : !stream.resource
//  ->
//  util.global.store %t, @foo_timepoint : !stream.timepoint
//  util.global.store %0, @foo : !stream.resource
static void expandGlobalStoreOp(IREE::Util::GlobalStoreOp op,
                                ExpandedGlobalMap &globalMap,
                                IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  auto timepointOperand = consumeTimepoint(op.getLoc(), op.getValue(),
                                           resourceTimepointMap, builder);
  auto &expandedGlobal = globalMap[op.getGlobal()];
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), timepointOperand.first,
      expandedGlobal.timepointOp.getName());
  op.getValueMutable().assign(timepointOperand.second);
}

static void expandInitializerOp(IREE::Util::InitializerOp op,
                                ExpandedGlobalMap &globalMap,
                                IRMapping &resourceTimepointMap) {
  expandRegion(op.getRegion(), /*canModifyEntryBlock=*/false, globalMap,
               resourceTimepointMap);
}

// Returns true if |op| is either public and visible to external modules or
// external and resolved later on. We can't modify their signatures.
static bool isPublicOrExternal(CallableOpInterface callableOp) {
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(callableOp.getOperation())) {
    if (symbolOp.isPublic())
      return true;
  }
  auto *region = callableOp.getCallableRegion();
  if (!region || region->empty())
    return true;
  return false;
}

// Inserts awaits on resource arguments.
// Requires that the ExpandCallOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant waits in cases where the call sites
// don't need a wait.
//
// Example:
//  func.func @foo(%0: !stream.resource)
//  ->
//  func.func @foo(%t: !stream.timepoint, %0: !stream.resource) {
//    %1 = stream.timepoint.await %t, %0
static void expandFuncOp(mlir::func::FuncOp op, ExpandedGlobalMap &globalMap,
                         IRMapping &resourceTimepointMap) {
  // Ignore public/external function signatures but still convert regions.
  bool canModifyEntryBlock = !isPublicOrExternal(op);
  if (canModifyEntryBlock) {
    auto oldType = op.getFunctionType();
    auto inputTypes = expandTypes(oldType.getInputs());
    auto resultTypes = expandTypes(oldType.getResults());
    auto newType = FunctionType::get(op.getContext(), inputTypes, resultTypes);
    if (newType != oldType) {
      op.setType(newType);
    }
  }
  expandRegion(op.getRegion(), canModifyEntryBlock, globalMap,
               resourceTimepointMap);
}

// Splits resource operands and results into (timepoint, resource).
// Requires that the ExpandFuncOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant waits in cases where the call sites
// don't need a wait.
//
// Example:
//  %1 = stream.timepoint.await %t, %0
//  %r = call @foo(%1)
//  ->
//  %rt, %r = call @foo(%t, %0)
//  stream.timepoint.await %rt, %t
static void expandCallOp(mlir::func::CallOp op,
                         IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;

  // Ignore calls to public/external functions.
  auto calleeOp = SymbolTable::lookupNearestSymbolFrom<CallableOpInterface>(
      op, op.getCalleeAttr());
  if (isPublicOrExternal(calleeOp))
    return;

  // Build the new call op with expanded operands and results.
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(),
                                 resourceTimepointMap, builder);
  auto resultTypes = expandTypes(op.getResultTypes());
  auto newOp = builder.create<mlir::func::CallOp>(op.getLoc(), op.getCallee(),
                                                  resultTypes, operands);

  // Insert awaits on results that we are sinking across the call edge.
  // The hope is that by moving the awaits here we can fold with uses inside
  // of this function.
  builder.setInsertionPointAfter(newOp);
  unsigned newIdx = 0;
  for (unsigned oldIdx = 0; oldIdx < op.getNumResults(); ++oldIdx) {
    auto oldResult = op.getResult(oldIdx);
    if (!isResourceType(oldResult.getType())) {
      auto newResult = newOp.getResult(newIdx++);
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    auto newTimepoint = newOp.getResult(newIdx++);
    auto newResult = newOp.getResult(newIdx++);
    resourceTimepointMap.map(newResult, newTimepoint);
    auto newResultSize =
        builder.create<IREE::Stream::ResourceSizeOp>(op.getLoc(), newResult)
            .getResult();
    auto awaitOp = builder.create<IREE::Stream::TimepointAwaitOp>(
        op.getLoc(), newResult, newResultSize, newTimepoint);
    oldResult.replaceAllUsesWith(awaitOp.getResults().front());
  }

  op.erase();
}

// Moves awaits to callers upon return.
// Requires that the ExpandFuncOp/ExpandCallOp patterns handle migrating the
// await.
//
// Example:
//  %1 = stream.timepoint.await %t, %0
//  return %1
//  ->
//  return %t, %0
static void expandReturnOp(mlir::func::ReturnOp op,
                           IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  if (isPublicOrExternal(op->getParentOfType<mlir::func::FuncOp>()))
    return;
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(),
                                 resourceTimepointMap, builder);
  builder.create<mlir::func::ReturnOp>(op.getLoc(), operands);
  op.erase();
}

// Moves awaits across branches.
// Requires that the ExpandFuncOp pattern handles modifying the block args.
//
// Example:
//    %1 = stream.timepoint.await %t, %0
//    br ^bb1(%1)
//  ^bb1(%b):
//  ->
//    br ^bb1(%t, %0)
//  ^bb1(%a, %b):
//    %1 = stream.timepoint.await %a, %b
static void expandBranchOp(mlir::cf::BranchOp op,
                           IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getDestOperands(),
                                 resourceTimepointMap, builder);
  builder.create<mlir::cf::BranchOp>(op.getLoc(), op.getDest(), operands);
  op.erase();
}

static void expandCondBranchOp(mlir::cf::CondBranchOp op,
                               IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  builder.create<mlir::cf::CondBranchOp>(
      op.getLoc(), op.getCondition(), op.getTrueDest(),
      expandOperands(op.getLoc(), op.getTrueDestOperands(),
                     resourceTimepointMap, builder),
      op.getFalseDest(),
      expandOperands(op.getLoc(), op.getFalseDestOperands(),
                     resourceTimepointMap, builder));
  op.erase();
}

static void expandSwitchOp(mlir::cf::SwitchOp op,
                           IRMapping &resourceTimepointMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  auto caseOperands = llvm::to_vector(
      llvm::map_range(op.getCaseOperands(), [&](ValueRange operands) {
        return expandOperands(op.getLoc(), operands, resourceTimepointMap,
                              builder);
      }));
  auto asValueRange = [](ArrayRef<Value> ref) -> ValueRange { return ref; };
  builder.create<mlir::cf::SwitchOp>(
      op.getLoc(), op.getFlag(), op.getDefaultDestination(),
      expandOperands(op.getLoc(), op.getDefaultOperands(), resourceTimepointMap,
                     builder),
      op.getCaseValuesAttr(), op.getCaseDestinations(),
      llvm::to_vector(llvm::map_range(caseOperands, asValueRange)));
  op.erase();
}

// Tracks timepoints associated with resources based on awaits.
// By nature of SSA we will encounter these and setup the mapping before any
// user of the resulting resource performs a lookup, avoiding the need to
// perform an initial scan to populate the mapping.
static void expandAwaitOp(IREE::Stream::TimepointAwaitOp op,
                          IRMapping &resourceTimepointMap) {
  for (auto result : op.getResults()) {
    resourceTimepointMap.map(op.getTiedResultOperand(result),
                             op.getAwaitTimepoint());
  }
}

// Expands resource operands captured by a stream.async.execute |op| to await
// on the timepoints of those resources. In the case of back-to-back execution
// regions this performs the chaining of unreadied results to awaited operands.
static void expandAsyncExecuteOp(IREE::Stream::AsyncExecuteOp op,
                                 IRMapping &resourceTimepointMap) {
  OpBuilder builder(op);
  SetVector<Value> newTimepoints;
  SmallVector<Value> newOperands;
  SmallVector<Value> newOperandSizes;
  if (op.getAwaitTimepoint()) {
    newTimepoints.insert(op.getAwaitTimepoint());
  }
  for (auto [operand, operandSize] : llvm::zip_equal(
           op.getResourceOperands(), op.getResourceOperandSizes())) {
    auto timepointOperand =
        consumeTimepoint(op.getLoc(), operand, resourceTimepointMap, builder);
    if (newTimepoints.insert(timepointOperand.first)) {
      // Not yet covered; need to add the timepoint.
      newOperands.push_back(timepointOperand.second);
      newOperandSizes.push_back(operandSize);
    } else {
      // Already covered in the timepoints set on this op, we can go right to
      // the source.
      newOperands.push_back(timepointOperand.second);
      newOperandSizes.push_back(operandSize);
    }
  }
  if (newTimepoints.empty()) {
    op.getAwaitTimepointMutable().clear();
  } else {
    auto newTimepoint = IREE::Stream::TimepointJoinOp::join(
        op.getLoc(), newTimepoints.takeVector(), builder);
    op.getAwaitTimepointMutable().assign(newTimepoint);
  }
  op.getResourceOperandsMutable().assign(newOperands);
  op.getResourceOperandSizesMutable().assign(newOperandSizes);
  for (auto result : op.getResults()) {
    resourceTimepointMap.map(result, op.getResultTimepoint());
  }
}

// Recursively expands resources into (timepoint, resource) in |op|.
// Resource timepoint chains are established when possible by looking through
// awaits.
static void expandTimepoints(Operation *op, ExpandedGlobalMap &globalMap,
                             IRMapping &resourceTimepointMap) {
  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
    expandGlobalLoadOp(loadOp, globalMap, resourceTimepointMap);
  } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOp>(op)) {
    expandGlobalStoreOp(storeOp, globalMap, resourceTimepointMap);
  } else if (auto initializerOp = dyn_cast<IREE::Util::InitializerOp>(op)) {
    expandInitializerOp(initializerOp, globalMap, resourceTimepointMap);
  } else if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
    expandFuncOp(funcOp, globalMap, resourceTimepointMap);
  } else if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
    expandCallOp(callOp, resourceTimepointMap);
  } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    expandReturnOp(returnOp, resourceTimepointMap);
  } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(op)) {
    expandBranchOp(branchOp, resourceTimepointMap);
  } else if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
    expandCondBranchOp(condBranchOp, resourceTimepointMap);
  } else if (auto switchOp = dyn_cast<mlir::cf::SwitchOp>(op)) {
    expandSwitchOp(switchOp, resourceTimepointMap);
  } else if (auto awaitOp = dyn_cast<IREE::Stream::TimepointAwaitOp>(op)) {
    expandAwaitOp(awaitOp, resourceTimepointMap);
  } else if (auto executeOp = dyn_cast<IREE::Stream::AsyncExecuteOp>(op)) {
    expandAsyncExecuteOp(executeOp, resourceTimepointMap);
  }
}

//===----------------------------------------------------------------------===//
// --iree-stream-propagate-timepoints
//===----------------------------------------------------------------------===//

// This does a relatively mechanical transformation of a module to expand all
// resource values (and globals) into a (timepoint, resource) pair. Ops that
// consume resources and timepoints attempt to dereference those expanded
// timepoints and bypass waits, effectively chaining execution across
// asynchronous ops.
//
// This is designed to be composed with generic optimization passes like global
// fusion/folding and IPO and as such performs all transformations locally. For
// example, calls are always updated to take/return timepoints and results are
// always awaited, with the elision/deduplication/etc left until cleanup.
struct PropagateTimepointsPass
    : public IREE::Stream::impl::PropagateTimepointsPassBase<
          PropagateTimepointsPass> {
  void runOnOperation() override {
    auto rootOp = getOperation();

    // Expand all util.global ops holding resources into (timepoint, resource).
    auto globalMap = expandResourceGlobals(rootOp);

    // Walk the entire IR tree and expand the globals.
    // We could do this via pattern application but that gets much trickier to
    // manage with the expansion as we'd need to prevent ourselves from
    // expanding multiple times.
    for (auto callableOp : rootOp.getOps<mlir::CallableOpInterface>()) {
      IRMapping resourceTimepointMap;
      expandTimepoints(callableOp, globalMap, resourceTimepointMap);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
