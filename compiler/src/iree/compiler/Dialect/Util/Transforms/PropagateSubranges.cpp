// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "iree/compiler/Utils/IntegerSet.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-util-propagate-subranges"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_PROPAGATESUBRANGESPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

// This pass is paired with the subrange type. Any type implementing the
// interface can be used.
static bool isResourceType(Type type) {
  return llvm::isa<IREE::Util::SubrangeTypeInterface>(type);
}

//===----------------------------------------------------------------------===//
// Global handling
//===----------------------------------------------------------------------===//

struct ExpandedGlobal {
  IREE::Util::GlobalOp resourceOp;
  IREE::Util::GlobalOp resourceSizeOp;
  IREE::Util::GlobalOp subrangeOffsetOp;
  IREE::Util::GlobalOp subrangeLengthOp;
};
using ExpandedGlobalMap = DenseMap<StringRef, ExpandedGlobal>;

// Expands each !stream.resource global in |rootOp| to have a matching
// parent resource size and subrange range. Does not behave optimally if there
// already exist offset globals as duplicates will get added and we'll need to
// rely on global fusion to get rid of them. Note that this only expands globals
// and does not yet update use sites - we just need the ops to reference.
static ExpandedGlobalMap expandResourceGlobals(Operation *rootOp,
                                               SymbolTable &symbolTable) {
  ExpandedGlobalMap expandedGlobals;

  // Gather all of the resource globals in the root.
  for (auto &region : rootOp->getRegions()) {
    for (auto globalOp : region.getOps<IREE::Util::GlobalOp>()) {
      if (!isResourceType(globalOp.getType()))
        continue;
      expandedGlobals[globalOp.getName()].resourceOp = globalOp;
    }
  }

  // Expand each global by adding the offset right next to it.
  auto indexType = IndexType::get(rootOp->getContext());
  for (auto &it : expandedGlobals) {
    auto &global = it.second;
    OpBuilder builder(global.resourceOp);
    builder.setInsertionPointAfter(global.resourceOp);

    auto sizeName = (global.resourceOp.getName() + "__storage_size").str();
    auto sizeOp = IREE::Util::GlobalOp::create(
        builder, global.resourceOp.getLoc(), sizeName,
        /*isMutable=*/true, indexType);
    sizeOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(sizeOp);
    global.resourceSizeOp = sizeOp;

    auto offsetName = (global.resourceOp.getName() + "__offset").str();
    auto offsetOp = IREE::Util::GlobalOp::create(
        builder, global.resourceOp.getLoc(), offsetName,
        /*isMutable=*/true, indexType);
    offsetOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(offsetOp);
    global.subrangeOffsetOp = offsetOp;

    auto lengthName = (global.resourceOp.getName() + "__length").str();
    auto lengthOp = IREE::Util::GlobalOp::create(
        builder, global.resourceOp.getLoc(), lengthName,
        /*isMutable=*/true, indexType);
    lengthOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(lengthOp);
    global.subrangeLengthOp = lengthOp;
  }

  return expandedGlobals;
}

//===----------------------------------------------------------------------===//
// Structural IR rewriting patterns
//===----------------------------------------------------------------------===//

// Returns true if an operands or results of |op| use !stream.resources.
static bool usesResources(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isResourceType) ||
         llvm::any_of(op->getResultTypes(), isResourceType);
}

static void expandType(Type type, SmallVectorImpl<Type> &newTypes) {
  newTypes.push_back(type);
  if (isResourceType(type)) {
    // resource size, subrance offset, subrange length
    newTypes.append(3, IndexType::get(type.getContext()));
  }
}

// Expands resources in the given |types| list to (resource, size, offset, len).
// This could be changed to some iterator magic to avoid the alloc.
static SmallVector<Type> expandTypes(TypeRange types) {
  if (types.empty())
    return {};
  SmallVector<Type> newTypes;
  newTypes.reserve(types.size() * 2);
  for (auto type : types) {
    expandType(type, newTypes);
  }
  return newTypes;
}

struct Subrange {
  Value resource;
  Value resourceSize;
  Value subrangeOffset;
  Value subrangeLength;
  IREE::Util::SubrangeTypeInterface getResourceType() {
    return llvm::cast<IREE::Util::SubrangeTypeInterface>(resource.getType());
  }
};
using SubrangeMap = llvm::DenseMap<Value, Subrange>;

// Attempts to find and consume a subrange associated with |value|.
// Returns the subrange - which may point at a different resource than |value|.
// In cases where no associated subrange was found the subrange will cover the
// entire resource (offset at 0, length at size).
static Subrange consumeSubrange(Location loc, Value value,
                                SubrangeMap &subrangeMap, IndexSet &indexSet,
                                OpBuilder &builder) {
  // TODO(benvanik): follow ties on value to try to consume there; there are a
  // few other ops we could look through as well (such as select, where we could
  // join). For now we just look at immediate defining ops.
  auto mapIt = subrangeMap.find(value);
  if (mapIt != subrangeMap.end()) {
    return mapIt->second;
  }

  if (auto subrangeOp = dyn_cast_or_null<IREE::Util::SubrangeOpInterface>(
          value.getDefiningOp())) {
    Subrange subrange;
    subrange.resource = subrangeOp.getSubrangeResource();
    subrange.resourceSize = subrangeOp.getSubrangeResourceSize();
    subrange.subrangeOffset = subrangeOp.getSubrangeOffset();
    subrange.subrangeLength = subrangeOp.getSubrangeLength();
    subrangeMap[value] = subrange;
    return subrange;
  } else {
    Subrange subrange;
    subrange.resource = value;
    subrange.resourceSize =
        IREE::Util::SizeAwareTypeInterface::queryValueSize(loc, value, builder);
    subrange.subrangeOffset = indexSet.get(0);
    subrange.subrangeLength = subrange.resourceSize;
    subrangeMap[value] = subrange;
    return subrange;
  }
}

static void expandOperand(Location loc, Value operand,
                          SmallVectorImpl<Value> &newOperands,
                          SubrangeMap &subrangeMap, IndexSet &indexSet,
                          OpBuilder &builder) {
  if (isResourceType(operand.getType())) {
    auto subrange =
        consumeSubrange(loc, operand, subrangeMap, indexSet, builder);
    llvm::append_values(newOperands, subrange.resource, subrange.resourceSize,
                        subrange.subrangeOffset, subrange.subrangeLength);
  } else {
    newOperands.push_back(operand);
  }
}

// Expands resources in |operands| into (resource, size, offset, length) tuples.
static SmallVector<Value> expandOperands(Location loc, ValueRange operands,
                                         SubrangeMap &subrangeMap,
                                         IndexSet &indexSet,
                                         OpBuilder &builder) {
  SmallVector<Value> result;
  result.reserve(operands.size() * 2);
  for (auto operand : operands) {
    expandOperand(loc, operand, result, subrangeMap, indexSet, builder);
  }
  return result;
}

static void expandSubranges(Operation *op, SymbolTable &symbolTable,
                            ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                            SubrangeMap &subrangeMap);

// Recursively expands resources into (resource, size, offset, length) tuples
// within the given |region|. All branches, ops, and nested regions will be
// processed.
static void expandRegion(Region &region, bool canModifyEntryBlock,
                         SymbolTable &symbolTable, ExpandedGlobalMap &globalMap,
                         IndexSet &indexSet, SubrangeMap subrangeMap) {
  if (region.empty())
    return;

  // Update all block arguments.
  auto indexType = IndexType::get(region.getContext());
  for (auto &block : region.getBlocks()) {
    if (!llvm::any_of(block.getArgumentTypes(), isResourceType))
      continue;
    if (block.isEntryBlock() && !canModifyEntryBlock)
      continue;

    // Insert and build a list of expanded (resource, size, offset) tuples.
    SmallVector<Subrange> expansions;
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      auto arg = block.getArgument(i);
      if (!isResourceType(arg.getType()))
        continue;
      Subrange subrange;
      subrange.resource = arg;
      subrange.resourceSize =
          block.insertArgument(i + 1, indexType, arg.getLoc());
      subrange.subrangeOffset =
          block.insertArgument(i + 2, indexType, arg.getLoc());
      subrange.subrangeLength =
          block.insertArgument(i + 3, indexType, arg.getLoc());
      expansions.push_back(subrange);
      subrangeMap[arg] = subrange;
    }

    // Insert subranges that we've sunk from callers.
    auto builder = OpBuilder::atBlockBegin(&block);
    for (auto &expansion : llvm::reverse(expansions)) {
      auto newSubrange = expansion.getResourceType().createSubrangeOp(
          region.getLoc(), expansion.resource, expansion.resourceSize,
          expansion.subrangeOffset, expansion.subrangeLength, builder);
      expansion.resource.replaceAllUsesExcept(newSubrange,
                                              newSubrange.getDefiningOp());
    }
  }

  // Walk blocks forward in domination order so that we add dominating values to
  // the offset map. Note that DominanceInfo is just determined not to be
  // cool about things when there's only one block so we have to special case.
  if (region.hasOneBlock()) {
    for (auto &op :
         llvm::make_early_inc_range(region.front().getOperations())) {
      expandSubranges(&op, symbolTable, globalMap, indexSet, subrangeMap);
    }
  } else {
    DominanceInfo domInfo(region.getParentOp());
    for (auto *blockInfo : llvm::breadth_first(domInfo.getRootNode(&region))) {
      auto *block = blockInfo->getBlock();
      for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
        expandSubranges(&op, symbolTable, globalMap, indexSet, subrangeMap);
      }
    }
  }
}

// Recursively expands all regions on the op.
static void expandRegions(Operation *op, bool canModifyEntryBlock,
                          SymbolTable &symbolTable,
                          ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                          SubrangeMap subrangeMap) {
  for (auto &region : op->getRegions()) {
    expandRegion(region, canModifyEntryBlock, symbolTable, globalMap, indexSet,
                 subrangeMap);
  }
}

// Updates the |subrangeMap| with the combined subrange of |op| and its source.
static void updateSubrangeOp(IREE::Util::SubrangeOpInterface op,
                             IndexSet &indexSet, SubrangeMap &subrangeMap) {
  // Ignore ops that are already in the map (we likely inserted them ourselves
  // earlier).
  auto resultResource = op.getSubrangeResult();
  if (!resultResource)
    return;
  if (subrangeMap.count(resultResource))
    return;

  // Get the subrange of the source resource which we should have by way of the
  // other insertions (func/block args, etc).
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto sourceSubrange = consumeSubrange(op.getLoc(), op.getSubrangeResource(),
                                        subrangeMap, indexSet, builder);
  if (op.getSubrangeResource() == sourceSubrange.resource)
    return;

  // Update the subrange in the map by adding the source offset and the local
  // offset from the op. Future ops that consume subranges will reference back
  // to the source with this subrange.
  Subrange updatedSubrange;
  updatedSubrange.resource = sourceSubrange.resource;
  updatedSubrange.resourceSize = sourceSubrange.resourceSize;
  updatedSubrange.subrangeOffset = builder.createOrFold<arith::AddIOp>(
      op.getLoc(), sourceSubrange.subrangeOffset, op.getSubrangeOffset());
  updatedSubrange.subrangeLength = op.getSubrangeLength();
  subrangeMap[resultResource] = updatedSubrange;
}

// Moves resource subranges from global stores to loads.
// Requires that the ExpandGlobalStoreOp pattern elides the await.
//
// Example:
//  %0 = util.global.load @foo : !stream.resource
//  ->
//  %0 = util.global.load @foo : !stream.resource
//  %s = util.global.load @foo_size : index
//  %o = util.global.load @foo_offset : index
//  %l = util.global.load @foo_length : index
//  %1 = stream.resource.subview %0[%o] :
//       !stream.resource<*>{%s} -> !stream.resource<*>{%l}
static void expandGlobalLoadOp(IREE::Util::GlobalLoadOpInterface op,
                               ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                               SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto &expandedGlobal = globalMap[op.getGlobalName()];
  Subrange subrange;
  subrange.resource = op.getLoadedGlobalValue();
  subrange.resourceSize =
      expandedGlobal.resourceSizeOp.createLoadOp(op.getLoc(), builder)
          .getLoadedGlobalValue();
  subrange.subrangeOffset =
      expandedGlobal.subrangeOffsetOp.createLoadOp(op.getLoc(), builder)
          .getLoadedGlobalValue();
  subrange.subrangeLength =
      expandedGlobal.subrangeLengthOp.createLoadOp(op.getLoc(), builder)
          .getLoadedGlobalValue();
  subrangeMap[op.getLoadedGlobalValue()] = subrange;
  auto newSubrange = subrange.getResourceType().createSubrangeOp(
      op.getLoc(), subrange.resource, subrange.resourceSize,
      subrange.subrangeOffset, subrange.subrangeLength, builder);
  op.getLoadedGlobalValue().replaceAllUsesExcept(newSubrange,
                                                 newSubrange.getDefiningOp());
}

// Moves resource subranges from global stores to loads.
// Requires that the ExpandGlobalLoadOp pattern inserts the await.
//
// Example:
//  %1 = stream.resource.subview %0[%o] :
//       !stream.resource<*>{%s} -> !stream.resource<*>{%l}
//  util.global.store %1, @foo : !stream.resource
//  ->
//  util.global.store %0, @foo : !stream.resource
//  util.global.store %s, @foo_size : index
//  util.global.store %o, @foo_offset : index
//  util.global.store %l, @foo_length : index
static void expandGlobalStoreOp(IREE::Util::GlobalStoreOpInterface op,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet, SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto subrange = consumeSubrange(op.getLoc(), op.getStoredGlobalValue(),
                                  subrangeMap, indexSet, builder);
  auto &expandedGlobal = globalMap[op.getGlobalName()];
  IREE::Util::GlobalStoreOp::create(builder, op.getLoc(), subrange.resource,
                                    expandedGlobal.resourceOp.getName());
  IREE::Util::GlobalStoreOp::create(builder, op.getLoc(), subrange.resourceSize,
                                    expandedGlobal.resourceSizeOp.getName());
  IREE::Util::GlobalStoreOp::create(builder, op.getLoc(),
                                    subrange.subrangeOffset,
                                    expandedGlobal.subrangeOffsetOp.getName());
  IREE::Util::GlobalStoreOp::create(builder, op.getLoc(),
                                    subrange.subrangeLength,
                                    expandedGlobal.subrangeLengthOp.getName());
  op.erase();
}

static void expandInitializerOp(IREE::Util::InitializerOp op,
                                SymbolTable &symbolTable,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet, SubrangeMap &subrangeMap) {
  expandRegion(op.getRegion(), /*canModifyEntryBlock=*/false, symbolTable,
               globalMap, indexSet, subrangeMap);
}

// Inserts subranges on resource arguments.
// Requires that the ExpandCallOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant subranges in cases where the call
// sites don't need a wait.
//
// Example:
//  util.func @foo(%0: !stream.resource)
//  ->
//  util.func @foo(%0: !stream.resource, %sz: index, %o: index, %l: index) {
//    %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
static void expandFuncOp(IREE::Util::FuncOp op, SymbolTable &symbolTable,
                         ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                         SubrangeMap &subrangeMap) {
  // Ignore public/external function signatures but still convert regions.
  bool canModifyEntryBlock = !IREE::Util::isPublicOrExternal(op);
  if (canModifyEntryBlock) {
    op.expandSignature(
        [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
          expandType(type, newTypes);
        },
        [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
          expandType(type, newTypes);
        });
  }
  expandRegion(op.getRegion(), canModifyEntryBlock, symbolTable, globalMap,
               indexSet, subrangeMap);
}

// Splits resource operands and results into (resource, resourceSize,
// subrangeOffset, subrangeLength).
// Requires that the ExpandFuncOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant values in cases where the call sites
// don't need a subrange.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  %r = util.call @foo(%1)
//  ->
//  %r, %rsz, %ro, %rl = util.call @foo(%0, %sz, %o, %l)
//  %2 = stream.resource.subview %r[%ro] : {%rsz} -> {%rl}
static void expandCallOp(IREE::Util::CallOp op, SymbolTable &symbolTable,
                         IndexSet &indexSet, SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;

  // Ignore calls to public/external functions.
  auto calleeOp = symbolTable.lookup<CallableOpInterface>(op.getCallee());
  if (IREE::Util::isPublicOrExternal(calleeOp))
    return;

  // Build the new call op with expanded operands and results.
  OpBuilder builder(op);
  auto newOp = op.cloneAndExpand(
      [&](unsigned i, Value operand, SmallVectorImpl<Value> &newOperands) {
        expandOperand(op.getLoc(), operand, newOperands, subrangeMap, indexSet,
                      builder);
      },
      [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
        expandType(type, newTypes);
      },
      builder);

  // Insert subranges on results that we are sinking across the call edge.
  // The hope is that by moving the subranges here we can fold with uses inside
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
    Subrange subrange;
    subrange.resource = newOp.getResult(newIdx++);
    subrange.resourceSize = newOp.getResult(newIdx++);
    subrange.subrangeOffset = newOp.getResult(newIdx++);
    subrange.subrangeLength = newOp.getResult(newIdx++);
    subrangeMap[subrange.resource] = subrange;
    auto newSubrange = subrange.getResourceType().createSubrangeOp(
        op.getLoc(), subrange.resource, subrange.resourceSize,
        subrange.subrangeOffset, subrange.subrangeLength, builder);
    oldResult.replaceAllUsesWith(newSubrange);
  }

  op.erase();
}

// Moves subranges to callers upon return.
// Requires that the ExpandFuncOp/ExpandCallOp patterns handle migrating the
// await.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  util.return %1
//  ->
//  util.return %0, %sz, %o, %l
static void expandReturnOp(IREE::Util::ReturnOp op, IndexSet &indexSet,
                           SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;
  if (IREE::Util::isPublicOrExternal(op->getParentOfType<IREE::Util::FuncOp>()))
    return;
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(), subrangeMap,
                                 indexSet, builder);
  IREE::Util::ReturnOp::create(builder, op.getLoc(), operands);
  op.erase();
}

// Moves subranges across branches.
// Requires that the ExpandFuncOp pattern handles modifying the block args.
//
// Example:
//    %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//    br ^bb1(%1)
//  ^bb1(%b):
//  ->
//    br ^bb1(%0, %sz, %o, %l)
//  ^bb1(%a, %b, %c, %d):
//    %1 = stream.resource.subview %a[%b] : {%c} -> {%d}
static void expandBranchOp(mlir::cf::BranchOp op, IndexSet &indexSet,
                           SubrangeMap &subrangeMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getDestOperands(), subrangeMap,
                                 indexSet, builder);
  mlir::cf::BranchOp::create(builder, op.getLoc(), op.getDest(), operands);
  op.erase();
}

static void expandCondBranchOp(mlir::cf::CondBranchOp op, IndexSet &indexSet,
                               SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  mlir::cf::CondBranchOp::create(
      builder, op.getLoc(), op.getCondition(), op.getTrueDest(),
      expandOperands(op.getLoc(), op.getTrueDestOperands(), subrangeMap,
                     indexSet, builder),
      op.getFalseDest(),
      expandOperands(op.getLoc(), op.getFalseDestOperands(), subrangeMap,
                     indexSet, builder));
  op.erase();
}

static ValueRange asValueRange(ArrayRef<Value> values) { return values; }

static void expandSwitchOp(mlir::cf::SwitchOp op, IndexSet &indexSet,
                           SubrangeMap &subrangeMap) {
  if (!usesResources(op))
    return;
  OpBuilder builder(op);
  auto caseOperands = llvm::to_vector(
      llvm::map_range(op.getCaseOperands(), [&](ValueRange operands) {
        return expandOperands(op.getLoc(), operands, subrangeMap, indexSet,
                              builder);
      }));
  mlir::cf::SwitchOp::create(
      builder, op.getLoc(), op.getFlag(), op.getDefaultDestination(),
      expandOperands(op.getLoc(), op.getDefaultOperands(), subrangeMap,
                     indexSet, builder),
      op.getCaseValuesAttr(), op.getCaseDestinations(),
      llvm::to_vector(llvm::map_range(caseOperands, asValueRange)));
  op.erase();
}

// Recursively expands resources into (resource, size, offset, length) in |op|.
// TODO(benvanik): make this a type switch.
static void expandSubranges(Operation *op, SymbolTable &symbolTable,
                            ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                            SubrangeMap &subrangeMap) {
  if (auto subrangeOp = dyn_cast<IREE::Util::SubrangeOpInterface>(op)) {
    return updateSubrangeOp(subrangeOp, indexSet, subrangeMap);
  }

  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
    return expandGlobalLoadOp(loadOp, globalMap, indexSet, subrangeMap);
  } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
    return expandGlobalStoreOp(storeOp, globalMap, indexSet, subrangeMap);
  } else if (auto initializerOp = dyn_cast<IREE::Util::InitializerOp>(op)) {
    return expandInitializerOp(initializerOp, symbolTable, globalMap, indexSet,
                               subrangeMap);
  } else if (auto funcOp = dyn_cast<IREE::Util::FuncOp>(op)) {
    return expandFuncOp(funcOp, symbolTable, globalMap, indexSet, subrangeMap);
  } else if (auto callOp = dyn_cast<IREE::Util::CallOp>(op)) {
    return expandCallOp(callOp, symbolTable, indexSet, subrangeMap);
  } else if (auto returnOp = dyn_cast<IREE::Util::ReturnOp>(op)) {
    return expandReturnOp(returnOp, indexSet, subrangeMap);
  } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(op)) {
    return expandBranchOp(branchOp, indexSet, subrangeMap);
  } else if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
    return expandCondBranchOp(condBranchOp, indexSet, subrangeMap);
  } else if (auto switchOp = dyn_cast<mlir::cf::SwitchOp>(op)) {
    return expandSwitchOp(switchOp, indexSet, subrangeMap);
  }

  // We could have a more generic thing here with RegionBranchOpInterface but
  // not all ops can contain subrange ops and some are isolated from above.
  // We could add an interface to ops we want to do this to, though, to at least
  // allow dialects to plug in. For now we just need SCF so this is hardcoded.
  if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
    return expandRegions(ifOp, /*canModifyEntryBlock=*/false, symbolTable,
                         globalMap, indexSet, subrangeMap);
  } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
    return expandRegions(forOp, /*canModifyEntryBlock=*/false, symbolTable,
                         globalMap, indexSet, subrangeMap);
  } else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
    return expandRegions(whileOp, /*canModifyEntryBlock=*/false, symbolTable,
                         globalMap, indexSet, subrangeMap);
  }
  // TODO(benvanik): also handle scf.yield: today we don't propagate across
  // return values.
}

//===----------------------------------------------------------------------===//
// -iree-util-propagate-subranges
//===----------------------------------------------------------------------===//

// This does a relatively mechanical transformation of a module to expand all
// resource values (and globals) into (resource, size, offset, length) tuples.
//
// This is designed to be composed with generic optimization passes like global
// fusion/folding and IPO and as such performs all transformations locally. For
// example, calls are always updated to take/return subrange ranges and results
// are always wrapped in a subrange op, with the elision/deduplication/etc left
// until cleanup.
class PropagateSubrangesPass
    : public impl::PropagateSubrangesPassBase<PropagateSubrangesPass> {
public:
  void runOnOperation() override {
    auto rootOp = getOperation();
    SymbolTable symbolTable(rootOp);

    // Expand all util.global ops holding resources into resource and subrange.
    auto globalMap = expandResourceGlobals(rootOp, symbolTable);

    // Walk the entire IR tree and expand the globals.
    // We could do this via pattern application but that gets much trickier to
    // manage with the expansion as we'd need to prevent ourselves from
    // expanding multiple times.
    for (auto callableOp : rootOp.getOps<mlir::CallableOpInterface>()) {
      // NOTE: the callable may be empty (like when an extern) - we still want
      // to process it but don't need an IndexSet.
      auto *region = callableOp.getCallableRegion();
      if (!region || region->empty())
        continue;
      IndexSet indexSet(callableOp.getLoc(),
                        OpBuilder::atBlockBegin(&region->front()));
      SubrangeMap subrangeMap;
      expandSubranges(callableOp, symbolTable, globalMap, indexSet,
                      subrangeMap);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
