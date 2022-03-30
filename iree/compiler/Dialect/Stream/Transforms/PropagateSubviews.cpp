// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-propagate-subviews"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
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
  IREE::Util::GlobalOp resourceSizeOp;
  IREE::Util::GlobalOp subviewOffsetOp;
  IREE::Util::GlobalOp subviewLengthOp;
};
using ExpandedGlobalMap = DenseMap<StringRef, ExpandedGlobal>;

// Expands each !stream.resource global in |rootOp| to have a matching
// parent resource size and subview range. Does not behave optimally if there
// already exist offset globals as duplicates will get added and we'll need to
// rely on global fusion to get rid of them. Note that this only expands globals
// and does not yet update use sites - we just need the ops to reference.
static ExpandedGlobalMap expandResourceGlobals(Operation *rootOp) {
  ExpandedGlobalMap expandedGlobals;

  // Gather all of the resource globals in the root.
  for (auto &region : rootOp->getRegions()) {
    for (auto globalOp : region.getOps<IREE::Util::GlobalOp>()) {
      if (!globalOp.type().isa<IREE::Stream::ResourceType>()) continue;
      expandedGlobals[globalOp.getName()].resourceOp = globalOp;
    }
  }

  // Expand each global by adding the offset right next to it.
  SymbolTable symbolTable(rootOp);
  auto indexType = IndexType::get(rootOp->getContext());
  for (auto &it : expandedGlobals) {
    auto &global = it.second;
    OpBuilder builder(global.resourceOp);
    builder.setInsertionPointAfter(global.resourceOp);

    auto sizeName = (global.resourceOp.getName() + "__storage_size").str();
    auto sizeOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), sizeName,
        /*isMutable=*/true, indexType);
    sizeOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(sizeOp);
    global.resourceSizeOp = sizeOp;

    auto offsetName = (global.resourceOp.getName() + "__offset").str();
    auto offsetOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), offsetName,
        /*isMutable=*/true, indexType);
    offsetOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(offsetOp);
    global.subviewOffsetOp = offsetOp;

    auto lengthName = (global.resourceOp.getName() + "__length").str();
    auto lengthOp = builder.create<IREE::Util::GlobalOp>(
        global.resourceOp.getLoc(), lengthName,
        /*isMutable=*/true, indexType);
    lengthOp.setVisibility(global.resourceOp.getVisibility());
    symbolTable.insert(lengthOp);
    global.subviewLengthOp = lengthOp;
  }

  return expandedGlobals;
}

//===----------------------------------------------------------------------===//
// Structural IR rewriting patterns
//===----------------------------------------------------------------------===//

static bool isResourceType(Type type) {
  return type.isa<IREE::Stream::ResourceType>();
}

// Returns true if an operands or results of |op| use !stream.resources.
static bool usesResources(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isResourceType) ||
         llvm::any_of(op->getResultTypes(), isResourceType);
}

// Expands resources in the given |types| list to (resource, size, offset, len).
// This could be changed to some iterator magic to avoid the alloc.
static SmallVector<Type> expandTypes(TypeRange types) {
  if (types.empty()) return {};
  auto indexType = IndexType::get(types.front().getContext());
  SmallVector<Type> newTypes;
  newTypes.reserve(types.size() * 2);
  for (auto type : types) {
    newTypes.push_back(type);
    if (isResourceType(type)) {
      newTypes.push_back(indexType);  // resource size
      newTypes.push_back(indexType);  // subview offset
      newTypes.push_back(indexType);  // subview length
    }
  }
  return newTypes;
}

struct Subview {
  Value resource;
  Value resourceSize;
  Value subviewOffset;
  Value subviewLength;
};
using SubviewMap = llvm::DenseMap<Value, Subview>;

// Attempts to find and consume a subview associated with |value|.
// Returns the subview - which may point at a different resource than |value|.
// In cases where no associated subview was found the subview will cover the
// entire resource (offset at 0, length at size).
static Subview consumeSubview(Location loc, Value value, SubviewMap &subviewMap,
                              IndexSet &indexSet, OpBuilder &builder) {
  // TODO(benvanik): follow ties on value to try to consume there; there are a
  // few other ops we could look through as well (such as select, where we could
  // join). For now we just look at immediate defining ops.
  auto mapIt = subviewMap.find(value);
  if (mapIt != subviewMap.end()) {
    return mapIt->second;
  }

  if (auto subviewOp = dyn_cast_or_null<IREE::Stream::ResourceSubviewOp>(
          value.getDefiningOp())) {
    Subview subview;
    subview.resource = subviewOp.source();
    subview.resourceSize = subviewOp.source_size();
    subview.subviewOffset = subviewOp.source_offset();
    subview.subviewLength = subviewOp.result_size();
    return subview;
  } else {
    Subview subview;
    subview.resource = value;
    subview.resourceSize =
        IREE::Util::SizeAwareTypeInterface::queryValueSize(loc, value, builder);
    subview.subviewOffset = indexSet.get(0);
    subview.subviewLength = subview.resourceSize;
    return subview;
  }
}

// Expands resources in |operands| into (resource, size, offset, length) tuples.
static SmallVector<Value> expandOperands(Location loc, ValueRange operands,
                                         SubviewMap &subviewMap,
                                         IndexSet &indexSet,
                                         OpBuilder &builder) {
  SmallVector<Value> result;
  result.reserve(operands.size() * 2);
  for (auto operand : operands) {
    if (isResourceType(operand.getType())) {
      auto subview =
          consumeSubview(loc, operand, subviewMap, indexSet, builder);
      result.push_back(subview.resource);
      result.push_back(subview.resourceSize);
      result.push_back(subview.subviewOffset);
      result.push_back(subview.subviewLength);
    } else {
      result.push_back(operand);
    }
  }
  return result;
}

static void expandSubviews(Operation *op, ExpandedGlobalMap &globalMap,
                           IndexSet &indexSet, SubviewMap &subviewMap);

// Recursively expands resources into (resource, size, offset, length) tuples
// within the given |region|. All branches, ops, and nested regions will be
// processed.
static void expandRegion(Region &region, ExpandedGlobalMap &globalMap,
                         IndexSet &indexSet, SubviewMap subviewMap) {
  if (region.empty()) return;

  // Update all block arguments.
  auto indexType = IndexType::get(region.getContext());
  for (auto &block : region.getBlocks()) {
    if (!llvm::any_of(block.getArgumentTypes(), isResourceType)) continue;

    // Insert and build a list of expanded (resource, size, offset) tuples.
    SmallVector<Subview> expansions;
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      auto arg = block.getArgument(i);
      if (!isResourceType(arg.getType())) continue;
      Subview subview;
      subview.resource = arg;
      subview.resourceSize =
          block.insertArgument(i + 1, indexType, arg.getLoc());
      subview.subviewOffset =
          block.insertArgument(i + 2, indexType, arg.getLoc());
      subview.subviewLength =
          block.insertArgument(i + 3, indexType, arg.getLoc());
      expansions.push_back(subview);
      subviewMap[arg] = subview;
    }

    // Insert subviews that we've sunk from callers.
    auto builder = OpBuilder::atBlockBegin(&block);
    for (auto &expansion : llvm::reverse(expansions)) {
      auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
          region.getLoc(), expansion.resource, expansion.resourceSize,
          expansion.subviewOffset, expansion.subviewLength);
      expansion.resource.replaceAllUsesExcept(subviewOp.result(), subviewOp);
    }
  }

  // Walk blocks forward in domination order so that we add dominating values to
  // the offset map. Note that DominanceInfo is just determined not to be
  // cool about things when there's only one block so we have to special case.
  if (region.hasOneBlock()) {
    for (auto &op :
         llvm::make_early_inc_range(region.front().getOperations())) {
      expandSubviews(&op, globalMap, indexSet, subviewMap);
    }
  } else {
    DominanceInfo domInfo(region.getParentOp());
    for (auto *blockInfo : llvm::breadth_first(domInfo.getRootNode(&region))) {
      auto *block = blockInfo->getBlock();
      for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
        expandSubviews(&op, globalMap, indexSet, subviewMap);
      }
    }
  }
}

// Moves resource subviews from global stores to loads.
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
static void expandGlobalLoadOp(IREE::Util::GlobalLoadOp op,
                               ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                               SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto indexType = builder.getIndexType();
  auto &expandedGlobal = globalMap[op.global()];
  Subview subview;
  subview.resource = op.result();
  subview.resourceSize =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.resourceSizeOp.getName())
          .result();
  subview.subviewOffset =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.subviewOffsetOp.getName())
          .result();
  subview.subviewLength =
      builder
          .create<IREE::Util::GlobalLoadOp>(
              op.getLoc(), indexType, expandedGlobal.subviewLengthOp.getName())
          .result();
  subviewMap[op.result()] = subview;
  auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
      op.getLoc(), subview.resource, subview.resourceSize,
      subview.subviewOffset, subview.subviewLength);
  op.result().replaceAllUsesExcept(subviewOp.result(), subviewOp);
}

// Moves resource subviews from global stores to loads.
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
static void expandGlobalStoreOp(IREE::Util::GlobalStoreOp op,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet, SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto subview =
      consumeSubview(op.getLoc(), op.value(), subviewMap, indexSet, builder);
  auto &expandedGlobal = globalMap[op.global()];
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.resource, expandedGlobal.resourceOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.resourceSize,
      expandedGlobal.resourceSizeOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.subviewOffset,
      expandedGlobal.subviewOffsetOp.getName());
  builder.create<IREE::Util::GlobalStoreOp>(
      op.getLoc(), subview.subviewLength,
      expandedGlobal.subviewLengthOp.getName());
  op.erase();
}

static void expandInitializerOp(IREE::Util::InitializerOp op,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet, SubviewMap &subviewMap) {
  expandRegion(op.getRegion(), globalMap, indexSet, subviewMap);
}

// Inserts subviews on resource arguments.
// Requires that the ExpandCallOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant subviews in cases where the call
// sites don't need a wait.
//
// Example:
//  func.func @foo(%0: !stream.resource)
//  ->
//  func.func @foo(%0: !stream.resource, %sz: index, %o: index, %l: index) {
//    %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
static void expandFuncOp(mlir::func::FuncOp op, ExpandedGlobalMap &globalMap,
                         IndexSet &indexSet, SubviewMap &subviewMap) {
  auto oldType = op.getFunctionType();
  auto inputTypes = expandTypes(oldType.getInputs());
  auto resultTypes = expandTypes(oldType.getResults());
  auto newType = FunctionType::get(op.getContext(), inputTypes, resultTypes);
  if (newType != oldType) {
    op.setType(newType);
  }
  expandRegion(op.getRegion(), globalMap, indexSet, subviewMap);
}

// Splits resource operands and results into (resource, resourceSize,
// subviewOffset, subviewLength).
// Requires that the ExpandFuncOp/ExpandReturnOp patterns handle migrating the
// await.
//
// NOTE: this needs IPO to remove redundant values in cases where the call sites
// don't need a subview.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  %r = call @foo(%1)
//  ->
//  %r, %rsz, %ro, %rl = call @foo(%0, %sz, %o, %l)
//  %2 = stream.resource.subview %r[%ro] : {%rsz} -> {%rl}
static void expandCallOp(mlir::func::CallOp op, IndexSet &indexSet,
                         SubviewMap &subviewMap) {
  if (!usesResources(op)) return;

  // Build the new call op with expanded operands and results.
  OpBuilder builder(op);
  auto operands =
      expandOperands(op.getLoc(), op.operands(), subviewMap, indexSet, builder);
  auto resultTypes = expandTypes(op.getResultTypes());
  auto newOp = builder.create<mlir::func::CallOp>(op.getLoc(), op.getCallee(),
                                                  resultTypes, operands);

  // Insert subviews on results that we are sinking across the call edge.
  // The hope is that by moving the subviews here we can fold with uses inside
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
    Subview subview;
    subview.resource = newOp.getResult(newIdx++);
    subview.resourceSize = newOp.getResult(newIdx++);
    subview.subviewOffset = newOp.getResult(newIdx++);
    subview.subviewLength = newOp.getResult(newIdx++);
    subviewMap[subview.resource] = subview;
    auto subviewOp = builder.create<IREE::Stream::ResourceSubviewOp>(
        op.getLoc(), subview.resource, subview.resourceSize,
        subview.subviewOffset, subview.subviewLength);
    oldResult.replaceAllUsesWith(subviewOp.result());
  }

  op.erase();
}

// Moves subviews to callers upon return.
// Requires that the ExpandFuncOp/ExpandCallOp patterns handle migrating the
// await.
//
// Example:
//  %1 = stream.resource.subview %0[%o] : {%sz} -> {%l}
//  return %1
//  ->
//  return %0, %sz, %o, %l
static void expandReturnOp(mlir::func::ReturnOp op, IndexSet &indexSet,
                           SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  auto operands =
      expandOperands(op.getLoc(), op.operands(), subviewMap, indexSet, builder);
  builder.create<mlir::func::ReturnOp>(op.getLoc(), operands);
  op.erase();
}

// Moves subviews across branches.
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
                           SubviewMap &subviewMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getDestOperands(), subviewMap,
                                 indexSet, builder);
  builder.create<mlir::cf::BranchOp>(op.getLoc(), op.getDest(), operands);
  op.erase();
}

static void expandCondBranchOp(mlir::cf::CondBranchOp op, IndexSet &indexSet,
                               SubviewMap &subviewMap) {
  if (!usesResources(op)) return;
  OpBuilder builder(op);
  builder.create<mlir::cf::CondBranchOp>(
      op.getLoc(), op.getCondition(), op.getTrueDest(),
      expandOperands(op.getLoc(), op.getTrueDestOperands(), subviewMap,
                     indexSet, builder),
      op.getFalseDest(),
      expandOperands(op.getLoc(), op.getFalseDestOperands(), subviewMap,
                     indexSet, builder));
  op.erase();
}

// Recursively expands resources into (resource, size, offset, length) in |op|.
static void expandSubviews(Operation *op, ExpandedGlobalMap &globalMap,
                           IndexSet &indexSet, SubviewMap &subviewMap) {
  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
    expandGlobalLoadOp(loadOp, globalMap, indexSet, subviewMap);
  } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOp>(op)) {
    expandGlobalStoreOp(storeOp, globalMap, indexSet, subviewMap);
  } else if (auto initializerOp = dyn_cast<IREE::Util::InitializerOp>(op)) {
    expandInitializerOp(initializerOp, globalMap, indexSet, subviewMap);
  } else if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
    expandFuncOp(funcOp, globalMap, indexSet, subviewMap);
  } else if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
    expandCallOp(callOp, indexSet, subviewMap);
  } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    expandReturnOp(returnOp, indexSet, subviewMap);
  } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(op)) {
    expandBranchOp(branchOp, indexSet, subviewMap);
  } else if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
    expandCondBranchOp(condBranchOp, indexSet, subviewMap);
  }
}

//===----------------------------------------------------------------------===//
// -iree-stream-propagate-subviews
//===----------------------------------------------------------------------===//

// This does a relatively mechanical transformation of a module to expand all
// resource values (and globals) into (resource, size, offset, length) tuples.
//
// This is designed to be composed with generic optimization passes like global
// fusion/folding and IPO and as such performs all transformations locally. For
// example, calls are always updated to take/return subview ranges and results
// are always wrapped in a stream.resource.subview, with the
// elision/deduplication/etc left until cleanup.
class PropagateSubviewsPass
    : public PropagateSubviewsBase<PropagateSubviewsPass> {
 public:
  PropagateSubviewsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto rootOp = getOperation();

    // Expand all util.global ops holding resources into resource and subview.
    auto globalMap = expandResourceGlobals(rootOp);

    // Walk the entire IR tree and expand the globals.
    // We could do this via pattern application but that gets much trickier to
    // manage with the expansion as we'd need to prevent ourselves from
    // expanding multiple times.
    for (auto callableOp : rootOp.getOps<mlir::CallableOpInterface>()) {
      // NOTE: the callable may be empty (like when an extern) - we still want
      // to process it but don't need an IndexSet.
      auto *region = callableOp.getCallableRegion();
      IndexSet indexSet(callableOp.getLoc(),
                        !region || region->empty()
                            ? OpBuilder(callableOp)
                            : OpBuilder::atBlockBegin(&region->front()));
      SubviewMap subviewMap;
      expandSubviews(callableOp, globalMap, indexSet, subviewMap);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateSubviewsPass() {
  return std::make_unique<PropagateSubviewsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
