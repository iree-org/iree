// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-global-opt-expand-tensor-shapes"

namespace mlir::iree_compiler::GlobalOptimization {
namespace {

// TODO(benvanik): factor out into a generic util pass base that lets us share
// with other expanded type propagation passes. The walking of
// functions/blocks/globals/etc are the same across all of them and only the
// exact type expansion and consumption/query ops differ.

//===----------------------------------------------------------------------===//
// Global handling
//===----------------------------------------------------------------------===//

struct ExpandedGlobal {
  IREE::Util::GlobalOp tensorOp;
  SmallVector<IREE::Util::GlobalOp> dynamicDimOps;
};
using ExpandedGlobalMap = DenseMap<StringRef, ExpandedGlobal>;

static bool isDynamicTensor(Type type) {
  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    return !tensorType.hasStaticShape();
  }
  return false;
}

// Expands each dynamically-shaped tensor global in |rootOp| to have one global
// for each dynamic dimension. Does not behave optimally if there already exist
// dynamic dims as globals as duplicates will get added and we'll need to rely
// on global fusion to get rid of them. Note that this only expands globals and
// does not yet update use sites - we just need the ops to reference.
static ExpandedGlobalMap expandGlobalTensorDims(Operation *rootOp) {
  ExpandedGlobalMap expandedGlobals;

  // Gather all of the dynamically-shaped tensor globals in the root.
  for (auto &region : rootOp->getRegions()) {
    for (auto globalOp : region.getOps<IREE::Util::GlobalOp>()) {
      if (isDynamicTensor(globalOp.getType())) {
        expandedGlobals[globalOp.getName()].tensorOp = globalOp;
      }
    }
  }

  // Expand each global by adding one global per dynamic dim beside it.
  SymbolTable symbolTable(rootOp);
  auto indexType = IndexType::get(rootOp->getContext());
  for (auto &it : expandedGlobals) {
    auto &global = it.second;
    OpBuilder builder(global.tensorOp);
    builder.setInsertionPointAfter(global.tensorOp);

    auto tensorType = llvm::cast<RankedTensorType>(global.tensorOp.getType());
    for (auto it : llvm::enumerate(tensorType.getShape())) {
      if (ShapedType::isDynamic(it.value())) {
        auto dimName =
            (global.tensorOp.getName() + "__d" + std::to_string(it.index()))
                .str();
        auto dimOp = builder.create<IREE::Util::GlobalOp>(
            global.tensorOp.getLoc(), dimName,
            /*isMutable=*/true, indexType);
        dimOp.setVisibility(global.tensorOp.getVisibility());
        symbolTable.insert(dimOp);
        global.dynamicDimOps.push_back(dimOp);
      }
    }
  }

  return expandedGlobals;
}

//===----------------------------------------------------------------------===//
// Structural IR rewriting patterns
//===----------------------------------------------------------------------===//

// Returns true if operands or results of |op| use dynamically-shaped tensors.
static bool usesDynamicTensors(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isDynamicTensor) ||
         llvm::any_of(op->getResultTypes(), isDynamicTensor);
}

// Expands tensors in the given |types| list to (tensor, dynamic dims...).
// This could be changed to some iterator magic to avoid the alloc.
static SmallVector<Type> expandTypes(TypeRange types) {
  if (types.empty())
    return {};
  auto indexType = IndexType::get(types.front().getContext());
  SmallVector<Type> newTypes;
  newTypes.reserve(types.size() * 2);
  for (auto type : types) {
    newTypes.push_back(type);
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
      newTypes.append(tensorType.getNumDynamicDims(), indexType);
    }
  }
  return newTypes;
}

struct ExpandedValue {
  Value tensor;
  SmallVector<Value> dynamicDims;
};
using TensorDimMap = llvm::DenseMap<Value, ExpandedValue>;

// Attempts to find and consume tensor metadata associated with |value|.
static ExpandedValue consumeExpandedValue(Location loc, Value value,
                                          TensorDimMap &tensorDimMap,
                                          IndexSet &indexSet,
                                          OpBuilder &builder) {
  // TODO(benvanik): follow ties on value to try to consume there; there are a
  // few other ops we could look through as well (such as select, where we could
  // join). For now we just look at immediate defining ops.
  auto mapIt = tensorDimMap.find(value);
  if (mapIt != tensorDimMap.end()) {
    return mapIt->second;
  }

  // If the value comes from a tie shape we can bypass the slower checks.
  // This happens a lot during expansion as we'll expand function and block args
  // and insert ties before processing nested ops that consume them.
  if (auto tieShapeOp = dyn_cast_or_null<IREE::Flow::TensorTieShapeOp>(
          value.getDefiningOp())) {
    ExpandedValue expandedValue;
    expandedValue.tensor = tieShapeOp.getOperand();
    expandedValue.dynamicDims = llvm::to_vector(tieShapeOp.getDynamicDims());
    return expandedValue;
  }

  // Perform deeper dimension analysis or insert dim ops (worst case).
  ExpandedValue expandedValue;
  expandedValue.tensor = value;
  expandedValue.dynamicDims =
      IREE::Util::buildDynamicDimsForValue(loc, value, builder);
  return expandedValue;
}

// Expands tensor in |operands| into (tensor, dynamic dims...) tuples.
static SmallVector<Value> expandOperands(Location loc, ValueRange operands,
                                         TensorDimMap &tensorDimMap,
                                         IndexSet &indexSet,
                                         OpBuilder &builder) {
  SmallVector<Value> result;
  result.reserve(operands.size() * 2);
  for (auto operand : operands) {
    if (isDynamicTensor(operand.getType())) {
      auto expandedValue =
          consumeExpandedValue(loc, operand, tensorDimMap, indexSet, builder);
      result.push_back(expandedValue.tensor);
      result.append(expandedValue.dynamicDims);
    } else {
      result.push_back(operand);
    }
  }
  return result;
}

static void expandTensorDims(Operation *op, ExpandedGlobalMap &globalMap,
                             IndexSet &indexSet, TensorDimMap &tensorDimMap);

// Recursively expands tensors into (tensor, dynamic dims...) tuples within the
// given |region|. All branches, ops, and nested regions will be processed.
static void expandRegion(Region &region, ExpandedGlobalMap &globalMap,
                         IndexSet &indexSet, TensorDimMap tensorDimMap) {
  if (region.empty())
    return;

  // Update all block arguments.
  auto indexType = IndexType::get(region.getContext());
  for (auto &block : region.getBlocks()) {
    if (!llvm::any_of(block.getArgumentTypes(), isDynamicTensor))
      continue;

    // Insert and build a list of expanded (tensor, dynamic dims...) tuples.
    SmallVector<ExpandedValue> expansions;
    for (int i = block.getNumArguments() - 1; i >= 0; --i) {
      auto arg = block.getArgument(i);
      auto tensorType = llvm::dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType || tensorType.hasStaticShape())
        continue;
      ExpandedValue expandedValue;
      expandedValue.tensor = arg;
      for (unsigned j = 0; j < tensorType.getNumDynamicDims(); ++j) {
        expandedValue.dynamicDims.push_back(
            block.insertArgument(i + 1 + j, indexType, arg.getLoc()));
      }
      expansions.push_back(expandedValue);
      tensorDimMap[arg] = expandedValue;
    }

    // Insert shape ties that we've sunk from callers.
    auto builder = OpBuilder::atBlockBegin(&block);
    for (auto &expansion : llvm::reverse(expansions)) {
      auto tieShapeOp = builder.create<IREE::Flow::TensorTieShapeOp>(
          region.getLoc(), expansion.tensor.getType(), expansion.tensor,
          expansion.dynamicDims);
      expansion.tensor.replaceAllUsesExcept(tieShapeOp.getResult(), tieShapeOp);
    }
  }

  // Walk blocks forward in domination order so that we add dominating values to
  // the dim map. Note that DominanceInfo is just determined not to be
  // cool about things when there's only one block so we have to special case.
  if (region.hasOneBlock()) {
    for (auto &op :
         llvm::make_early_inc_range(region.front().getOperations())) {
      expandTensorDims(&op, globalMap, indexSet, tensorDimMap);
    }
  } else {
    DominanceInfo domInfo(region.getParentOp());
    for (auto *blockInfo : llvm::breadth_first(domInfo.getRootNode(&region))) {
      auto *block = blockInfo->getBlock();
      for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
        expandTensorDims(&op, globalMap, indexSet, tensorDimMap);
      }
    }
  }
}

// Insert shape ties on results that we are sinking across the call edge. The
// hope is that by moving the ties here we can fold with queries inside of
// this function.
static void retieResults(Operation *op, Operation *newOp,
                         TensorDimMap &tensorDimMap) {
  OpBuilder builder(newOp);

  builder.setInsertionPointAfter(newOp);
  unsigned newIdx = 0;
  for (unsigned oldIdx = 0; oldIdx < op->getNumResults(); ++oldIdx) {
    auto oldResult = op->getResult(oldIdx);
    auto tensorType = llvm::dyn_cast<RankedTensorType>(oldResult.getType());
    if (!tensorType || tensorType.hasStaticShape()) {
      auto newResult = newOp->getResult(newIdx++);
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    ExpandedValue expandedValue;
    expandedValue.tensor = newOp->getResult(newIdx++);
    expandedValue.dynamicDims =
        newOp->getResults().slice(newIdx, tensorType.getNumDynamicDims());
    newIdx += expandedValue.dynamicDims.size();
    tensorDimMap[expandedValue.tensor] = expandedValue;
    auto tieShapeOp = builder.create<IREE::Flow::TensorTieShapeOp>(
        op->getLoc(), expandedValue.tensor.getType(), expandedValue.tensor,
        expandedValue.dynamicDims);
    oldResult.replaceAllUsesExcept(tieShapeOp.getResult(), tieShapeOp);
  }
}

// Moves tensor dims from global stores to loads.
// Requires that the ExpandGlobalStoreOp pattern performs the stores.
//
// Example:
//  %0 = util.global.load @foo : tensor<?xf32>
//  ->
//  %0 = util.global.load @foo : tensor<?xf32>
//  %d = util.global.load @foo__d0 : index
//  %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
static void expandGlobalLoadOp(IREE::Util::GlobalLoadOpInterface op,
                               ExpandedGlobalMap &globalMap, IndexSet &indexSet,
                               TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto &expandedGlobal = globalMap[op.getGlobalName()];
  ExpandedValue expandedValue;
  expandedValue.tensor = op.getLoadedGlobalValue();
  expandedValue.dynamicDims.reserve(expandedGlobal.dynamicDimOps.size());
  for (auto dimOp : expandedGlobal.dynamicDimOps) {
    expandedValue.dynamicDims.push_back(
        dimOp.createLoadOp(op.getLoc(), builder).getLoadedGlobalValue());
  }
  tensorDimMap[op.getLoadedGlobalValue()] = expandedValue;
  auto tieShapeOp = builder.create<IREE::Flow::TensorTieShapeOp>(
      op.getLoc(), expandedValue.tensor.getType(), expandedValue.tensor,
      expandedValue.dynamicDims);
  op.getLoadedGlobalValue().replaceAllUsesExcept(tieShapeOp.getResult(),
                                                 tieShapeOp);
}

// Moves tensor dims from global stores to loads.
// Requires that the ExpandGlobalLoadOp pattern performs the loads.
//
// Example:
//  %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
//  util.global.store %1, @foo : tensor<?xf32>
//  ->
//  util.global.store %0, @foo : tensor<?xf32>
//  util.global.store %d, @foo__d0 : index
static void expandGlobalStoreOp(IREE::Util::GlobalStoreOpInterface op,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet,
                                TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto expandedValue = consumeExpandedValue(
      op.getLoc(), op.getStoredGlobalValue(), tensorDimMap, indexSet, builder);
  auto &expandedGlobal = globalMap[op.getGlobalName()];
  expandedGlobal.tensorOp.createStoreOp(op.getLoc(), expandedValue.tensor,
                                        builder);
  for (auto [valueDynamicDims, globalDynamicDimOps] : llvm::zip_equal(
           expandedValue.dynamicDims, expandedGlobal.dynamicDimOps)) {
    globalDynamicDimOps.createStoreOp(op.getLoc(), valueDynamicDims, builder);
  }
  op.erase();
}

static void expandInitializerOp(IREE::Util::InitializerOp op,
                                ExpandedGlobalMap &globalMap,
                                IndexSet &indexSet,
                                TensorDimMap &tensorDimMap) {
  expandRegion(op.getRegion(), globalMap, indexSet, tensorDimMap);
}

// Inserts dimension associate reshapes on tensor arguments.
// Requires that the ExpandCallOp/ExpandReturnOp patterns handle passing dims.
//
// Example:
//  func.func @foo(%0: tensor<?xf32>)
//  ->
//  func.func @foo(%0: tensor<?xf32>, %d: index) {
//    %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
static void expandFuncOp(mlir::func::FuncOp op, ExpandedGlobalMap &globalMap,
                         IndexSet &indexSet, TensorDimMap &tensorDimMap) {
  auto oldType = op.getFunctionType();
  auto inputTypes = expandTypes(oldType.getInputs());
  auto resultTypes = expandTypes(oldType.getResults());
  auto newType = FunctionType::get(op.getContext(), inputTypes, resultTypes);
  if (newType != oldType) {
    op.setType(newType);
  }
  expandRegion(op.getRegion(), globalMap, indexSet, tensorDimMap);
}

// Splits tensor operands and results into (tensor, dynamic dims...).
// Requires that the ExpandFuncOp/ExpandReturnOp patterns handle passing dims.
//
// Example:
//  %a = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
//  %r = call @foo(%a)
//  ->
//  %r, %rd = call @foo(%a, %ad)
//  %2 = flow.tensor.tie_shape %r : tensor<?xf32>{%rd}
static void expandCallOp(mlir::func::CallOp op, IndexSet &indexSet,
                         TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;

  // Build the new call op with expanded operands and results.
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(), tensorDimMap,
                                 indexSet, builder);
  auto resultTypes = expandTypes(op.getResultTypes());
  auto newOp = builder.create<mlir::func::CallOp>(op.getLoc(), op.getCallee(),
                                                  resultTypes, operands);

  retieResults(op, newOp, tensorDimMap);
  op.erase();
}

// Moves dynamic dims to callers upon return.
// Requires that the ExpandFuncOp/ExpandCallOp patterns handle passing dims.
//
// Example:
//  %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
//  return %1
//  ->
//  return %0, %d
static void expandReturnOp(mlir::func::ReturnOp op, IndexSet &indexSet,
                           TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(), tensorDimMap,
                                 indexSet, builder);
  builder.create<mlir::func::ReturnOp>(op.getLoc(), operands);
  op.erase();
}

// Moves dynamic dims across branches.
// Requires that the ExpandFuncOp pattern handles modifying the block args.
//
// Example:
//    %1 = flow.tensor.tie_shape %0 : tensor<?xf32>{%d}
//    br ^bb1(%1)
//  ^bb1(%b):
//  ->
//    br ^bb1(%0, %d)
//  ^bb1(%arg0, %arg1):
//    %1 = flow.tensor.tie_shape %arg0 : tensor<?xf32>{%arg1}
static void expandBranchOp(mlir::cf::BranchOp op, IndexSet &indexSet,
                           TensorDimMap &tensorDimMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getDestOperands(),
                                 tensorDimMap, indexSet, builder);
  builder.create<mlir::cf::BranchOp>(op.getLoc(), op.getDest(), operands);
  op.erase();
}

static void expandCondBranchOp(mlir::cf::CondBranchOp op, IndexSet &indexSet,
                               TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;
  OpBuilder builder(op);
  builder.create<mlir::cf::CondBranchOp>(
      op.getLoc(), op.getCondition(), op.getTrueDest(),
      expandOperands(op.getLoc(), op.getTrueDestOperands(), tensorDimMap,
                     indexSet, builder),
      op.getFalseDest(),
      expandOperands(op.getLoc(), op.getFalseDestOperands(), tensorDimMap,
                     indexSet, builder));
  op.erase();
}

// Expands select ops on tensors to also select on the dynamic dims.
//
// Example:
//   %0 = flow.tensor.tie_shape %arg0 : tensor<?xf32>{%d0}
//   %1 = flow.tensor.tie_shape %arg1 : tensor<?xf32>{%d1}
//   %2 = arith.select %cond, %0, %1 : tensor<?xf32>
//  ->
//   %2 = arith.select %cond, %0, %1 : tensor<?xf32>
//   %3 = arith.select %cond, %d0, %d1 : index
//   %4 = flow.tensor.tie_shape %2 : tensor<?xf32>{%3}
static void expandSelectOp(mlir::arith::SelectOp op, IndexSet &indexSet,
                           TensorDimMap &tensorDimMap) {
  if (!usesDynamicTensors(op))
    return;
  OpBuilder builder(op);

  auto trueValue = consumeExpandedValue(op.getLoc(), op.getTrueValue(),
                                        tensorDimMap, indexSet, builder);
  auto falseValue = consumeExpandedValue(op.getLoc(), op.getFalseValue(),
                                         tensorDimMap, indexSet, builder);

  auto selectOp = builder.create<mlir::arith::SelectOp>(
      op.getLoc(), op.getCondition(), op.getTrueValue(), op.getFalseValue());

  SmallVector<Value> selectedDims;
  for (auto [trueDynamicDims, falseDynamicDims] :
       llvm::zip_equal(trueValue.dynamicDims, falseValue.dynamicDims)) {
    selectedDims.push_back(
        builder
            .create<arith::SelectOp>(op.getLoc(), op.getCondition(),
                                     trueDynamicDims, falseDynamicDims)
            .getResult());
  }
  auto tieShapeOp = builder.create<IREE::Flow::TensorTieShapeOp>(
      selectOp.getLoc(), selectOp.getResult().getType(), selectOp.getResult(),
      selectedDims);

  op.getResult().replaceAllUsesExcept(tieShapeOp.getResult(), tieShapeOp);
  op.erase();
}

static void expandWhileOp(mlir::scf::WhileOp op, ExpandedGlobalMap &globalMap,
                          IndexSet &indexSet, TensorDimMap &tensorDimMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(), tensorDimMap,
                                 indexSet, builder);
  auto resultTypes = expandTypes(op.getResultTypes());

  auto newOp = builder.create<scf::WhileOp>(op.getLoc(), resultTypes, operands,
                                            /*beforeBody*/ nullptr,
                                            /*afterBody*/ nullptr);

  newOp.getBefore().takeBody(op.getBefore());
  newOp.getAfter().takeBody(op.getAfter());

  expandRegion(newOp.getBefore(), globalMap, indexSet, tensorDimMap);
  expandRegion(newOp.getAfter(), globalMap, indexSet, tensorDimMap);
  retieResults(op, newOp, tensorDimMap);
  op.erase();
}

static void expandIfOp(mlir::scf::IfOp op, ExpandedGlobalMap &globalMap,
                       IndexSet &indexSet, TensorDimMap &tensorDimMap) {
  OpBuilder builder(op);
  auto resultTypes = expandTypes(op.getResultTypes());

  auto newOp = builder.create<scf::IfOp>(
      op.getLoc(), resultTypes, op.getOperand(), op.elseBlock() != nullptr);

  newOp.getBodyRegion().takeBody(op.getBodyRegion());
  expandRegion(newOp.getBodyRegion(), globalMap, indexSet, tensorDimMap);

  if (newOp.elseBlock()) {
    newOp.getElseRegion().takeBody(op.getElseRegion());
    expandRegion(newOp.getElseRegion(), globalMap, indexSet, tensorDimMap);
  }

  retieResults(op, newOp, tensorDimMap);
  op.erase();
}

static void expandScfYieldOp(mlir::scf::YieldOp op, IndexSet &indexSet,
                             TensorDimMap &tensorDimMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getOperands(), tensorDimMap,
                                 indexSet, builder);
  builder.create<mlir::scf::YieldOp>(op.getLoc(), operands);
  op.erase();
}

static void expandScfConditionOp(mlir::scf::ConditionOp op, IndexSet &indexSet,
                                 TensorDimMap &tensorDimMap) {
  OpBuilder builder(op);
  auto operands = expandOperands(op.getLoc(), op.getArgs(), tensorDimMap,
                                 indexSet, builder);
  builder.create<mlir::scf::ConditionOp>(op.getLoc(), op.getCondition(),
                                         operands);
  op.erase();
}

// Recursively expands tensors into (tensor, dynamic dims...) in |op|.
static void expandTensorDims(Operation *op, ExpandedGlobalMap &globalMap,
                             IndexSet &indexSet, TensorDimMap &tensorDimMap) {
  if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
    expandGlobalLoadOp(loadOp, globalMap, indexSet, tensorDimMap);
  } else if (auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
    expandGlobalStoreOp(storeOp, globalMap, indexSet, tensorDimMap);
  } else if (auto initializerOp = dyn_cast<IREE::Util::InitializerOp>(op)) {
    expandInitializerOp(initializerOp, globalMap, indexSet, tensorDimMap);
  } else if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
    expandFuncOp(funcOp, globalMap, indexSet, tensorDimMap);
  } else if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
    expandCallOp(callOp, indexSet, tensorDimMap);
  } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    expandReturnOp(returnOp, indexSet, tensorDimMap);
  } else if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(op)) {
    expandBranchOp(branchOp, indexSet, tensorDimMap);
  } else if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
    expandCondBranchOp(condBranchOp, indexSet, tensorDimMap);
  } else if (auto selectOp = dyn_cast<mlir::arith::SelectOp>(op)) {
    expandSelectOp(selectOp, indexSet, tensorDimMap);
  } else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
    expandWhileOp(whileOp, globalMap, indexSet, tensorDimMap);
  } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
    expandIfOp(ifOp, globalMap, indexSet, tensorDimMap);
  } else if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
    expandScfYieldOp(yieldOp, indexSet, tensorDimMap);
  } else if (auto conditionOp = dyn_cast<mlir::scf::ConditionOp>(op)) {
    expandScfConditionOp(conditionOp, indexSet, tensorDimMap);
  }
}

//===----------------------------------------------------------------------===//
// -iree-global-opt-expand-tensor-shapes
//===----------------------------------------------------------------------===//

// This does a relatively mechanical transformation of a module to expand all
// tensor values (and globals) into (tensor, dynamic dims...) tuples.
//
// This is designed to be composed with generic optimization passes like global
// fusion/folding and IPO and as such performs all transformations locally. For
// example, calls are always updated to take/return dynamic dimensions and
// results are always wrapped in a flow.tensor.tie_shape, with the
// elision/deduplication/etc left until cleanup.
class ExpandTensorShapesPass
    : public ExpandTensorShapesBase<ExpandTensorShapesPass> {
public:
  ExpandTensorShapesPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto rootOp = getOperation();

    // Expand all util.global ops holding tensor into tensor + dynamic dims.
    auto globalMap = expandGlobalTensorDims(rootOp);

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
      TensorDimMap tensorDimMap;
      expandTensorDims(callableOp, globalMap, indexSet, tensorDimMap);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createExpandTensorShapesPass() {
  return std::make_unique<ExpandTensorShapesPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
