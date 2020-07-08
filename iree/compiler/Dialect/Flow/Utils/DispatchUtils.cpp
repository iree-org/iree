// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

bool isOpOfKnownDialect(Operation *op) {
  if (!op->getDialect()) return false;
  // TODO(benvanik): replace with op dispatchability interface to allow dialects
  // to opt into dispatch.
  auto dialectNamespace = op->getDialect()->getNamespace();
  return dialectNamespace == mhlo::MhloDialect::getDialectNamespace() ||
         dialectNamespace == mlir::StandardOpsDialect::getDialectNamespace() ||
         dialectNamespace == FlowDialect::getDialectNamespace() ||
         dialectNamespace == ShapeDialect::getDialectNamespace();
}

namespace {

// Returns the set of values that must be captured for use by |ops| and the
// set of values defined by |ops| that are used outside of the set.
LogicalResult analyzeOpRangeValues(ArrayRef<Operation *> ops,
                                   llvm::SetVector<Value> *capturedValues,
                                   llvm::SetVector<Value> *escapingValues) {
  llvm::SmallDenseSet<Operation *> opSet;
  opSet.reserve(ops.size());
  opSet.insert(ops.begin(), ops.end());
  for (auto *op : ops) {
    for (auto value : op->getOperands()) {
      if (!llvm::is_contained(opSet, value.getDefiningOp())) {
        // Op is using a value not in the ops set, ensure we capture it.
        capturedValues->insert(value);
      }
    }
    for (auto value : op->getResults()) {
      for (auto &use : value.getUses()) {
        if (!llvm::is_contained(opSet, use.getOwner())) {
          // An op outside of the ops set is using the value, needs to escape.
          escapingValues->insert(value);
          continue;
        }
      }
    }
  }
  return success();
}

}  // namespace

LogicalResult buildDispatchRegion(Block *parentBlock, Value workload,
                                  ArrayRef<Operation *> ops) {
  // Fused location with all ops.
  SmallVector<Location, 16> opLocs;
  for (auto *op : ops) {
    opLocs.push_back(op->getLoc());
  }
  auto regionLoc = FusedLoc::get(opLocs, workload.getContext());

  // Get a list of values that we need to capture and values that escape the
  // region and need to be returned.
  llvm::SetVector<Value> capturedValues;
  llvm::SetVector<Value> escapingValues;
  if (failed(analyzeOpRangeValues(ops, &capturedValues, &escapingValues))) {
    return failure();
  }
  SmallVector<Type, 8> escapingTypes;
  for (auto value : escapingValues) escapingTypes.push_back(value.getType());

  // Build the region op and add it to the parent block.
  OpBuilder parentBuilder = OpBuilder::atBlockEnd(parentBlock);
  parentBuilder.setInsertionPoint(ops.back());
  auto dispatchRegionOp = parentBuilder.create<IREE::Flow::DispatchRegionOp>(
      regionLoc, escapingTypes, workload, capturedValues.getArrayRef());

  // Create the block and setup the arg mapping for captured values.
  auto *regionBlock = new Block();
  dispatchRegionOp.body().push_back(regionBlock);
  OpBuilder regionBuilder = OpBuilder::atBlockEnd(regionBlock);
  BlockAndValueMapping mapping;
  for (auto capturedValue : capturedValues) {
    auto blockArg = regionBlock->addArgument(capturedValue.getType());
    mapping.map(capturedValue, blockArg);
  }

  // Clone ops into the new region block.
  for (auto *op : ops) {
    // Note that this updates the mapping with the new values (so at the end
    // we have those new values).
    regionBuilder.clone(*op, mapping);
  }

  // Return results (as we need a terminator in our block).
  // These are all of the values that escape our region.
  SmallVector<Value, 8> resultValues;
  for (auto oldValue : escapingValues) {
    resultValues.push_back(mapping.lookupOrDefault(oldValue));
  }
  regionBuilder.create<IREE::Flow::ReturnOp>(opLocs.back(), resultValues);

  // Replace usage of values with the results of the region.
  for (int i = 0; i < escapingValues.size(); ++i) {
    escapingValues[i].replaceAllUsesWith(dispatchRegionOp.getResult(i));
  }

  // Remove original ops from the parent region.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    (*it)->erase();
  }

  return success();
}

namespace {

// Recursively finds all reachable functions from the given |rootFunc| and adds
// them to the |reachableFuncs| set.
//
// Note that indirect calls are not supported, however we don't allow those in
// dispatch regions anyway so they should not be present here.
LogicalResult findReachableFunctions(
    FuncOp rootFuncOp, llvm::SetVector<FuncOp> &reachableFuncs,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  llvm::SetVector<FuncOp> worklist;
  worklist.insert(rootFuncOp);
  while (!worklist.empty()) {
    auto funcOp = worklist.pop_back_val();
    funcOp.walk([&](CallOp callOp) {
      auto calleeOp = dispatchableFuncOps.find(callOp.callee())->second;
      if (reachableFuncs.insert(calleeOp)) {
        worklist.insert(calleeOp);
      }
    });
  }
  return success();
}

}  // namespace

ExecutableOp createExecutable(Location loc, StringRef executableName,
                              ArrayRef<FuncOp> funcOps, ModuleOp parentModuleOp,
                              llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  assert(!funcOps.empty() && "must have at least one entry function");

  // Gather all reachable functions.
  llvm::SetVector<FuncOp> reachableFuncs;
  for (auto funcOp : funcOps) {
    findReachableFunctions(funcOp, reachableFuncs, dispatchableFuncOps);
  }

  // Create the executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  auto executableOp =
      parentModuleBuilder.create<IREE::Flow::ExecutableOp>(loc, executableName);

  // Create the inner ModuleOp that contains the original functions. We need
  // to provide this shim as some ops (like std.call) look for the
  // containing module to provide symbol resolution.
  OpBuilder executableBuilder(executableOp);
  executableBuilder.setInsertionPointToStart(&executableOp.getBlock());
  auto innerModule = executableBuilder.create<ModuleOp>(loc);
  for (auto funcOp : funcOps) {
    innerModule.push_back(funcOp);
  }

  // Copy all reachable functions into the executable.
  // Linker passes may dedupe these later on.
  OpBuilder innerModuleBuilder = OpBuilder::atBlockEnd(innerModule.getBody());
  innerModuleBuilder.setInsertionPoint(innerModule.getBody(),
                                       ++innerModule.getBody()->begin());
  for (auto reachableFunc : reachableFuncs) {
    innerModuleBuilder.clone(*reachableFunc);
  }

  return executableOp;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
